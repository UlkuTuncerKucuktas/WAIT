import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

CLIENT_PARAMS = (
    "llite.*.max_read_ahead_mb", "llite.*.max_read_ahead_per_file_mb",
    "llite.*.max_read_ahead_whole_mb", "llite.*.statahead_max",
    "osc.*.max_rpcs_in_flight", "osc.*.max_pages_per_rpc",
    "osc.*.short_io_bytes", "mdc.*.mdc_dom_min_repsize",
)


class LustreError(RuntimeError):
    pass


def _run(argv):
    # A missing binary means this is not a Lustre client.  The pure half of the
    # harness runs on a laptop, where absent counters are a recorded fact.
    try:
        return subprocess.run(argv, capture_output=True, text=True)
    except FileNotFoundError:
        return subprocess.CompletedProcess(argv, 127, "", "%s not found" % argv[0])


@dataclass(frozen=True)
class GrantedComponent:
    start_bytes: int
    end_bytes: Optional[int]
    stripe_count: int
    stripe_bytes: int
    pattern: str
    pool: Optional[str]
    instantiated: bool
    objects: int

    @property
    def is_mdt(self):
        return "mdt" in self.pattern


@dataclass(frozen=True)
class Granted:
    composite: bool
    components: Tuple[GrantedComponent, ...]
    raw: str

    @property
    def is_dom(self):
        return bool(self.components) and self.components[0].is_mdt

    @property
    def dom_extent_bytes(self):
        return self.components[0].end_bytes if self.is_dom else 0

    @property
    def objects_allocated(self):
        return sum(c.objects for c in self.components if c.instantiated)


def _int(text, field, default=None):
    m = re.search(r"%s:\s+(-?\d+)" % re.escape(field), text)
    return int(m.group(1)) if m else default


def _pool(text):
    m = re.search(r"lmm_pool:\s+(\S+)", text)
    return m.group(1) if m else None


def _objects(text, stripe_count):
    # Two formats.  A plain layout prints a four-column obdidx table; a composite
    # one prints "lmm_objects:" with "- N: { l_ost_idx: ... }" lines, so a parser
    # counting obdidx rows reports 0 objects for every PFL file.
    rows = sum(1 for l in text.splitlines()
               if len(l.split()) == 4 and l.split()[0].isdigit())
    listed = len(re.findall(r"l_ost_idx:", text))
    return rows or listed or (stripe_count if stripe_count > 0 else 0)


def parse_getstripe(raw):
    if "lcm_entry_count" not in raw:
        count = _int(raw, "lmm_stripe_count", 0)
        comp = GrantedComponent(
            start_bytes=0, end_bytes=None, stripe_count=count,
            stripe_bytes=_int(raw, "lmm_stripe_size", 0),
            pattern=(re.search(r"lmm_pattern:\s+(\S+)", raw) or [None, ""])[1]
            if re.search(r"lmm_pattern:\s+(\S+)", raw) else "",
            pool=_pool(raw), instantiated=True, objects=_objects(raw, count))
        return Granted(False, (comp,), raw)

    blocks = re.split(r"\n(?=[ \t]*lcme_id:)", raw)[1:]
    components = []
    for b in blocks:
        end = None if re.search(r"e_end:\s+EOF", b) else _int(b, "lcme_extent.e_end")
        count = _int(b, "lmm_stripe_count", 0)
        pattern = re.search(r"lmm_pattern:\s+(\S+)", b)
        components.append(GrantedComponent(
            start_bytes=_int(b, "lcme_extent.e_start", 0), end_bytes=end,
            stripe_count=count, stripe_bytes=_int(b, "lmm_stripe_size", 0),
            pattern=pattern.group(1) if pattern else "", pool=_pool(b),
            instantiated=bool(re.search(r"lcme_flags:\s+init\b", b)),
            objects=_objects(b, 0) if "mdt" not in (pattern.group(1) if pattern else "") else 0))
    return Granted(True, tuple(components), raw)


def getstripe(path):
    r = _run(["lfs", "getstripe", str(path)])
    if r.returncode:
        raise LustreError("getstripe %s: %s" % (path, r.stderr.strip()))
    return parse_getstripe(r.stdout)


def setstripe(path, layout):
    # Never trust the request.  A value above a server-side limit is silently
    # truncated, not refused: -E 2M -L mdt comes back granted as 1 MiB.
    r = _run(["lfs", "setstripe"] + layout.spec().split() + [str(path)])
    if r.returncode:
        raise LustreError("setstripe %s: %s" % (layout.spec(), r.stderr.strip()))
    return getstripe(path)


def create_with_layout(path, layout, payload):
    # For a class of files that needs its own layout inside a directory holding
    # other classes, since a directory carries one default.
    #
    # setstripe creates the file, so the write must not create or truncate it.
    # Truncating a file whose DoM component setstripe has already instantiated
    # leaves it on the MDT and silently un-inlined: getstripe still reports DoM
    # and the read still issues no OST bulk RPC, so every structural check
    # passes, while the read costs 271 us instead of 7 -- the whole benefit,
    # gone with no symptom.  Measured five ways: directory default 6.6 us,
    # create-close-reopen 6.5, setstripe then O_WRONLY 11.6, setstripe then
    # truncate 271.0, plain OST 355.3.
    setstripe(path, layout)
    fd = os.open(path, os.O_WRONLY)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def flush_paths(paths):
    # One file per invocation: `lfs data_version -w a b c` prints a single
    # version and flushes only the first, so a batched call would silently leave
    # the rest warm.  Parallel instead of batched -- sixteen files cost about ten
    # milliseconds at -P 32 and a hundred and sixty serially.
    if not paths:
        return
    subprocess.run("xargs -0 -P 32 -n 1 lfs data_version -w > /dev/null",
                   input="\0".join(str(p) for p in paths) + "\0",
                   shell=True, text=True, capture_output=True)


def flush(directory):
    # Closing a file releases the write lock but leaves pages warm enough to
    # erase the effect: without this DoM inflates 4.17x and OST only 1.65x, so it
    # does not cancel in a ratio.  Belongs in the write phase, never in a clock.
    subprocess.run(
        "find %s -type f -print0 | xargs -0 -P 32 -n 1 lfs data_version -w"
        % directory, shell=True, capture_output=True)


@dataclass(frozen=True)
class Counters:
    ost_bulk_rpcs: int
    mdc_intent_locks: int
    osc_cached_mb: int

    def __sub__(self, other):
        return Counters(self.ost_bulk_rpcs - other.ost_bulk_rpcs,
                        self.mdc_intent_locks - other.mdc_intent_locks,
                        self.osc_cached_mb - other.osc_cached_mb)


def _param(pattern):
    return _run(["lctl", "get_param", "-n", pattern]).stdout


def parse_rpc_stats(text):
    # rpc_stats holds several histograms -- pages per rpc, rpcs in flight,
    # offset -- and each one's rpcs column sums to the same RPC total, so
    # summing every row reports a multiple of the count.  Only the pages per
    # rpc block counts.  The text is every OSC's file concatenated, one block
    # per OSC, so the scan resumes after each block instead of stopping at the
    # first: stopping there counts one OSC out of forty-eight.
    total, inside = 0, False
    for line in text.splitlines():
        if "pages per rpc" in line:
            inside = True
            continue
        if not inside:
            continue
        head, sep, rest = line.partition(":")
        if not sep or not head.strip().isdigit() or "|" not in rest:
            inside = False
            continue
        total += int(rest.split("|")[0].split()[0])
    return total


def parse_md_stats(text, field="intent_lock"):
    # stat() registers as intent_lock, not getattr -- 300 distinct stats moved
    # intent_lock and left getattr at zero.
    total = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == field:
            total += int(parts[1])
    return total


def parse_cached_mb(text):
    return sum(int(m.group(1)) for m in re.finditer(r"used_mb:\s+(\d+)", text))


def counters():
    return Counters(parse_rpc_stats(_param("osc.*.rpc_stats")),
                    parse_md_stats(_param("mdc.*.md_stats")),
                    parse_cached_mb(_param("osc.*.osc_cached_mb")))


def parse_mdt_used_kib(text):
    return sum(int(l.split()[2]) for l in text.splitlines() if "[MDT:" in l)


def mdt_used_kib():
    return parse_mdt_used_kib(_run(["lfs", "df", "/arf"]).stdout)


def client_params():
    out = {}
    for p in CLIENT_PARAMS:
        values = _param(p).split()
        out[p.split(".")[-1]] = values[0] if values else None
    return out
