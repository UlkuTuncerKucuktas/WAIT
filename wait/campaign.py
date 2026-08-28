import importlib
import os
import shutil
import socket
import subprocess

from wait import ledger, lustre, ranks
from wait.probe import now_ns

PROVENANCE = "provenance.json"


class PhaseError(RuntimeError):
    pass


def git_rev():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def env_hash(params):
    import hashlib
    import json
    text = json.dumps(params, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def workdir(base, experiment, key):
    # Re-reading one file set understates variance about threefold: it holds OST
    # assignment and server cache state fixed across repeats.
    path = os.path.join(base, experiment, key)
    os.makedirs(path, exist_ok=True)
    return path


def _provenance_path(work):
    return os.path.join(work, PROVENANCE)


def write_provenance(work, extra=None):
    import json
    record = {"host": socket.gethostname(), "git_rev": git_rev(),
              "written_ns": now_ns()}
    record.update(extra or {})
    with open(_provenance_path(work), "w") as fh:
        json.dump(record, fh)
    return record


def read_provenance(work):
    import json
    path = _provenance_path(work)
    if not os.path.exists(path):
        raise PhaseError("no prepare phase ran in %s" % work)
    with open(path) as fh:
        return json.load(fh)


def check_reader_is_not_the_writer(work, same_node_allowed):
    prepared = read_provenance(work)
    here = socket.gethostname()
    # Reading on the writer's own node measured DoM at 78.1 us against OST at
    # 77.8 -- no difference at all, which reads as a clean result rather than a
    # broken setup.  A1 declares same_node deliberately; nothing else may.
    if prepared["host"] == here and not same_node_allowed:
        raise PhaseError("reader and writer are both %s" % here)
    return prepared["host"], here


def run(module_name, phase, base, ledger_path):
    experiment = importlib.import_module(module_name)
    name = module_name.rsplit(".", 1)[-1]
    finished = ledger.done(ledger_path)
    params = lustre.client_params()
    stamp = env_hash(params)

    for repeat in range(experiment.repeats):
        for cell in experiment.cells:
            scale = ranks.world() if ranks.world() > 1 else None
            key = ledger.key(name, cell, repeat, getattr(cell, "arm", None), scale)
            if phase == "measure" and key in finished:
                continue
            work = workdir(base, name, key)
            if phase == "prepare":
                experiment.prepare(cell, work)
                lustre.flush(work)
                write_provenance(work, {"repeat": repeat, "env_hash": stamp})
                continue
            _measure_one(experiment, name, cell, repeat, key, work,
                         stamp, ledger_path, scale)


def _measure_one(experiment, name, cell, repeat, key, work, stamp, ledger_path,
                 scale=None):
    prepared = read_provenance(work)
    same_node = bool(getattr(cell, "same_node", False))
    writer, reader = check_reader_is_not_the_writer(work, same_node)

    row = {"key": key, "experiment": name, "repeat": repeat, "scale": scale,
           "arm": getattr(cell, "arm", None), "prepare_host": writer,
           "measure_host": reader, "git_rev": git_rev(), "env_hash": stamp,
           "measured_ns": now_ns()}
    row.update({"cell_%s" % k: v for k, v in ledger.cell_fields(cell).items()})

    before = lustre.counters()
    try:
        row.update(experiment.measure(cell, work))
    except Exception as exc:                     # a failure is a row, not a gap
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
    after = lustre.counters()
    delta = after - before
    row.update({"ost_bulk_rpcs": delta.ost_bulk_rpcs,
                "mdc_intent_locks": delta.mdc_intent_locks,
                "osc_cached_mb_before": before.osc_cached_mb,
                "osc_cached_mb_after": after.osc_cached_mb})
    # Every task runs measure so it can take part in the barrier; one of them
    # records.  Appending from all would write the same cell N times.
    if ranks.rank() == 0:
        ledger.append(ledger_path, row)

    # Keep a failed cell's directory: it is the only forensic evidence there is.
    if "error" not in row and ranks.rank() == 0:
        shutil.rmtree(work, ignore_errors=True)


def reap(base, ledger_path):
    keys = {r["key"] for r in ledger.rows(ledger_path)}
    removed = 0
    for experiment in os.listdir(base) if os.path.isdir(base) else []:
        root = os.path.join(base, experiment)
        for key in os.listdir(root):
            if key not in keys:
                shutil.rmtree(os.path.join(root, key), ignore_errors=True)
                removed += 1
    return removed
