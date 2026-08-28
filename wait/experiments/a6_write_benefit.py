import os
import statistics as st
from dataclasses import dataclass

from wait import lustre, probe
from wait.layout import KIB, MIB, dom, plain

SIZES_KIB = (4, 16, 64)
FILES = 200

repeats = 3


@dataclass(frozen=True)
class Cell:
    tier: str
    size_bytes: int
    files: int = FILES


cells = [Cell(tier, kib * KIB) for tier in ("dom", "ost") for kib in SIZES_KIB]


def layout_for(tier):
    return dom(128 * KIB, 1, MIB) if tier == "dom" else plain(1, MIB)


def prepare(cell, workdir):
    lustre.setstripe(workdir, layout_for(cell.tier))


def measure(cell, workdir):
    payload = b"x" * cell.size_bytes
    staged = [probe.write_staged(os.path.join(workdir, "c%d" % i), payload)
              for i in range(cell.files)]
    granted = lustre.getstripe(os.path.join(workdir, "c0"))

    row = {"files_written": len(staged), "objects": granted.objects_allocated}
    for stage in ("create_ns", "write_ns", "fsync_ns", "durable_ns", "total_ns"):
        values = [s[stage] for s in staged]
        row["%s_p50" % stage] = st.median(values)
        row["%s_p90" % stage] = probe.nearest_rank(values, 0.90)
        row["%s_sum" % stage] = sum(values)
    return row
