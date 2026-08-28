import os
import statistics as st
from dataclasses import dataclass

from wait import lustre, probe
from wait.layout import KIB, MIB, dom, plain

SIZE_BYTES = 64 * KIB
FILES = 100
SHM = "/dev/shm"

repeats = 3


@dataclass(frozen=True)
class Cell:
    arm: str
    tier: str
    size_bytes: int = SIZE_BYTES
    files: int = FILES
    chunk_bytes: int = MIB


cells = [Cell(arm, tier)
         for arm in ("right", "wrong")
         for tier in ("dom", "ost")] + [Cell("floor", "shm")]


def layout_for(tier):
    return dom(128 * KIB, 1, MIB) if tier == "dom" else plain(1, MIB)


def prepare(cell, workdir):
    # The wrong and floor arms write inside measure, on the node that reads them:
    # that is the condition they exist to demonstrate.
    if cell.arm != "right":
        return
    lustre.setstripe(workdir, layout_for(cell.tier))
    probe.write_files(workdir, cell.files, cell.size_bytes)


def measure(cell, workdir):
    if cell.arm == "floor":
        directory = os.path.join(SHM, "wait_a1_%d" % os.getpid())
        os.makedirs(directory, exist_ok=True)
        probe.write_files(directory, cell.files, cell.size_bytes)
    elif cell.arm == "wrong":
        directory = workdir
        lustre.setstripe(directory, layout_for(cell.tier))
        probe.write_files(directory, cell.files, cell.size_bytes)
    else:
        directory = workdir

    staged = [probe.read_staged(os.path.join(directory, "f%d" % i),
                                cell.chunk_bytes) for i in range(cell.files)]
    if cell.arm == "floor":
        for i in range(cell.files):
            os.unlink(os.path.join(directory, "f%d" % i))
        os.rmdir(directory)

    row = {"files_read": len(staged)}
    for stage in ("open_ns", "fstat_ns", "first_read_ns", "rest_read_ns", "total_ns"):
        values = [s[stage] for s in staged]
        row["%s_p50" % stage] = st.median(values)
        row["%s_p90" % stage] = probe.nearest_rank(values, 0.90)
    return row
