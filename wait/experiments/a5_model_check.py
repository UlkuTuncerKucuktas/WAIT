import json
import os
import statistics as st
from dataclasses import dataclass

from wait import lustre, probe
from wait.layout import KIB, MIB, Component, Layout, Tier, plain
from wait.model import Constants, predict_fstat_ns

CONSTANTS = "constants.json"
FILES = 100

repeats = 3

ADVISOR = Layout((Component(128 * KIB, Tier.MDT),
                  Component(8 * MIB, Tier.OST, 1, MIB, "flash"),
                  Component(None, Tier.OST, 8, 16 * MIB, "disk")))
CASCADE = Layout((Component(4 * MIB, Tier.OST, 1, MIB, "flash"),
                  Component(None, Tier.OST, 4, 4 * MIB, "flash")))
WIDE = plain(16, 4 * MIB)

LAYOUTS = {"advisor": ADVISOR, "cascade": CASCADE, "wide": WIDE}


@dataclass(frozen=True)
class Cell:
    layout: str
    size_bytes: int
    files: int = FILES
    chunk_bytes: int = MIB


# A2 sweeps 4 to 256 KiB and instantiates no component past the first, so these
# sizes leave every cell here untested.
SIZES_KIB = (512, 2048, 16384)

cells = [Cell(name, kib * KIB) for name in LAYOUTS for kib in SIZES_KIB]


def layout_for(cell):
    return LAYOUTS[cell.layout]


def constants():
    if not os.path.exists(CONSTANTS):
        return None
    with open(CONSTANTS) as fh:
        return Constants(**json.load(fh))


def prepare(cell, workdir):
    lustre.setstripe(workdir, layout_for(cell))
    probe.write_files(workdir, cell.files, cell.size_bytes)


def measure(cell, workdir):
    granted = lustre.getstripe(os.path.join(workdir, "f0"))
    staged = [probe.read_staged(os.path.join(workdir, "f%d" % i),
                                cell.chunk_bytes) for i in range(cell.files)]
    fstat = [s["fstat_ns"] for s in staged]

    row = {"files_read": len(staged),
           "granted_objects": granted.objects_allocated,
           "granted_instantiated": sum(c.instantiated for c in granted.components),
           "fstat_ns_p50": st.median(fstat),
           "fstat_ns_p90": probe.nearest_rank(fstat, 0.90),
           "total_ns_p50": st.median([s["total_ns"] for s in staged])}

    known = constants()
    if known:
        row["predicted_objects"] = predict_fstat_ns(
            layout_for(cell), cell.size_bytes, 0, 1)
        row["predicted_fstat_ns"] = predict_fstat_ns(
            layout_for(cell), cell.size_bytes,
            known.base_fstat_ns, known.per_object_fstat_ns)
    return row
