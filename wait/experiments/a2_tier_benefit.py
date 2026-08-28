import os
import statistics as st
from dataclasses import dataclass

from wait import lustre, probe
from wait.layout import KIB, MIB, Component, Layout, Tier, dom, plain

SIZES_KIB = (4, 8, 12, 16, 32, 64, 80, 96, 112, 128, 160, 256)
WIDTHS = (1, 8, 24)
FILES = 100
WIDTH_SIZE_BYTES = MIB

repeats = 3


@dataclass(frozen=True)
class Cell:
    arm: str
    size_bytes: int
    stripe_count: int = 1
    files: int = FILES
    chunk_bytes: int = MIB


cells = [Cell(arm, kib * KIB)
         for arm in ("dom2", "dom3", "ost")
         for kib in SIZES_KIB] + [
        Cell("width", WIDTH_SIZE_BYTES, stripe_count=c) for c in WIDTHS]


def layout_for(cell):
    if cell.arm == "dom2":
        return dom(128 * KIB, 1, MIB)
    if cell.arm == "dom3":
        # A third component carries a larger layout descriptor in the same reply
        # buffer the inline data rides in.
        return Layout((Component(128 * KIB, Tier.MDT),
                       Component(8 * MIB, Tier.OST, 1, MIB, "flash"),
                       Component(None, Tier.OST, 8, 16 * MIB, "disk")))
    return plain(cell.stripe_count, MIB)


def prepare(cell, workdir):
    lustre.setstripe(workdir, layout_for(cell))
    probe.write_files(workdir, cell.files, cell.size_bytes)


def measure(cell, workdir):
    granted = lustre.getstripe(os.path.join(workdir, "f0"))
    staged = [probe.read_staged(os.path.join(workdir, "f%d" % i),
                                cell.chunk_bytes) for i in range(cell.files)]
    # A second pass over the same files reports what the client cache serves, so
    # the first-pass figure is the only cold one.
    warm = [probe.read_staged(os.path.join(workdir, "f%d" % i), cell.chunk_bytes)
            for i in range(cell.files)]

    row = {"files_read": len(staged), "granted_objects": granted.objects_allocated,
           "granted_dom_extent": granted.dom_extent_bytes,
           "granted_components": len(granted.components),
           "granted_instantiated": sum(c.instantiated for c in granted.components)}
    for stage in ("open_ns", "fstat_ns", "first_read_ns", "rest_read_ns", "total_ns"):
        cold = [s[stage] for s in staged]
        row["%s_p50" % stage] = st.median(cold)
        row["%s_p90" % stage] = probe.nearest_rank(cold, 0.90)
    row["warm_total_ns_p50"] = st.median([s["total_ns"] for s in warm])
    return row
