import json
import os
from dataclasses import dataclass

from wait import lustre, probe
from wait.layout import KIB, MIB, dom

EXTENTS_KIB = (64, 128, 256)
SIZES_KIB = (4, 16, 64, 256, 1024)
DRIFT_SECONDS = 15
READINGS = "mdt.json"

repeats = 2


@dataclass(frozen=True)
class Cell:
    extent_bytes: int
    size_bytes: int
    files: int


def files_for(size_bytes):
    # Enough files that the MDT delta clears background drift, without writing
    # gigabytes of OST data for the sizes that spill past the extent.
    return min(5000, max(500, 64 * MIB // size_bytes))


# Sizes on both sides of every extent: a file that spills past it must still cost
# only the extent, and a cell that never spills cannot show that.
cells = [Cell(e * KIB, s * KIB, files_for(s * KIB))
         for e in EXTENTS_KIB for s in SIZES_KIB]


def prepare(cell, workdir):
    granted = lustre.setstripe(workdir, dom(cell.extent_bytes, 1, MIB))
    # lfs df reports the whole filesystem, so other tenants move it while we
    # write.  Bracket the run with an idle sample of the same length.
    idle_before = lustre.mdt_used_kib()
    probe.sleep_seconds(DRIFT_SECONDS)
    idle_after = lustre.mdt_used_kib()

    before = lustre.mdt_used_kib()
    probe.write_files(workdir, cell.files, cell.size_bytes)
    os.sync()
    after = lustre.mdt_used_kib()

    with open(os.path.join(workdir, READINGS), "w") as fh:
        json.dump({"drift_kib": idle_after - idle_before,
                   "used_kib": after - before,
                   "granted_extent": granted.dom_extent_bytes,
                   "granted_components": len(granted.components)}, fh)


def measure(cell, workdir):
    with open(os.path.join(workdir, READINGS)) as fh:
        readings = json.load(fh)
    sample = os.path.join(workdir, "f0")
    granted = lustre.getstripe(sample)
    predicted_kib = cell.files * min(cell.size_bytes, cell.extent_bytes) // 1024
    row = dict(readings)
    row.update({"predicted_kib": predicted_kib,
                "file_objects": granted.objects_allocated,
                "file_dom_extent": granted.dom_extent_bytes,
                "stat_blocks": os.stat(sample).st_blocks})
    return row
