import multiprocessing
import os
import statistics as st
from dataclasses import dataclass

from wait import arms, lustre, probe
from wait.layout import KIB
from wait.model import FileClass, Regime

ITEMS = 300
# Below the inline limit, or the advisor declines both classes and the scenario
# has nothing to allocate: a file above it burns MDT bytes and still pays a full
# read round trip.
ITEM_BYTES = 64 * KIB
# What a 64 KiB cold cross-node read costs here, staged open+fstat+read: 1058 us
# measured in the unpromoted arm.  Compute is a multiple of it so the ratio, not
# the absolute, is the parameter -- and the achieved ratio is recorded per cell,
# because promoting the class makes its read cheaper and moves the ratio.
NOMINAL_READ_S = 0.00106
PREFETCH_DEPTH = 8
# Two compute ratios, as a robustness check rather than a threshold sweep.  The
# prefetcher cannot be made to under-run here at any ratio: it reads one item per
# main-loop iteration while the main loop reads one of its own *and* computes, so
# it is never the slower side.  Measured, the hidden stall share is 2.2 % at 2.0
# and 3.4 % at 0.5.  Un-hiding the class would need the hidden side to cost more
# per item, which is exactly the asymmetry the heuristics must not be able to
# see, so indifference and a demonstrable under-run cannot both be had in one
# workload.  The under-run is measured separately and reported as its own row.
RATIOS = (2.0, 0.5)

repeats = 3


# Named for what they hold, not for what they cost.  Calling them
# "hidden" and "blocking" put the answer to question one in the class
# name, where an agent reading the source would not have to derive it.
CLASSES = ("tiles", "masks")


@dataclass(frozen=True)
class Cell:
    arm: str
    compute_ratio: float
    items: int = ITEMS


cells = [Cell(arm, ratio) for ratio in RATIOS
         for arm in ("default", "heuristic", "size", "wait")]


def file_classes(cell):
    # Identical in size, in count and in access count -- everything a heuristic
    # can see.  They differ only in whether anything waits on them, which is the
    # one thing that has to be read out of the source.
    return (FileClass("tiles", ITEM_BYTES, accesses=1, ranks_coupled=1,
                      synchronized=False, regime=Regime.HIDDEN, count=cell.items),
            FileClass("masks", ITEM_BYTES, accesses=1, ranks_coupled=1,
                      synchronized=False, regime=Regime.BLOCKING, count=cell.items))


def plan(cell, consts=None):
    """How many files of each class this arm promotes, and how to place them.

    Derived, never named here: the advisor and the baseline policies are what
    this measures, so they decide and the scenario applies what comes back.  A
    count per class rather than one name, because a threshold that cannot rank
    two classes of one size promotes part of each.
    """
    counts = arms.promoted_counts(cell.arm, file_classes(cell), budget_bytes(cell),
                                  consts or arms.constants())
    return {"counts": counts, "total": cell.items}


def path_for(workdir, name, index, placing):
    directory = arms.tier_dir(os.path.join(workdir, name), index,
                              placing["counts"].get(name, 0), placing["total"])
    return os.path.join(directory, "i%d" % index)


def budget_files(cell):
    return cell.items


def budget_bytes(cell):
    # Room for exactly one class: below it nothing fits, above the union
    # everything does and no policy has a decision to make.
    return cell.items * ITEM_BYTES


def prepare(cell, workdir):
    consts = arms.constants()
    placing = plan(cell, consts)
    for name in CLASSES:
        for directory, promoted in arms.tier_dirs(
                os.path.join(workdir, name), placing["counts"].get(name, 0), cell.items):
            os.makedirs(directory, exist_ok=True)
            lustre.setstripe(directory, arms.layout_for(promoted, consts))
        probe.write_paths([path_for(workdir, name, i, placing)
                           for i in range(cell.items)], ITEM_BYTES)


def _prefetch(workdir, items, queue, placing):
    # A separate process, not a thread: the point is that this work overlaps the
    # main loop's compute, and the queue depth is what bounds how far it runs
    # ahead.
    for i in range(items):
        probe.read_staged(path_for(workdir, "tiles", i, placing),
                          ITEM_BYTES)
        queue.put(i)


def measure(cell, workdir):
    placing = plan(cell)
    queue = multiprocessing.Queue(maxsize=PREFETCH_DEPTH)
    worker = multiprocessing.Process(
        target=_prefetch, args=(workdir, cell.items, queue, placing))
    compute_s = cell.compute_ratio * NOMINAL_READ_S

    waits, gated, staged = [], [], []
    worker.start()
    start = probe.NS()
    for i in range(cell.items):
        # Waiting on the prefetcher.  Near zero means the class really was
        # hidden, which is the in-run evidence the scenario rests on.
        t0 = probe.NS()
        queue.get()
        waits.append(probe.NS() - t0)
        t1 = probe.NS()
        staged.append(probe.read_staged(
            path_for(workdir, "masks", i, placing), ITEM_BYTES))
        gated.append(probe.NS() - t1)
        probe.sleep_seconds(compute_s)
    phase = probe.NS() - start
    worker.join()

    # Every class, not the promoted one by name: naming it here tells a reader
    # of this source which class the advisor took.
    on_mdt = {name: lustre.getstripe(
        path_for(workdir, name, 0, placing)).is_dom
        for name in CLASSES}
    queue_wait, gated_total = sum(waits), sum(gated)
    measured_read_s = (gated_total / len(gated)) / 1e9 if gated else 0.0
    return {"phase_ns": phase,
            "compute_s": compute_s,
            "measured_read_s": measured_read_s,
            "achieved_ratio": compute_s / measured_read_s if measured_read_s else 0.0,
            "predicted_ns": arms.predicted_value_ns(
                cell.arm, file_classes(cell), budget_bytes(cell), arms.constants()),
            "baselines": arms.baseline_decisions(
                file_classes(cell), budget_bytes(cell), arms.constants()),
            "tiles_stall_ns": queue_wait,
            "masks_stall_ns": gated_total,
            "tiles_stall_share": queue_wait / phase if phase else 0.0,
            "masks_stall_share": gated_total / phase if phase else 0.0,
            "open_ns": st.median([s["open_ns"] for s in staged]),
            "fstat_ns": st.median([s["fstat_ns"] for s in staged]),
            "read_ns": st.median([s["first_read_ns"] for s in staged]),
            "on_mdt": on_mdt,
            "promoted_files": sum(placing["counts"].values()),
            "promoted_by_class": placing["counts"]}
