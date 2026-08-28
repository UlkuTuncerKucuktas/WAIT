import os
import statistics as st
from dataclasses import dataclass

from wait import arms, lustre, probe, ranks
from wait.layout import KIB
from wait.model import FileClass, Regime

RESULT_BYTES = 4 * KIB
# Matched to the results in size and in count.  Without that the heuristics are
# not indifferent -- a size threshold or a count ranking would have a basis to
# discriminate, and the scenario would be measuring the wrong thing.
DIAGNOSTIC_BYTES = RESULT_BYTES
TASKS_PER_RANK = 50
# Four generations at fifty tasks is two hundred durable files per rank, the
# point at which the arms stop overlapping: at forty per rank the margin is
# 4.4 % and inside the spread.
GENERATIONS = 4
COORDINATOR = os.environ.get("WAIT_COORDINATOR", "127.0.0.1")

repeats = 7




@dataclass(frozen=True)
class Cell:
    arm: str
    generations: int = GENERATIONS
    tasks_per_rank: int = TASKS_PER_RANK


cells = [Cell(arm) for arm in ("default", "heuristic", "size", "wait")]

# Both classes are one directory per rank, so a promoted class is whole
# directories and no file has to name its own layout.
CLASSES = ("results", "diagnostics")


def file_classes(cell):
    # Equal count defeats a count ranking and equal size defeats a threshold, so
    # the heuristic is indifferent.  The results are fsynced and every rank waits
    # on them at the generation barrier; the diagnostics are buffered and block
    # nobody.
    # Each rank writes its own files, so a result is private and the barrier
    # synchronises the generation rather than any one write: the machine saves
    # the per-rank saving on every rank at once, which is the file count times
    # the saving, not that times the rank count.  S1 is where the multiplier
    # lives; this is the write path with the same blocking-versus-hidden axis.
    count = budget_files(cell)
    return (FileClass("results", RESULT_BYTES, accesses=1, ranks_coupled=1,
                      synchronized=False, regime=Regime.BLOCKING,
                      count=count, writes=True),
            FileClass("diagnostics", DIAGNOSTIC_BYTES, accesses=1,
                      ranks_coupled=1, synchronized=False,
                      regime=Regime.HIDDEN, count=count, writes=True))


def budget_bytes(cell):
    return budget_files(cell) * RESULT_BYTES


def plan(cell, consts=None):
    """How many files of each class this arm promotes, and how to place them.

    Derived, never named here.  A count per class rather than one name: a
    threshold that cannot rank two classes of one size promotes part of each,
    and every rank writes its own share, so the split lands inside each rank's
    directory rather than by rank.
    """
    counts = arms.promoted_counts(cell.arm, file_classes(cell), budget_bytes(cell),
                                  consts or arms.constants())
    per_rank = cell.generations * cell.tasks_per_rank
    return {"counts": counts, "per_rank": per_rank, "tasks": cell.tasks_per_rank,
            "total": budget_files(cell)}


def promoted_in_rank(placing, name):
    if placing["total"] <= 0:
        return 0
    return placing["counts"].get(name, 0) * placing["per_rank"] // placing["total"]


def class_dir(workdir, name, rank, index, placing):
    return arms.tier_dir(os.path.join(workdir, name, "r%d" % rank), index,
                         promoted_in_rank(placing, name), placing["per_rank"])


def path_for(workdir, name, rank, generation, task, placing):
    index = generation * placing["tasks"] + task
    return os.path.join(class_dir(workdir, name, rank, index, placing),
                        "g%d_t%d" % (generation, task))


def budget_files(cell):
    return cell.generations * cell.tasks_per_rank * ranks.world()


def prepare(cell, workdir):
    consts = arms.constants()
    placing = plan(cell, consts)
    for name in CLASSES:
        for rank in range(ranks.world()):
            for directory, promoted in arms.tier_dirs(
                    os.path.join(workdir, name, "r%d" % rank),
                    promoted_in_rank(placing, name), placing["per_rank"]):
                os.makedirs(directory, exist_ok=True)
                lustre.setstripe(directory, arms.layout_for(promoted, consts))


def measure(cell, workdir):
    placing = plan(cell)
    me, size = ranks.rank(), ranks.world()
    barrier = ranks.Barrier(COORDINATOR, size=size, me=me)
    barrier.wait()

    result, diagnostic = b"x" * RESULT_BYTES, b"x" * DIAGNOSTIC_BYTES
    generations, staged = [], []
    for g in range(cell.generations):
        # Every rank starts the generation together, so its cost is the slowest
        # rank's and not the spread of when each happened to begin.
        barrier.wait()
        start = probe.NS()
        for t in range(cell.tasks_per_rank):
            # The result must survive a node failure mid-generation, so it is
            # fsynced and every rank waits on it at the barrier.  The diagnostic
            # is written alongside and nobody waits on it.
            staged.append(probe.write_staged(
                path_for(workdir, "results", me, g, t, placing), result))
            probe.write_buffered(
                path_for(workdir, "diagnostics", me, g, t, placing), diagnostic)
        generations.append(probe.NS() - start)

    gathered = barrier.gather({"generations_ns": generations,
                               "staged": [(s["create_ns"], s["write_ns"],
                                           s["fsync_ns"]) for s in staged]})
    barrier.close()
    if me != 0:
        return {"ranks": size, "rank": me}

    per_generation = [[g["generations_ns"][i] for g in gathered]
                      for i in range(cell.generations)]
    release = sum(max(times) for times in per_generation)
    flat = [s for g in gathered for s in g["staged"]]
    # Every class, not the promoted one by name.
    on_mdt = {name: lustre.getstripe(
        path_for(workdir, name, 0, 0, 0, placing)).is_dom for name in CLASSES}
    return {"ranks": size,
            "predicted_ns": arms.predicted_value_ns(
                cell.arm, file_classes(cell), budget_bytes(cell), arms.constants()),
            "baselines": arms.baseline_decisions(
                file_classes(cell), budget_bytes(cell), arms.constants()),
            "generation_release_ns": release,
            "typical_ns": sum(st.median(t) for t in per_generation),
            "core_ns": release * size,
            "create_ns": st.median([s[0] for s in flat]),
            "write_ns": st.median([s[1] for s in flat]),
            "fsync_ns": st.median([s[2] for s in flat]),
            "durable_ns": st.median([s[1] + s[2] for s in flat]),
            "on_mdt": on_mdt,
            "promoted_files": sum(placing["counts"].values()),
            "promoted_by_class": placing["counts"]}
