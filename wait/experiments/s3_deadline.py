"""The deadline regime: a unit is worth nothing late and everything on time.

**The sweep is arithmetic, not a cell dimension, and that is the whole design.**
An earlier version swept an absolute period and reported the miss rate as the
measurement.  It did not reproduce: across seven repeats the consumer's time per
unit was bimodal -- 24 ms in two repeats and 38 in five for the unpromoted arm,
12.7 and 20 for the promoted one -- because the filesystem was in two different
states, and a hard threshold sitting near the mean snapped whole cells between
0 % and 100 % miss.  The ratio between the arms was 1.9x in both states.

So the run measures what is stable: how long the deadline work takes, per unit,
per arm.  The band of arrival periods at which the tier decides whether you make
it is (promoted time, unpromoted time), and the miss rate at any period follows
from the recorded distribution rather than from a threshold applied inside the
measurement.  Reporting it the other way round let a shift in the machine read as
a result.
"""
import os
import statistics as st
from dataclasses import dataclass

from wait import arms, lustre, probe, ranks
from wait.layout import KIB
from wait.model import FileClass, Regime

UNITS = 40
# Files per class per unit, split across the producers.
FILES_PER_UNIT = 32
FILE_BYTES = 4 * KIB
# Sixteen rather than four, which cut a unit from 53 ms to 33 -- but not to the
# 7 ms the arithmetic suggested, because the cost is `lfs` process spawns and
# there is one per file however the files are split.  At about a millisecond a
# spawn the producer sustains roughly FILES_PER_UNIT milliseconds, which lands
# inside the band the consumer's arms separate in and cannot be moved out of it
# by adding producers.  So the sweep runs past the producer's own rate: the
# operative reading is the fastest period the producer can actually feed, and
# the producer's cost is reported beside every cell.
PRODUCERS = 16
# The statistics are re-read for reporting, which is what makes a count ranking
# prefer them.  Every re-read after the first is a page-cache hit, so a
# server-side counter sees one access for either class and is blind here too --
# S3 is the only scenario that defeats all three baselines.
STATISTICS_READS = 8
COORDINATOR = os.environ.get("WAIT_COORDINATOR", "127.0.0.1")

# Seven.  At three, the miss rate at the one period the producer can actually
# feed varies by more than the arm difference -- 0.425 against a spread of 0.475 --
# so the cell that matters does not separate while the two that do are periods
# nothing could produce.
repeats = 7

CLASSES = ("index", "statistics")


@dataclass(frozen=True)
class Cell:
    arm: str
    units: int = UNITS


cells = [Cell(arm) for arm in ("default", "heuristic", "size", "wait")]


def role(nodeid, localid):
    # Producers on one node and the consumer on another, so the writer is not
    # the reader within the run as well as between the phases.
    if nodeid == 0 and localid < PRODUCERS:
        return "producer"
    if nodeid == 1 and localid == 0:
        return "consumer"
    return "idle"


def file_classes(cell):
    count = FILES_PER_UNIT * cell.units
    # Both classes couple every rank, because that is what the harness does:
    # the consumer reads the unit and then reports on it while every producer
    # sits at the closing barrier, which is inside the unit loop.  Declaring
    # either class ranks_coupled=1 describes a pipeline that would not block
    # its producer on its consumer -- the intent rather than the program.
    coupled = max(2, ranks.world())
    return (FileClass("index", FILE_BYTES, accesses=1, ranks_coupled=coupled,
                      synchronized=True, regime=Regime.DEADLINE, count=count),
            FileClass("statistics", FILE_BYTES, accesses=STATISTICS_READS,
                      ranks_coupled=coupled, synchronized=True,
                      regime=Regime.BLOCKING, count=count))


def budget_bytes(cell):
    return FILES_PER_UNIT * cell.units * FILE_BYTES


def plan(cell, consts=None):
    """How many files of each class this arm promotes, and how to place them.

    A count per class rather than one name: a threshold that cannot rank two
    classes of one size promotes part of each, and collapsing that to a single
    name measures whichever class sorts first instead of the arm asked for.
    """
    counts = arms.promoted_counts(cell.arm, file_classes(cell), budget_bytes(cell),
                                  consts or arms.constants())
    return {"counts": counts, "per": per_producer(cell),
            "total": cell.units * FILES_PER_UNIT}


def path_for(workdir, name, unit, producer, index, placing):
    at = unit * FILES_PER_UNIT + producer * placing["per"] + index
    directory = arms.tier_dir(os.path.join(workdir, name), at,
                              placing["counts"].get(name, 0), placing["total"])
    return os.path.join(directory, "u%d_p%d_%d" % (unit, producer, index))


def per_producer(cell):
    return FILES_PER_UNIT // PRODUCERS


def prepare(cell, workdir):
    consts = arms.constants()
    placing = plan(cell, consts)
    for name in CLASSES:
        for directory, promoted in arms.tier_dirs(
                os.path.join(workdir, name),
                placing["counts"].get(name, 0), placing["total"]):
            os.makedirs(directory, exist_ok=True)
            lustre.setstripe(directory, arms.layout_for(promoted, consts))


def _emit(cell, workdir, unit, producer):
    """Write this producer's share of a unit and make it durable.

    A unit is published at flush-complete, not at write-return: an OST write is
    buffered, so without the flush the consumer would read from the producer's
    cache and the deadline would be measured against nothing.
    """
    written = []
    placing = plan(cell)
    for name in CLASSES:
        paths = [path_for(workdir, name, unit, producer, i, placing)
                 for i in range(per_producer(cell))]
        probe.write_paths(paths, FILE_BYTES)
        written += paths
    lustre.flush_paths(written)


def _consume(cell, workdir, unit, budget_ns, placing):
    """Read the unit's indices before the next unit is due, or abandon it.

    The deadline is what makes this class different from every other class in
    this project: a unit read too late is not worth less, it is worth nothing,
    so the consumer stops reading and drops it.  The budget is the time the
    producer took to make the unit -- carried on the barrier release, not
    measured from the consumer's own pace, which would widen as the consumer
    slowed and could never be missed.
    """
    started, staged = probe.NS(), []
    for p in range(PRODUCERS):
        for i in range(per_producer(cell)):
            if budget_ns and probe.NS() - started > budget_ns:
                return staged, True          # dropped: the rest is worthless
            staged.append(probe.read_staged(
                path_for(workdir, "index", unit, p, i, placing), FILE_BYTES))
    return staged, False


def _report(cell, workdir, unit, placing):
    for _pass in range(STATISTICS_READS):
        for p in range(PRODUCERS):
            for i in range(per_producer(cell)):
                probe.read_staged(
                    path_for(workdir, "statistics", unit, p, i, placing),
                    FILE_BYTES)


def measure(cell, workdir):
    placing = plan(cell)
    me, size = ranks.rank(), ranks.world()
    nodeid = int(os.environ.get("SLURM_NODEID", "0"))
    localid = int(os.environ.get("SLURM_LOCALID", "0"))
    mine = role(nodeid, localid)
    barrier = ranks.Barrier(COORDINATOR, size=size, me=me)

    produce, consume, staged, dropped, budgets = [], [], [], [], []
    barrier.wait()
    for unit in range(cell.units):
        start = probe.NS()
        if mine == "producer":
            _emit(cell, workdir, unit, localid)
            produce.append(probe.NS() - start)
        # The unit becomes available now, and the release carries how long it
        # took to make -- which is the window the consumer has before the next
        # one is due.
        budget = barrier.broadcast(produce[-1] if mine == "producer" and me == 0
                                   else 0)
        if mine == "consumer":
            # The window the consumer was actually handed, which is the one it
            # was judged against -- the slowest producer's time is a different
            # quantity and reporting it as the budget overstates the window.
            budgets.append(budget)
            t0 = probe.NS()
            got, missed = _consume(cell, workdir, unit, budget, placing)
            consume.append(probe.NS() - t0)
            staged += got
            dropped.append(missed)
            if not missed:
                _report(cell, workdir, unit, placing)
        barrier.wait()

    gathered = barrier.gather({"role": mine, "produce_ns": produce,
                               "consume_ns": consume, "dropped": dropped,
                               "budget_ns": budgets,
                               "staged": [(s["open_ns"], s["fstat_ns"],
                                           s["first_read_ns"]) for s in staged]})
    barrier.close()
    if me:
        return {"role": mine}

    producers = [g for g in gathered if g["role"] == "producer"]
    consumer = next(g for g in gathered if g["role"] == "consumer")
    # A unit the producers publish late is late for a reason on the write path,
    # and crediting the tier for it would double-count A6.
    slowest = [max(g["produce_ns"][u] for g in producers)
               for u in range(cell.units)]
    flat = consumer["staged"]
    # Every class, not the promoted one by name: naming it here would tell a
    # reader of this source which class the advisor took.
    on_mdt = {name: lustre.getstripe(
        path_for(workdir, name, 0, 0, 0, placing)).is_dom for name in CLASSES}
    consts = arms.constants()
    classes, budget = file_classes(cell), budget_bytes(cell)
    return {"ranks": size,
            # Every unit, so the miss rate at any period is arithmetic later and
            # no threshold sits inside the measurement.
            "consume_ns": consumer["consume_ns"],
            "dropped": sum(consumer["dropped"]),
            "drop_rate": sum(consumer["dropped"]) / len(consumer["dropped"]),
            "produce_ns": slowest,
            "consume_ns_p50": st.median(consumer["consume_ns"]),
            "consume_ns_p90": probe.nearest_rank(consumer["consume_ns"], 0.90),
            "produce_ns_p50": st.median(slowest),
            "produce_ns_p90": probe.nearest_rank(slowest, 0.90),
            "budget_ns": consumer["budget_ns"],
            "budget_ns_p50": st.median(consumer["budget_ns"]),
            "open_ns": st.median([s[0] for s in flat]),
            "fstat_ns": st.median([s[1] for s in flat]),
            "read_ns": st.median([s[2] for s in flat]),
            "on_mdt": on_mdt,
            "predicted_ns": arms.predicted_value_ns(cell.arm, classes, budget, consts),
            "baselines": arms.baseline_decisions(classes, budget, consts),
            "promoted_files": sum(placing["counts"].values()),
            "promoted_by_class": placing["counts"]}
