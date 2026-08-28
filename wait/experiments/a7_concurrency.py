import os
import statistics as st
from dataclasses import dataclass

from wait import arms, lustre, probe, ranks
from wait.layout import KIB

FILE_BYTES = 4 * KIB
ROUNDS = 60
# Participants are spread over both reader nodes.  Served entirely by one
# client, a shared read is one cold fetch and C-1 page-cache hits, and the arms
# cannot separate.
READER_NODES = 2
CONCURRENCY = (1, 2, 4, 8, 16, 32)
COORDINATOR = os.environ.get("WAIT_COORDINATOR", "127.0.0.1")

repeats = 3


@dataclass(frozen=True)
class Cell:
    arm: str
    share: str
    concurrency: int
    rounds: int = ROUNDS


# The two arms of a comparison run back to back, so drift over a run cannot
# land on one of them.  Whichever arm runs first wins, in both directions, so
# the pair order alternates along the sweep as well: half the comparisons put
# DoM first and half put OST first, which leaves the position effect in the
# spread instead of in the result.
cells = [Cell(arm, share, c)
         for share in ("distinct", "same")
         for i, c in enumerate(CONCURRENCY)
         for arm in (("dom", "ost") if i % 2 == 0 else ("ost", "dom"))]


def per_node(concurrency):
    return [concurrency // READER_NODES + (1 if n < concurrency % READER_NODES else 0)
            for n in range(READER_NODES)]


def participant(cell, nodeid, localid):
    """Which participant this rank is, or None if it only barriers."""
    counts = per_node(cell.concurrency)
    if nodeid >= READER_NODES or localid >= counts[nodeid]:
        return None
    return sum(counts[:nodeid]) + localid


def path_for(cell, workdir, index, round_index):
    # `distinct` gives every participant its own file, so the participant count
    # is also the number of cold fetches.  `same` puts all of them on one file,
    # which the client page cache turns into one fetch per node -- so the two
    # modes separate the rate of distinct cold fetches from the rank count.
    name = ("r%d_%d" % (index, round_index) if cell.share == "distinct"
            else "shared_%d" % round_index)
    return os.path.join(workdir, name)


def prepare(cell, workdir):
    lustre.setstripe(workdir, arms.layout_for(cell.arm == "dom", arms.constants()))
    holders = range(cell.concurrency) if cell.share == "distinct" else [0]
    probe.write_paths([path_for(cell, workdir, p, r)
                       for p in holders for r in range(cell.rounds)], FILE_BYTES)


def measure(cell, workdir):
    me, size = ranks.rank(), ranks.world()
    nodeid = int(os.environ.get("SLURM_NODEID", "0"))
    localid = int(os.environ.get("SLURM_LOCALID", "0"))
    index = participant(cell, nodeid, localid)
    barrier = ranks.Barrier(COORDINATOR, size=size, me=me)

    barrier.wait()
    # Counters are per client mount, so one rank per node reads them.
    before = lustre.counters() if localid == 0 else None
    mine = []
    for r in range(cell.rounds):
        # Every rank lines up, participant or not, so the timed read is the only
        # traffic in flight and the concurrency is the whole of it.
        barrier.wait()
        if index is None:
            continue
        # Staged, because DoM carries its data in the open reply and OST in the
        # read: a total says which arm is faster but never which stage
        # concurrency taxes.
        mine.append(probe.read_staged(path_for(cell, workdir, index, r),
                                      FILE_BYTES))
    barrier.wait()

    payload = {"rank": me, "host": os.uname().nodename, "participant": index,
               "staged": [(s["open_ns"], s["fstat_ns"], s["first_read_ns"])
                          for s in mine]}
    if before is not None:
        after = lustre.counters()
        payload["ost_bulk"] = after.ost_bulk_rpcs - before.ost_bulk_rpcs
        payload["intent_lock"] = after.mdc_intent_locks - before.mdc_intent_locks
    gathered = barrier.gather(payload)
    barrier.close()
    if me:
        return {"concurrency": cell.concurrency, "rank": me}

    readers = [g for g in gathered if g["participant"] is not None]
    totals = [[sum(g["staged"][r]) for g in readers] for r in range(cell.rounds)]
    flat = [s for g in readers for s in g["staged"]]
    counted = [g for g in gathered if "ost_bulk" in g]
    return {"ranks": size,
            "readers": len(readers),
            "per_node": per_node(cell.concurrency),
            # Median over ranks and maximum over ranks, each taken per round and
            # then over rounds: the first resolves the tier, the second is what a
            # barrier would actually cost.
            "median_ns": st.median([st.median(t) for t in totals]),
            "max_ns": st.median([max(t) for t in totals]),
            "open_ns": st.median([s[0] for s in flat]),
            "fstat_ns": st.median([s[1] for s in flat]),
            "read_ns": st.median([s[2] for s in flat]),
            "ost_bulk_per_node": [g["ost_bulk"] for g in counted],
            "intent_lock_per_node": [g["intent_lock"] for g in counted]}
