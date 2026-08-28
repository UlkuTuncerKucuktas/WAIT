import os
import statistics as st
from dataclasses import dataclass

from wait import arms, lustre, probe, ranks
from wait.layout import KIB
from wait.model import FileClass, Regime

MANIFEST_BYTES = 4 * KIB
SIDECAR_BYTES = 4 * KIB
SHARD_BYTES = 256 * KIB
# The barrier releases on the slowest rank, so a round costs the maximum over
# ranks -- a statistic drawn from the tail, where per-round spread runs about
# 500 us against a tier effect of 300.  Noise over R rounds grows as sqrt(R)
# while signal grows as R, so R has to be large: at twenty the ratio is under
# three, at two hundred it is near eight.
ROUNDS = 200
# A floor, not the value.  The budget is one class's bytes, so the competing
# class must be able to absorb all of it or the count ranking spills into the
# manifests and captures the benefit by accident -- at four ranks ten sidecars
# each is forty files against a budget of two hundred.  Scaling the floor keeps
# the discriminator valid at every rank count.
SIDECARS_PER_RANK = 10
SHARDS_PER_RANK = 2
# Ranks span two reader nodes, so the rendezvous host is the coordinator's, not
# the loopback each node would resolve to itself.
COORDINATOR = os.environ.get("WAIT_COORDINATOR", "127.0.0.1")

# Five, not three: at thirty-two ranks the arm difference is 26 ms against a
# spread of 60, and three repeats cannot resolve it.
repeats = 5




# The rank count comes from SLURM_NTASKS, not from a cell: sixty-four Python
# processes on four cores make the barrier report scheduler latency rather than
# the fetch every rank is waiting on.  Ranks are tasks, and N is swept by
# submitting the job again.
@dataclass(frozen=True)
class Cell:
    arm: str
    rounds: int = ROUNDS
    sidecars_per_rank: int = SIDECARS_PER_RANK
    shards_per_rank: int = SHARDS_PER_RANK


cells = [Cell(arm) for arm in ("default", "heuristic", "size", "wait")]


def release_ns(rounds):
    # Everyone waits for the slowest fetch, so a round costs the maximum over
    # ranks -- and the machine loses that once per rank, not once per fetch.
    return sum(max(times) for times in rounds.values())


def ranks_near_max(rounds, within=0.25):
    # How many ranks a round's maximum actually represents.  The release metric
    # is a maximum over ranks, and ranks per node rises with N while the file is
    # fetched cold once per node -- so if the population it maximises over changes
    # composition, the absolute is not comparable across N even though the arm
    # difference at one N still is.  Two, at every N, means it is the cold fetch.
    counts = []
    for times in rounds.values():
        top = max(times)
        counts.append(sum(1 for t in times if t >= top * (1 - within)))
    return st.median(counts)


def spread_ratio(rounds):
    # A round's maximum over its median.  Near one means the maximum is a cached
    # read like the rest, and the metric has stopped measuring the fetch.
    ratios = [max(t) / st.median(t) for t in rounds.values() if st.median(t) > 0]
    return st.median(ratios) if ratios else 0.0


def typical_ns(rounds):
    # The same rounds by median rank rather than slowest.  It is not what the
    # barrier costs, but it resolves the tier without reaching into the tail, so
    # reporting both separates the effect from the statistic.
    return sum(st.median(times) for times in rounds.values())


def budget_files(cell):
    # Room for exactly one class of manifests: below it nothing fits, above the
    # union everything does and no policy has a decision to make.
    return cell.rounds


def sidecars_per_rank(cell, world=None):
    world = ranks.world() if world is None else world
    return max(cell.sidecars_per_rank,
               -(-budget_files(cell) // world))       # ceil, so no spill


def file_classes(cell):
    # The manifest gates every rank and is read once; a sidecar is private and
    # re-read every round, so a count ranking prefers it and the shards are too
    # large for the tier at all.
    return (FileClass("manifest", MANIFEST_BYTES, accesses=1,
                      ranks_coupled=ranks.world(), synchronized=True,
                      regime=Regime.BLOCKING, count=cell.rounds),
            FileClass("sidecar", SIDECAR_BYTES, accesses=cell.rounds,
                      ranks_coupled=1, synchronized=False,
                      regime=Regime.BLOCKING,
                      count=sidecars_per_rank(cell) * ranks.world()),
            FileClass("shard", SHARD_BYTES, accesses=cell.rounds,
                      ranks_coupled=1, synchronized=False,
                      regime=Regime.BLOCKING,
                      count=cell.shards_per_rank * ranks.world()))


def budget_bytes(cell):
    return budget_files(cell) * MANIFEST_BYTES


def plan(cell, consts=None):
    """How many files of each class this arm promotes.

    Derived, never named here.  A count per class rather than one name: a
    threshold that cannot rank two classes of one size promotes part of each,
    and applying the first file's layout to the whole directory would promote
    all of them.
    """
    allocated = arms.allocation(cell.arm, file_classes(cell), budget_bytes(cell),
                                consts or arms.constants())
    per_rank = sidecars_per_rank(cell)
    world = ranks.world()
    return {"manifests": min(allocated.files("manifest"), cell.rounds),
            "rounds": cell.rounds,
            "sidecars_per_rank": per_rank,
            # Spread over ranks rather than filling whole ranks first, so the
            # split is the same shape on every rank and no rank is a special
            # case in the read loop.
            "sidecars": min(allocated.files("sidecar") // max(1, world), per_rank)}


def promoted_paths(cell, workdir, consts=None):
    placing = plan(cell, consts)
    paths = [manifest(workdir, r, placing) for r in range(placing["manifests"])]
    paths += [sidecar(workdir, rank, i, placing) for rank in range(ranks.world())
              for i in range(placing["sidecars"])]
    return paths


def manifest(workdir, round_index, placing):
    # One directory per layout: a directory holding two forces a per-file
    # setstripe, and that costs the promoted open 612.9 us against 223.4 at
    # thirty-two ranks -- measured, both arms, same round.
    directory = arms.tier_dir(os.path.join(workdir, "manifests"), round_index,
                              placing["manifests"], placing["rounds"])
    return os.path.join(directory, "m%d" % round_index)


def sidecar(workdir, rank, index, placing):
    directory = arms.tier_dir(os.path.join(workdir, "sidecars", "r%d" % rank),
                              index, placing["sidecars"],
                              placing["sidecars_per_rank"])
    return os.path.join(directory, "s%d" % index)


def shard(workdir, rank, index):
    return os.path.join(workdir, "shards", "r%d" % rank, "s%d" % index)


def _fill(paths, layout, size_bytes):
    # The paths measure will read, so the two cannot drift apart.
    # One directory, one layout, inherited.  A directory holding two layouts
    # forces a per-file setstripe, and that costs the promoted open 612.9 us
    # against 223.4 at thirty-two ranks -- measured, both arms, same round.
    directory = os.path.dirname(paths[0])
    os.makedirs(directory, exist_ok=True)
    lustre.setstripe(directory, layout)
    probe.write_paths(paths, size_bytes)


def prepare(cell, workdir):
    consts = arms.constants()
    placing = plan(cell, consts)
    # A fresh manifest per round.  Re-reading one is served from every node's
    # page cache, so the sum over rounds would accumulate noise, not delta.
    _fill_split(workdir, "manifests", None, placing["manifests"],
                placing["rounds"],
                lambda i: manifest(workdir, i, placing), MANIFEST_BYTES, consts)
    for rank in range(ranks.world()):
        _fill_split(workdir, "sidecars", rank, placing["sidecars"],
                    placing["sidecars_per_rank"],
                    lambda i, r=rank: sidecar(workdir, r, i, placing),
                    SIDECAR_BYTES, consts)
        _fill([shard(workdir, rank, i) for i in range(cell.shards_per_rank)],
              arms.layout_for(False, consts), SHARD_BYTES)


def _fill_split(workdir, klass, rank, promoted, total, path_of, size_bytes, consts):
    """Write a class, one directory per layout, promoted files first."""
    base = os.path.join(workdir, klass)
    if rank is not None:
        base = os.path.join(base, "r%d" % rank)
    for directory, is_dom in arms.tier_dirs(base, promoted, total):
        os.makedirs(directory, exist_ok=True)
        lustre.setstripe(directory, arms.layout_for(is_dom, consts))
    paths = [path_of(i) for i in range(total)]
    for directory in {os.path.dirname(p) for p in paths}:
        probe.write_paths([p for p in paths if os.path.dirname(p) == directory],
                          size_bytes)


def _read(path, size_bytes):
    fd = os.open(path, os.O_RDONLY)
    os.fstat(fd)
    os.pread(fd, size_bytes, 0)
    os.close(fd)


def _read_staged(path, size_bytes):
    # The tiers carry their data in different stages -- DoM in the open reply,
    # OST in the read -- so a total says which arm is faster but never why.
    t0 = probe.NS()
    fd = os.open(path, os.O_RDONLY)
    t1 = probe.NS()
    os.fstat(fd)
    t2 = probe.NS()
    os.pread(fd, size_bytes, 0)
    t3 = probe.NS()
    os.close(fd)
    return t1 - t0, t2 - t1, t3 - t2


def measure(cell, workdir):
    placing = plan(cell)
    me, size = ranks.rank(), ranks.world()
    barrier = ranks.Barrier(COORDINATOR, size=size, me=me)
    barrier.wait()

    mine, stages, start = [], [], probe.NS()
    for r in range(cell.rounds):
        # Line every rank up before the timed read.  Without it the private reads
        # below leave each rank arriving at the next manifest at a different
        # moment, and their concurrent traffic lands inside the measurement.
        barrier.wait()
        t0 = probe.NS()
        o, f, d = _read_staged(manifest(workdir, r, placing), MANIFEST_BYTES)
        mine.append(probe.NS() - t0)
        stages.append((o, f, d))
        barrier.wait()
        # The work the gate holds up, not part of the stall it costs.
        for i in range(sidecars_per_rank(cell)):
            _read(sidecar(workdir, me, i, placing), SIDECAR_BYTES)
        for i in range(cell.shards_per_rank):
            _read(shard(workdir, me, i), SHARD_BYTES)
    phase = probe.NS() - start

    gathered = barrier.gather({"manifest_ns": mine, "phase_ns": phase,
                               "stages": stages})
    barrier.close()
    if me != 0:
        return {"ranks": size, "rank": me}

    rounds = {r: [g["manifest_ns"][r] for g in gathered] for r in range(cell.rounds)}
    release = release_ns(rounds)
    flat = [s for g in gathered for s in g["stages"]]
    consts = arms.constants()
    classes, budget = file_classes(cell), budget_bytes(cell)
    return {"ranks": size,
            "predicted_ns": arms.predicted_value_ns(cell.arm, classes, budget, consts),
            "baselines": arms.baseline_decisions(classes, budget, consts),
            "open_ns": st.median([s[0] for s in flat]),
            "fstat_ns": st.median([s[1] for s in flat]),
            "read_ns": st.median([s[2] for s in flat]),
            "gate_ns": st.median([sum(s) for s in flat]),
            "barrier_release_ns": release,
            "ranks_near_max": ranks_near_max(rounds),
            "max_over_median": spread_ratio(rounds),
            "typical_ns": typical_ns(rounds),
            "core_ns": release * size,
            "phase_ns_max": max(g["phase_ns"] for g in gathered),
            "promoted_files": len(promoted_paths(cell, workdir))}
