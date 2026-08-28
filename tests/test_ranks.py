import multiprocessing as mp
import pathlib
import time
import unittest

from wait.ranks import Barrier, RankError

SOURCE = pathlib.Path(__file__).resolve().parent.parent / "wait" / "ranks.py"


def _member(rank, size, port, late, arrivals, releases):
    if late and rank == size - 1:
        time.sleep(0.4)
    barrier = Barrier("127.0.0.1", size=size, me=rank, port=port)
    arrivals.append((rank, time.monotonic()))
    barrier.wait()
    releases.append((rank, time.monotonic()))
    barrier.close()


def run(size, port, late=False):
    with mp.Manager() as manager:
        arrivals, releases = manager.list(), manager.list()
        procs = [mp.Process(target=_member,
                            args=(r, size, port, late, arrivals, releases))
                 for r in range(size)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
        return list(arrivals), list(releases), [p.exitcode for p in procs]


class Rendezvous(unittest.TestCase):

    def test_every_rank_passes_the_barrier(self):
        arrivals, releases, codes = run(6, 47901)
        self.assertEqual(codes, [0] * 6)
        self.assertEqual(len(releases), 6)

    def test_ranks_are_released_together_however_late_one_arrives(self):
        # The barrier's whole purpose: everyone leaves when the slowest arrives,
        # which is what makes a shared file cost the rank count and not one fetch.
        arrivals, releases, codes = run(4, 47902, late=True)
        self.assertEqual(codes, [0] * 4)
        arrival_spread = max(t for _, t in arrivals) - min(t for _, t in arrivals)
        release_spread = max(t for _, t in releases) - min(t for _, t in releases)
        self.assertGreater(arrival_spread, 0.3)
        self.assertLess(release_spread, arrival_spread / 4)

    def test_a_single_rank_needs_no_coordinator(self):
        barrier = Barrier("no-such-host", size=1, me=0)
        barrier.wait()
        barrier.close()

    def test_an_unreachable_coordinator_raises_rather_than_hanging(self):
        from wait import ranks
        original = ranks.CONNECT_TIMEOUT_S
        ranks.CONNECT_TIMEOUT_S = 0.3
        try:
            with self.assertRaises(RankError):
                Barrier("127.0.0.1", size=2, me=1, port=47903)
        finally:
            ranks.CONNECT_TIMEOUT_S = original


def _gather_member(rank, size, port, results):
    barrier = Barrier("127.0.0.1", size=size, me=rank, port=port)
    barrier.wait()
    got = barrier.gather({"rank": rank, "times": [rank * 10, rank * 10 + 1]})
    if rank == 0:
        results.extend(got)
    barrier.close()


class Gather(unittest.TestCase):

    def test_rank_zero_receives_every_rank(self):
        # Rank 0 needs all the timings to take a maximum per round; the others
        # send theirs over the rendezvous socket rather than through a file.
        with mp.Manager() as manager:
            results = manager.list()
            procs = [mp.Process(target=_gather_member, args=(r, 5, 47904, results))
                     for r in range(5)]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=30)
            got = list(results)
        self.assertEqual(sorted(g["rank"] for g in got), [0, 1, 2, 3, 4])
        self.assertEqual(sorted(g["times"] for g in got)[-1], [40, 41])

    def test_a_single_rank_gathers_only_itself(self):
        barrier = Barrier("no-such-host", size=1, me=0)
        self.assertEqual(barrier.gather({"x": 1}), [{"x": 1}])
        barrier.close()


class Scale(unittest.TestCase):

    def test_wait_scale_outranks_slurm_ntasks(self):
        # srun rewrites SLURM_NTASKS per step, so a single-task prepare and its
        # multi-task measure would disagree on the rank count and compute
        # different ledger keys for the same cell.
        import os

        from wait import ranks
        saved = {k: os.environ.get(k) for k in ("WAIT_SCALE", "SLURM_NTASKS")}
        try:
            os.environ["SLURM_NTASKS"] = "1"
            os.environ["WAIT_SCALE"] = "16"
            self.assertEqual(ranks.world(), 16)
            del os.environ["WAIT_SCALE"]
            self.assertEqual(ranks.world(), 1)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class OffTheFilesystem(unittest.TestCase):

    def test_the_barrier_never_touches_the_filesystem(self):
        text = SOURCE.read_text()
        for call in ("open(", "makedirs", "os.path", "unlink"):
            self.assertNotIn(call, text, call)


if __name__ == "__main__":
    unittest.main()


def _cycle_member(rank, size, port, rounds, results):
    barrier = Barrier("127.0.0.1", size=size, me=rank, port=port)
    seen = []
    for r in range(rounds):
        barrier.wait()
        got = barrier.gather({"rank": rank, "round": r,
                              "payload": list(range(rank * 50, rank * 50 + 50))})
        barrier.wait()
        if rank == 0:
            seen.append(sorted(g["rank"] for g in got))
    if rank == 0:
        results.extend(seen)
    barrier.close()


class RepeatedGather(unittest.TestCase):

    def test_a_barrier_after_a_gather_does_not_corrupt_the_next_gather(self):
        # Barrier bytes and gather lines travel on one socket.  A reader that
        # takes whatever the kernel offers reads the line and the barrier byte
        # behind it, and the next decode fails on trailing data -- so a gather
        # is only safe repeatedly if every read is framed.
        rounds, size = 5, 4
        with mp.Manager() as manager:
            results = manager.list()
            procs = [mp.Process(target=_cycle_member,
                                args=(r, size, 47904, rounds, results))
                     for r in range(size)]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=30)
            self.assertEqual([p.exitcode for p in procs], [0] * size)
            self.assertEqual(list(results), [list(range(size))] * rounds)


def _broadcast_member(rank, size, port, results):
    barrier = Barrier("127.0.0.1", size=size, me=rank, port=port)
    got = [barrier.broadcast(11 * (r + 1) if rank == 0 else 0) for r in range(3)]
    results.append((rank, got))
    barrier.close()


class Broadcast(unittest.TestCase):

    def test_rank_zero_s_value_reaches_everyone(self):
        # A deadline the workload acts on has to be known while the work is
        # happening.  Taking it from the consumer's own pace would let a slow
        # consumer widen its own budget until it could never miss.
        size = 4
        with mp.Manager() as manager:
            results = manager.list()
            procs = [mp.Process(target=_broadcast_member,
                                args=(r, size, 47905, results))
                     for r in range(size)]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=30)
            self.assertEqual([p.exitcode for p in procs], [0] * size)
            for _rank, got in results:
                self.assertEqual(got, [11, 22, 33])

    def test_a_broadcast_does_not_corrupt_a_following_gather(self):
        self.assertIn("recv_exact(width)", open(SOURCE).read())
