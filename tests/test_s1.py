import os
import unittest
from unittest import mock

from wait.experiments import s1_barrier as s1
from wait.layout import KIB


class Workload(unittest.TestCase):

    def test_a_fresh_manifest_per_round(self):
        # Re-reading one manifest is served from the page cache after the first
        # round, so the sum over rounds would accumulate noise rather than delta.
        cell = s1.Cell("wait")
        placing = s1.plan(cell)
        paths = [s1.manifest("/w", r, placing) for r in range(cell.rounds)]
        self.assertEqual(len(set(paths)), cell.rounds)
        promoted = s1.promoted_paths(cell, "/w")
        self.assertEqual(len(set(promoted)), cell.rounds)

    def test_sidecars_are_private_to_a_rank(self):
        # Shared sidecars would carry the barrier multiplier too, and the
        # discriminator between the two classes would vanish.
        placing = s1.plan(s1.cells[0])
        a = {s1.sidecar("/w", 0, i, placing) for i in range(50)}
        b = {s1.sidecar("/w", 1, i, placing) for i in range(50)}
        self.assertFalse(a & b)

    def test_manifests_and_sidecars_are_the_same_size(self):
        # A size threshold must have no basis to prefer one over the other.
        self.assertEqual(s1.MANIFEST_BYTES, s1.SIDECAR_BYTES)


RANK_COUNTS = (4, 16, 32)


def paths(arm, world):
    with mock.patch.object(s1.ranks, "world", lambda: world):
        return s1.promoted_paths(s1.Cell(arm), "/w")


class Arms(unittest.TestCase):

    def test_the_default_arm_promotes_nothing(self):
        for world in RANK_COUNTS:
            self.assertEqual(paths("default", world), [])

    def test_wait_promotes_every_manifest(self):
        for world in RANK_COUNTS:
            promoted = paths("wait", world)
            self.assertEqual(len(promoted), s1.budget_files(s1.Cell("wait")))
            self.assertTrue(all("manifests/" in p for p in promoted))

    def test_the_arms_promote_disjoint_classes(self):
        for world in RANK_COUNTS:
            h, w = set(paths("heuristic", world)), set(paths("wait", world))
            self.assertFalse(h & w, world)
            self.assertTrue(all("sidecars/" in p for p in h), world)

    def test_the_competing_class_can_absorb_the_whole_budget(self):
        # The budget is one class's bytes.  If the competing class is smaller
        # than that, the count ranking spends what is left on the manifests and
        # captures the benefit by accident -- at four ranks, ten sidecars each is
        # forty files against a budget of two hundred.
        for world in RANK_COUNTS:
            cell = s1.Cell("heuristic")
            with mock.patch.object(s1.ranks, "world", lambda: world):
                available = s1.sidecars_per_rank(cell) * world
            self.assertGreaterEqual(available, s1.budget_files(cell), world)

    def test_the_heuristic_never_spills_into_the_class_wait_wants(self):
        for world in RANK_COUNTS:
            self.assertFalse([p for p in paths("heuristic", world)
                              if "manifests/" in p], world)

    def test_each_arm_spends_the_budget_it_is_charged_for(self):
        # Promotion is whole directories, so the heuristic can fall short of the
        # budget by less than one rank's worth of sidecars and no more.
        for world in RANK_COUNTS:
            cell = s1.Cell("heuristic")
            with mock.patch.object(s1.ranks, "world", lambda: world):
                per_rank = s1.sidecars_per_rank(cell)
            spent = len(paths("heuristic", world))
            self.assertLessEqual(s1.budget_files(cell) - spent, per_rank, world)

    def test_no_directory_holds_two_layouts(self):
        # A mixed directory forces a per-file setstripe, and that costs the
        # promoted open 612.9 us against 223.4 at thirty-two ranks -- and if the
        # write then truncates, the file keeps every mark of the fast tier and
        # none of its benefit.  So a promoted class must be whole directories.
        import os
        for world in RANK_COUNTS:
            for arm in ("default", "heuristic", "size", "wait"):
                cell = s1.Cell(arm)
                with mock.patch.object(s1.ranks, "world", lambda: world):
                    per_rank = s1.sidecars_per_rank(cell)
                    placing = s1.plan(cell)
                    promoted = set(s1.promoted_paths(cell, "/w"))
                    every = ([s1.manifest("/w", r, placing)
                              for r in range(cell.rounds)]
                             + [s1.sidecar("/w", k, i, placing)
                                for k in range(world) for i in range(per_rank)]
                             + [s1.shard("/w", k, i) for k in range(world)
                                for i in range(cell.shards_per_rank)])
                self.assertTrue(promoted <= set(every), (arm, world))
                rooms = {}
                for path in every:
                    rooms.setdefault(os.path.dirname(path), set()).add(path in promoted)
                for room, kinds in rooms.items():
                    self.assertEqual(len(kinds), 1,
                                     "%s at %d: %s holds both" % (arm, world, room))

    def test_the_grid_is_exactly_the_four_arms(self):
        # The size threshold runs as an arm rather than being scored by an
        # indifference argument, so "beats both baselines" is a measurement.
        self.assertEqual({c.arm for c in s1.cells},
                         {"default", "heuristic", "size", "wait"})
        self.assertEqual(len(s1.cells), 4)


class Composition(unittest.TestCase):

    def test_a_round_reports_how_many_ranks_its_maximum_stands_for(self):
        # One cold fetch per node, so two ranks should sit at the top however
        # many ranks there are.  More than that and the maximum has become a
        # cached read, and the absolute stops being comparable across N.
        cold, warm = 900, 100
        rounds = {0: [cold, cold] + [warm] * 30}
        self.assertEqual(s1.ranks_near_max(rounds), 2)
        self.assertGreater(s1.spread_ratio(rounds), 5)

    def test_an_all_cached_round_is_visible_as_one(self):
        rounds = {0: [100, 101, 99, 102]}
        self.assertEqual(s1.ranks_near_max(rounds), 4)
        self.assertLess(s1.spread_ratio(rounds), 1.1)


class Release(unittest.TestCase):

    def test_a_round_costs_the_slowest_rank_not_the_fastest(self):
        # Taking the minimum would report the one rank that hit page cache and
        # miss the fetch every other rank is waiting on.
        rounds = {0: [100, 900], 1: [200, 800]}
        self.assertEqual(s1.release_ns(rounds), 1700)

    def test_release_sums_over_rounds(self):
        self.assertEqual(s1.release_ns({0: [5], 1: [7], 2: [11]}), 23)

    def test_the_typical_round_is_reported_beside_the_slowest(self):
        # The max is what the barrier costs; the median resolves the tier without
        # drawing from the tail.  Both are recorded so one cannot hide the other.
        rounds = {0: [100, 500, 900], 1: [100, 300, 1100]}
        self.assertEqual(s1.release_ns(rounds), 2000)
        self.assertEqual(s1.typical_ns(rounds), 800)

    def test_enough_rounds_that_the_tail_averages_out(self):
        # Noise over R rounds grows as sqrt(R) and signal as R, so a
        # twenty-round run cannot resolve a 300 us effect against 500 us spread.
        self.assertGreaterEqual(s1.ROUNDS, 100)


class TimedRegion(unittest.TestCase):

    def test_ranks_are_lined_up_before_the_timed_read(self):
        # Two barriers per round: one to synchronise before timing the gating
        # read, one to release.  With only the release barrier, each rank reaches
        # the next manifest at a different moment and the private reads of the
        # others land inside the measurement.
        import inspect
        body = inspect.getsource(s1.measure)
        before = body.index("barrier.wait()")
        timed = body.index("t0 = probe.NS()")
        after = body.index("barrier.wait()", timed)
        self.assertLess(before, timed)
        self.assertLess(timed, after)

    def test_private_reads_are_outside_the_timed_region(self):
        import inspect
        body = inspect.getsource(s1.measure)
        self.assertLess(body.index("mine.append"), body.index("sidecar(workdir, me"))


class ColdFetches(unittest.TestCase):

    def test_the_coordinator_is_configurable(self):
        # Ranks span two reader nodes, so loopback would have each node
        # rendezvous with itself.
        self.assertTrue(hasattr(s1, "COORDINATOR"))

    def test_the_submit_script_uses_two_reader_nodes(self):
        # With one reader only the first rank of each round pays a cold fetch and
        # the rest hit page cache, so the maximum over ranks -- what the barrier
        # costs -- is buried in the spread of cached reads.
        import pathlib
        script = (pathlib.Path(__file__).resolve().parent.parent
                  / "scripts" / "submit_scenario.sh").read_text()
        self.assertIn("#SBATCH -N 3", script)
        self.assertIn('srun -N2 -n"$TASKS"', script)


class Scaling(unittest.TestCase):

    def test_the_rank_count_is_not_a_cell_field(self):
        # It comes from SLURM_NTASKS: ranks must be tasks with real cores, or the
        # barrier reports scheduler latency instead of the fetch.
        self.assertNotIn("ranks", s1.Cell.__dataclass_fields__)

    def test_the_budget_does_not_grow_with_the_rank_count(self):
        # What WAIT recovers scales with N; what the heuristics recover does not.
        # A budget that grew with N would hide exactly that.
        budgets = {s1.budget_files(c) for c in s1.cells}
        self.assertEqual(len(budgets), 1)


if __name__ == "__main__":
    unittest.main()
