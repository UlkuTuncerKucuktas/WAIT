import ast
import inspect
import os
import tempfile
import unittest
from unittest import mock

from wait import probe
from wait.experiments import s4_ensemble as s4

RANKS = 4


def _calls_fsync(func):
    # By call, not by substring: the comment on write_buffered says "No fsync",
    # and a substring check passes for the wrong reason.
    tree = ast.parse(inspect.getsource(func).lstrip())
    return any(isinstance(n, ast.Attribute) and n.attr == "fsync"
               for n in ast.walk(tree))


def prepared(cell):
    work = tempfile.mkdtemp()
    with mock.patch.object(s4.lustre, "setstripe", lambda p, l: None), \
         mock.patch.object(s4.ranks, "world", lambda: RANKS):
        s4.prepare(cell, work)
    return work


class Indifference(unittest.TestCase):

    def test_the_two_classes_are_the_same_size(self):
        # A size threshold must have no basis to prefer one over the other.
        self.assertEqual(s4.RESULT_BYTES, s4.DIAGNOSTIC_BYTES)

    def test_the_two_classes_have_the_same_count(self):
        # And a count ranking must have none either: one diagnostic per result,
        # so neither heuristic can discriminate and its expectation is the mean
        # of the two promotable arms.
        cell = s4.Cell("wait")
        work = prepared(cell)
        with mock.patch.object(s4.ranks, "world", lambda: RANKS):
            per_class = {name: len({s4.path_for(work, name, r, g, t, s4.plan(cell))
                                    for r in range(RANKS)
                                    for g in range(cell.generations)
                                    for t in range(cell.tasks_per_rank)})
                         for name in s4.CLASSES}
        self.assertEqual(per_class["results"], per_class["diagnostics"])


class Arms(unittest.TestCase):

    def test_the_default_arm_promotes_nothing(self):
        self.assertEqual(set(s4.plan(s4.Cell("default"))["counts"].values()), {0})

    def test_the_arms_promote_disjoint_classes(self):
        picked = [{n for n, k in s4.plan(s4.Cell(a))["counts"].items() if k}
                  for a in ("heuristic", "wait")]
        self.assertEqual(set.union(*picked), set(s4.CLASSES))
        self.assertFalse(set.intersection(*picked))

    def test_the_size_arm_promotes_part_of_both_classes(self):
        placing = s4.plan(s4.Cell("size"))
        for name, k in placing["counts"].items():
            self.assertGreater(k, 0, name)
            self.assertLess(k, placing["total"], name)

    def test_each_promoting_arm_spends_the_same_budget(self):
        with mock.patch.object(s4.ranks, "world", lambda: RANKS):
            budgets = {s4.budget_files(s4.Cell(a)) for a in ("heuristic", "wait")}
        self.assertEqual(len(budgets), 1)

    def test_no_directory_holds_two_layouts(self):
        # Promotion is by directory, so no file names its own layout -- which
        # would cost its open 2.7x at scale and, if the write truncated, its
        # inlining outright.
        for arm in ("default", "heuristic", "size", "wait"):
            cell = s4.Cell(arm)
            work = prepared(cell)
            placing = s4.plan(cell)
            for name in s4.CLASSES:
                for r in range(RANKS):
                    got = s4.promoted_in_rank(placing, name)
                    rooms = {}
                    for g in range(cell.generations):
                        for t in range(cell.tasks_per_rank):
                            path = s4.path_for(work, name, r, g, t, placing)
                            at = g * placing["tasks"] + t
                            rooms.setdefault(os.path.dirname(path), set()).add(at < got)
                    for room, tiers in rooms.items():
                        self.assertEqual(len(tiers), 1,
                                         "%s: %s holds both" % (arm, room))


class Durability(unittest.TestCase):

    def test_results_are_fsynced_and_diagnostics_are_not(self):
        # The whole scenario: the class every rank waits on is durable, and the
        # class nobody waits on defers its cost.  fsync both and the arms carry
        # the same value; fsync neither and there is nothing to wait for.
        self.assertTrue(_calls_fsync(probe.write_staged))
        self.assertFalse(_calls_fsync(probe.write_buffered))
        body = inspect.getsource(s4.measure)
        self.assertIn('write_staged(\n                path_for(workdir, "results"', body)
        self.assertIn('write_buffered(\n                path_for(workdir, "diagnostics"', body)

    def test_prepare_makes_every_directory_measure_writes_into(self):
        # prepare creates directories and measure creates the files in them; a
        # class whose directory is missing fails mid-generation with the
        # allocation already spent.
        cell = s4.Cell("wait")
        work = prepared(cell)
        wanted = {os.path.dirname(s4.path_for(work, name, r, g, t, s4.plan(cell)))
                  for name in s4.CLASSES for r in range(RANKS)
                  for g in range(cell.generations)
                  for t in range(cell.tasks_per_rank)}
        made = {os.path.join(root, d)
                for root, ds, _f in os.walk(work) for d in ds}
        self.assertTrue(wanted <= made, wanted - made)


class Repeats(unittest.TestCase):

    def test_enough_repeats_to_estimate_a_spread(self):
        # Three leaves max-minus-min an estimate over two gaps, and S4's arm
        # difference does not grow with the rank count while the noise in a
        # max over ranks does.  S3 runs seven for the same reason.
        self.assertGreaterEqual(s4.repeats, 7)
