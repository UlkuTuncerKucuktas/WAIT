import ast
import inspect
import os
import tempfile
import unittest
from unittest import mock

from wait.experiments import s2_hidden as s2
from wait.model import Constants
from wait.model import is_promotable


def prepared(cell):
    work = tempfile.mkdtemp()
    with mock.patch.object(s2.lustre, "setstripe", lambda p, l: None):
        s2.prepare(cell, work)
    return work


class Indifference(unittest.TestCase):

    def test_the_classes_match_in_size_count_and_accesses(self):
        # All three are what a heuristic could discriminate on.  Matching them
        # is the scenario: neither the size threshold nor the count ranking has
        # a basis to prefer one class, so each runs as its own measured arm.
        cell = s2.Cell("wait", 2.0)
        work = prepared(cell)
        placing = s2.plan(cell)
        written = {name: len(os.listdir(os.path.join(work, name)))
                   for name in s2.CLASSES}
        self.assertEqual(written["tiles"], written["masks"])
        sizes = {name: os.path.getsize(
            s2.path_for(work, name, 0, placing))
            for name in s2.CLASSES}
        self.assertEqual(sizes["tiles"], sizes["masks"])
        # Each item is read exactly once, on either side.
        body = inspect.getsource(s2.measure) + inspect.getsource(s2._prefetch)
        self.assertEqual(body.count("read_staged("), 2)


class Eligibility(unittest.TestCase):

    def test_both_classes_are_small_enough_to_promote(self):
        # Above the inline limit the advisor declines a class outright, so a
        # scenario whose files sit above it has nothing to allocate and measures
        # the same arm three times.
        self.assertTrue(is_promotable(s2.ITEM_BYTES, 114688))


class Arms(unittest.TestCase):

    def test_the_default_arm_promotes_nothing(self):
        self.assertEqual(set(s2.plan(s2.Cell("default", 2.0))["counts"].values()),
                         {0})

    def test_the_arms_promote_disjoint_classes(self):
        picked = [{n for n, k in s2.plan(s2.Cell(a, 2.0))["counts"].items() if k}
                  for a in ("heuristic", "wait")]
        self.assertEqual(set.union(*picked), set(s2.CLASSES))
        self.assertFalse(set.intersection(*picked))

    def test_the_size_arm_promotes_part_of_both_classes(self):
        # It cannot rank two classes of one size, so it fills both -- and a
        # scenario that collapses the allocation to one name would apply
        # whichever class sorts first and measure some other arm entirely.
        counts = s2.plan(s2.Cell("size", 2.0))["counts"]
        self.assertEqual(set(counts), set(s2.CLASSES))
        for name, k in counts.items():
            self.assertGreater(k, 0, name)
            self.assertLess(k, s2.ITEMS, name)

    def test_no_class_is_named_after_its_own_answer(self):
        # The classes were called "hidden" and "blocking", which put the answer
        # to the agent's first question in the class name -- and the agent reads
        # this source.  Names describe what a file holds, never what it costs.
        forbidden = ("hidden", "blocking", "deadline", "sync", "critical")
        for name in s2.CLASSES:
            self.assertFalse([w for w in forbidden if w in name.lower()], name)

    def test_wait_promotes_the_class_that_blocks(self):
        # The whole thesis in one assertion: the other class is larger by no
        # measure the heuristics can see, and worth nothing to promote.
        counts = s2.plan(s2.Cell("wait", 2.0))["counts"]
        self.assertEqual({n for n, k in counts.items() if k}, {"masks"})

    def test_no_directory_holds_two_layouts(self):
        # A split class gets a directory per layout; what is forbidden is one
        # directory carrying both, which forces a per-file setstripe.
        for arm in ("default", "heuristic", "size", "wait"):
            cell = s2.Cell(arm, 2.0)
            placing = s2.plan(cell)
            work = prepared(cell)
            for name in s2.CLASSES:
                by_dir = {}
                for i in range(s2.ITEMS):
                    path = s2.path_for(work, name, i, placing)
                    by_dir.setdefault(os.path.dirname(path), set()).add(
                        i < placing["counts"][name])
                for directory, tiers in by_dir.items():
                    self.assertEqual(len(tiers), 1, directory)


class Rates(unittest.TestCase):

    def test_the_result_is_checked_at_more_than_one_compute_ratio(self):
        # Compute is the parameter a reviewer would suspect of carrying the
        # result, so it is varied.  It cannot un-hide the class: the prefetcher
        # reads one item per iteration while the main loop reads one and
        # computes, so the prefetcher is never the slower side at any ratio.
        self.assertGreater(len({c.compute_ratio for c in s2.cells}), 1)

    def test_the_hidden_side_costs_no_more_per_item_than_the_blocking_side(self):
        # Which is what keeps the two classes indifferent to a heuristic -- and
        # is also why no compute ratio can starve the consumer.
        body = inspect.getsource(s2._prefetch)
        self.assertIn("ITEM_BYTES", body)
        self.assertNotIn("for _ in range", body)

    def test_the_prefetcher_is_a_process_not_a_thread(self):
        # It must overlap the main loop's compute rather than interleave with it.
        tree = ast.parse(inspect.getsource(s2.measure))
        self.assertTrue(any(isinstance(n, ast.Attribute) and n.attr == "Process"
                            for n in ast.walk(tree)))
