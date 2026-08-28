import unittest

from wait.layout import KIB
from wait.model import Constants, FileClass, Regime
from wait import advisor, policies

EXTENT = 128 * KIB
LIMIT = 112 * KIB
CONSTANTS = Constants(inline_limit_bytes=LIMIT, saved_ns_per_access=616_000)

MANIFEST = FileClass("manifest", 4 * KIB, 1, 64, True, Regime.BLOCKING, count=20)
SIDECAR = FileClass("sidecar", 4 * KIB, 50, 1, False, Regime.BLOCKING, count=2000)
S1 = [MANIFEST, SIDECAR]
BUDGET = 20 * 4 * KIB


class Scenario1(unittest.TestCase):

    def test_a_count_ranking_spends_the_whole_budget_on_sidecars(self):
        # The manifest is read once per round against sidecars re-read every
        # iteration, so counts rank it last -- and promoting a re-read file
        # measured indistinguishable from zero.
        got = policies.access_count(S1, BUDGET, EXTENT)
        self.assertEqual(got.promoted, {"sidecar": 20})
        self.assertNotIn("manifest", got.promoted)

    def test_a_size_threshold_cannot_separate_two_4kib_classes(self):
        got = policies.size_threshold(S1, BUDGET, EXTENT, threshold_bytes=128 * KIB)
        self.assertTrue(got.indifferent)
        self.assertNotIn("manifest", got.promoted)

    def test_wait_promotes_the_manifest(self):
        got = advisor.allocate(S1, BUDGET, CONSTANTS)
        self.assertEqual(got.promoted, {"manifest": 20})


class BudgetBinding(unittest.TestCase):

    def test_a_policy_never_spends_more_than_the_budget(self):
        for alloc in (policies.access_count(S1, BUDGET, EXTENT),
                      policies.size_threshold(S1, BUDGET, EXTENT, 128 * KIB),
                      advisor.allocate(S1, BUDGET, CONSTANTS)):
            spent = sum(n * 4 * KIB for n in alloc.promoted.values())
            self.assertLessEqual(spent, BUDGET)

    def test_nothing_promotes_nothing(self):
        self.assertEqual(policies.nothing(S1, BUDGET, EXTENT).promoted, {})


class Indifference(unittest.TestCase):

    def test_equal_counts_leave_a_count_ranking_with_no_basis(self):
        # S2 and S4: two classes identical in size, count and accesses.  An
        # indifferent policy picks at random, so its expectation is the mean of
        # the two arms -- it is not entitled to the better one.
        a = FileClass("blocking", 4 * KIB, 10, 1, False, Regime.BLOCKING, count=100)
        b = FileClass("hidden", 4 * KIB, 10, 1, False, Regime.HIDDEN, count=100)
        self.assertTrue(policies.access_count([a, b], BUDGET, EXTENT).indifferent)
        self.assertTrue(
            policies.size_threshold([a, b], BUDGET, EXTENT, 128 * KIB).indifferent)

    def test_unequal_counts_give_a_count_ranking_a_basis(self):
        self.assertFalse(policies.access_count(S1, BUDGET, EXTENT).indifferent)


if __name__ == "__main__":
    unittest.main()


class TieBreak(unittest.TestCase):

    def test_a_tie_on_count_does_not_break_on_declaration_order(self):
        # Both orders must decide the same way, or the baseline is reading the
        # order the scenario author wrote its classes in.
        big = FileClass("big", 256 * KIB, accesses=200, ranks_coupled=1,
                        synchronized=False, regime=Regime.BLOCKING, count=32)
        small = FileClass("small", 4 * KIB, accesses=200, ranks_coupled=1,
                          synchronized=False, regime=Regime.BLOCKING, count=32)
        forward = policies.access_count([big, small], BUDGET, EXTENT).promoted
        reverse = policies.access_count([small, big], BUDGET, EXTENT).promoted
        self.assertEqual(forward, reverse)
        self.assertIn("small", forward)

    def test_classes_alike_in_count_and_size_are_indifferent(self):
        a = FileClass("a", 4 * KIB, accesses=3, ranks_coupled=1,
                      synchronized=False, regime=Regime.BLOCKING, count=8)
        b = FileClass("b", 4 * KIB, accesses=3, ranks_coupled=1,
                      synchronized=False, regime=Regime.BLOCKING, count=8)
        self.assertTrue(policies.access_count([a, b], BUDGET, EXTENT).indifferent)
