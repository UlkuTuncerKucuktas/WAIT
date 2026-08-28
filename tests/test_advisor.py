import unittest

from wait.layout import KIB, MIB
from wait.model import Constants, FileClass, Regime
from wait import advisor

LIMIT = 112 * KIB
CONSTANTS = Constants(inline_limit_bytes=LIMIT, saved_ns_per_access=616_000)


def cls(name, regime, size=4 * KIB, accesses=1, ranks=1, sync=False, count=10):
    return FileClass(name, size, accesses, ranks, sync, regime, count=count)


class Ordering(unittest.TestCase):

    def test_a_hidden_class_is_never_promoted(self):
        # Declined, not demoted: a wrong hidden label costs foregone benefit,
        # never added stall against the site default.
        hot = cls("hidden", Regime.HIDDEN, accesses=100_000)
        got = advisor.allocate([hot], 10 * 4 * KIB, CONSTANTS)
        self.assertEqual(got.promoted, {})

    def test_deadline_classes_are_allocated_before_density_ranking(self):
        # A deadline is a step, not a slope, so it cannot be traded off against
        # a blocking class that happens to score higher per byte.
        deadline = cls("index", Regime.DEADLINE, count=5)
        dense = cls("blocking", Regime.BLOCKING, ranks=64, sync=True, count=5)
        got = advisor.allocate([dense, deadline], 5 * 4 * KIB, CONSTANTS)
        self.assertEqual(got.promoted, {"index": 5})

    def test_a_shared_class_outranks_a_private_one_at_equal_bytes(self):
        shared = cls("shared", Regime.BLOCKING, ranks=64, sync=True, count=10)
        private = cls("private", Regime.BLOCKING, accesses=8, count=10)
        got = advisor.allocate([private, shared], 10 * 4 * KIB, CONSTANTS)
        self.assertEqual(got.promoted, {"shared": 10})


class Clamp(unittest.TestCase):

    def test_a_class_above_the_inline_limit_is_not_promoted(self):
        # 120 KiB fits the 128 KiB extent the limit rounds up to, but sits above
        # the limit itself: it would burn MDT bytes and still pay a round trip.
        big = cls("big", Regime.BLOCKING, size=120 * KIB, ranks=64, sync=True)
        small = cls("small", Regime.BLOCKING, size=4 * KIB, ranks=64, sync=True)
        got = advisor.allocate([big, small], 10 * 120 * KIB, CONSTANTS)
        self.assertNotIn("big", got.promoted)
        self.assertEqual(got.promoted, {"small": 10})


class Emission(unittest.TestCase):

    def test_the_promoted_layout_rounds_the_extent_to_the_64k_floor(self):
        self.assertEqual(advisor.promoted_layout(CONSTANTS, 1, MIB).spec(),
                         "-E 128K -L mdt -E -1 -c 1 -S 1M")

    def test_the_default_layout_carries_no_mdt_component(self):
        layout = advisor.default_layout(1, MIB, "flash")
        self.assertFalse(layout.has_dom())
        self.assertEqual(layout.spec(), "-c 1 -S 1M -p flash")


if __name__ == "__main__":
    unittest.main()
