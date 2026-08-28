import csv
import pathlib
import unittest

from wait.layout import KIB, MIB, Component, Layout, LayoutError, Tier, dom
from wait.model import allocated_objects, is_promotable

PROBES = pathlib.Path(__file__).resolve().parent.parent / "out" / "probes"

ADVISOR = Layout((
    Component(128 * KIB, Tier.MDT),
    Component(8 * MIB, Tier.OST, 1, MIB, "flash"),
    Component(None, Tier.OST, 8, 16 * MIB, "disk"),
))


def rows(name):
    with open(next(PROBES.glob("*_%s.csv" % name))) as fh:
        return list(csv.DictReader(fh))


class AgreesWithTheMachine(unittest.TestCase):
    """The model must reproduce what was measured, or one of them is wrong."""

    def test_lazy_instantiation_matches_the_measured_object_counts(self):
        for row in rows("lazy_instantiation"):
            if not row["components_instantiated"]:
                continue
            size = int(row["file_size_kib"]) * KIB
            self.assertEqual(allocated_objects(ADVISOR, size),
                             int(row["ost_objects"]),
                             "%s KiB" % row["file_size_kib"])

    def test_the_inline_limit_lies_between_the_measured_bounds(self):
        measured = rows("inline_limit_by_extent")
        inlined = {int(r["file_size_kib"]) for r in measured if r["inlined"] == "yes"}
        spilled = {int(r["file_size_kib"]) for r in measured if r["inlined"] == "no"}
        # Every extent inlines 64 KiB and none inlines 128 KiB, so the boundary is
        # in (64, 128] and does not move with the extent.
        self.assertEqual(max(inlined), 64)
        self.assertEqual(min(spilled), 128)
        for limit in (64 * KIB, 96 * KIB, 112 * KIB):
            self.assertTrue(is_promotable(64 * KIB, limit))
            self.assertFalse(is_promotable(128 * KIB, limit))

    def test_extents_the_server_refused_are_refused_here_too(self):
        for row in rows("extents_granted"):
            if row["truncated"] != "refused":
                continue
            size = int(row["requested"].rstrip("K")) * KIB
            with self.assertRaises(LayoutError, msg=row["requested"]):
                dom(size, 1, MIB)

    def test_a_re_read_promotion_is_indistinguishable_from_zero(self):
        by = {(r["arm"], r["metric"]): float(r["value_us"])
              for r in rows("s1_barrier_release")}
        gain = by[("ost", "sidecar_reread")] - by[("dom", "sidecar_reread")]
        # Reported as indistinguishable from zero rather than as a signed value:
        # it sits far below the 10-20 percent independent-set spread.
        self.assertLess(abs(gain), 0.10 * by[("ost", "sidecar_reread")])


if __name__ == "__main__":
    unittest.main()
