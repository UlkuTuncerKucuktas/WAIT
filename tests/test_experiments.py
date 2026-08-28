import importlib
import unittest

from wait import ledger
from wait.model import allocated_objects

MODULES = ("a1_validity", "a2_tier_benefit", "a4_tier_bytes",
           "a5_model_check", "a6_write_benefit")


def load(name):
    return importlib.import_module("wait.experiments." + name)


class Contract(unittest.TestCase):

    def test_every_experiment_declares_cells_repeats_prepare_and_measure(self):
        for name in MODULES:
            module = load(name)
            self.assertGreater(len(module.cells), 0, name)
            self.assertGreaterEqual(module.repeats, 2, name)
            self.assertTrue(callable(module.prepare), name)
            self.assertTrue(callable(module.measure), name)

    def test_no_two_cells_share_a_ledger_key(self):
        # A collision silently overwrites one arm with another, and the grid
        # reports as complete.
        for name in MODULES:
            module = load(name)
            keys = [ledger.key(name, cell, repeat, getattr(cell, "arm", None))
                    for repeat in range(module.repeats) for cell in module.cells]
            self.assertEqual(len(keys), len(set(keys)), name)


class Grids(unittest.TestCase):

    def test_a1_covers_a_wrong_arm_a_right_arm_and_the_harness_floor(self):
        arms = {c.arm for c in load("a1_validity").cells}
        self.assertEqual(arms, {"wrong", "right", "floor"})

    def test_a2_spans_the_inline_boundary_on_both_sides(self):
        sizes = sorted({c.size_bytes for c in load("a2_tier_benefit").cells})
        self.assertLess(min(sizes), 64 * 1024)
        self.assertGreater(max(sizes), 128 * 1024)
        self.assertIn(96 * 1024, sizes)

    def test_a2_holds_data_in_one_object_across_the_width_arm(self):
        # Every width cell is a 1 MiB file at S=1M, so only allocation varies and
        # the arm cannot be confounded with a placement effect.
        module = load("a2_tier_benefit")
        width = [c for c in module.cells if c.arm == "width"]
        self.assertEqual({c.size_bytes for c in width}, {1024 * 1024})
        self.assertEqual({c.stripe_count for c in width}, {1, 8, 24})

    def test_a4_spans_both_sides_of_every_extent(self):
        # min(size, extent) is only demonstrated by a file that spills past it.
        cells = load("a4_tier_bytes").cells
        for extent in {c.extent_bytes for c in cells}:
            sizes = [c.size_bytes for c in cells if c.extent_bytes == extent]
            self.assertTrue(any(s < extent for s in sizes), extent)
            self.assertTrue(any(s > extent for s in sizes), extent)

    def test_a4_writes_enough_of_each_size_to_clear_drift(self):
        # Background drift on a shared MDT ran 1-2 MB per fifteen seconds.
        for cell in load("a4_tier_bytes").cells:
            mdt_bytes = cell.files * min(cell.size_bytes, cell.extent_bytes)
            self.assertGreater(mdt_bytes, 8 * 1024 ** 2, cell)

    def test_every_layout_a_grid_builds_is_valid(self):
        for name in ("a1_validity", "a2_tier_benefit", "a6_write_benefit"):
            module = load(name)
            for cell in module.cells:
                tier = getattr(cell, "tier", None)
                if tier == "shm":
                    continue
                spec = (module.layout_for(tier) if tier
                        else module.layout_for(cell)).spec()
                self.assertIn("-c ", spec)


class Predictions(unittest.TestCase):

    def test_a5_predicts_a_different_object_count_per_layout(self):
        # The check is only meaningful if the layouts disagree: at 256 KiB the
        # advisor's shape carries one object where a plain c=16 carries sixteen.
        module = load("a5_model_check")
        small = 256 * 1024
        counts = {name: allocated_objects(layout, small)
                  for name, layout in module.LAYOUTS.items()}
        self.assertEqual(counts["advisor"], 1)
        self.assertEqual(counts["wide"], 16)

    def test_no_a5_cell_was_already_measured_by_a2(self):
        # A5 predicts layouts it has never seen.  The advisor's shape appears in
        # A2 as well, so the cells must differ by size -- A2 sweeps the inline
        # boundary and instantiates no later component.
        a5, a2 = load("a5_model_check"), load("a2_tier_benefit")
        measured = {(a2.layout_for(c).spec(), c.size_bytes) for c in a2.cells}
        for cell in a5.cells:
            self.assertNotIn((a5.layout_for(cell).spec(), cell.size_bytes),
                             measured, cell)

    def test_a5_reaches_sizes_that_instantiate_later_components(self):
        a5 = load("a5_model_check")
        counts = {allocated_objects(a5.layout_for(c), c.size_bytes)
                  for c in a5.cells if c.layout == "advisor"}
        self.assertIn(9, counts)


if __name__ == "__main__":
    unittest.main()
