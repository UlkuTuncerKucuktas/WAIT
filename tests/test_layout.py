import unittest

from wait.layout import (KIB, MIB, Component, Layout, LayoutError, Tier,
                         dom, format_size, plain)


class SpecEmission(unittest.TestCase):

    def test_spec_never_emits_S_on_a_dom_component(self):
        # lfs refuses the entire layout, not just the flag:
        # "Option 'stripe-size' can't be specified with Data-on-MDT component"
        text = dom(128 * KIB, 1, MIB).spec()
        self.assertEqual(text, "-E 128K -L mdt -E -1 -c 1 -S 1M")
        self.assertNotIn("-L mdt -S", text)

    def test_spec_always_emits_S_on_non_dom_components(self):
        # Omitted, the component inherits the DoM extent as its stripe size:
        # -E 128K -L mdt -E -1 -c 4 was granted stripe_size 131072.
        text = Layout((
            Component(128 * KIB, Tier.MDT),
            Component(8 * MIB, Tier.OST, 1, MIB, "flash"),
            Component(None, Tier.OST, 8, 16 * MIB, "disk"),
        )).spec()
        self.assertEqual(
            text,
            "-E 128K -L mdt -E 8M -c 1 -S 1M -p flash -E -1 -c 8 -S 16M -p disk")

    def test_a_plain_layout_emits_no_extent(self):
        self.assertEqual(plain(1, MIB).spec(), "-c 1 -S 1M")

    def test_format_size_prefers_the_largest_exact_unit(self):
        self.assertEqual(format_size(131072), "128K")
        self.assertEqual(format_size(1048576), "1M")
        self.assertEqual(format_size(100), "100")


class Rejections(unittest.TestCase):

    def test_an_mdt_component_cannot_carry_a_stripe_size(self):
        with self.assertRaises(LayoutError):
            Layout((Component(128 * KIB, Tier.MDT, stripe_bytes=64 * KIB),
                    Component(None, Tier.OST, 1, MIB)))

    def test_a_dom_extent_below_the_64k_floor_is_refused(self):
        # lfs: "invalid component end '32K'"
        with self.assertRaises(LayoutError):
            dom(32 * KIB, 1, MIB)

    def test_an_ost_component_without_a_stripe_size_is_refused(self):
        with self.assertRaises(LayoutError):
            Layout((Component(128 * KIB, Tier.MDT),
                    Component(None, Tier.OST, stripe_count=4)))

    def test_a_component_end_must_align_to_its_stripe_size(self):
        # lfs: "The component end must be aligned by the stripe size"
        with self.assertRaises(LayoutError):
            Layout((Component(3 * MIB, Tier.OST, 1, 2 * MIB),
                    Component(None, Tier.OST, 4, MIB)))

    def test_the_mdt_component_must_come_first(self):
        with self.assertRaises(LayoutError):
            Layout((Component(MIB, Tier.OST, 1, MIB),
                    Component(None, Tier.MDT)))

    def test_the_last_component_must_run_to_eof(self):
        with self.assertRaises(LayoutError):
            Layout((Component(MIB, Tier.OST, 1, MIB),))


class Geometry(unittest.TestCase):

    def test_starts_follow_the_previous_end(self):
        layout = Layout((
            Component(128 * KIB, Tier.MDT),
            Component(8 * MIB, Tier.OST, 1, MIB),
            Component(None, Tier.OST, 8, 16 * MIB),
        ))
        self.assertEqual(layout.starts(), (0, 128 * KIB, 8 * MIB))
        self.assertEqual(layout.dom_extent_bytes(), 128 * KIB)
        self.assertEqual(plain(1, MIB).dom_extent_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
