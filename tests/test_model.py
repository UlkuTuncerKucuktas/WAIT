import unittest

from wait.layout import KIB, MIB, Component, Layout, Tier, plain
from wait.model import (FileClass, ModelError, Regime, allocated_objects,
                        cold_accesses, cold_fraction, dom_bytes, extent_for,
                        is_promotable, predict_fstat_ns, value_ns)

ADVISOR = Layout((
    Component(128 * KIB, Tier.MDT),
    Component(8 * MIB, Tier.OST, 1, MIB, "flash"),
    Component(None, Tier.OST, 8, 16 * MIB, "disk"),
))


def blocking(ranks, accesses=1, size=4 * KIB):
    return FileClass("shared", size, accesses, ranks, True, Regime.BLOCKING)


def private(accesses, size=4 * KIB):
    return FileClass("private", size, accesses, 1, False, Regime.BLOCKING)


class Value(unittest.TestCase):

    def test_value_synchronized_charges_ranks_at_barrier_not_cold_fetches(self):
        # 64 ranks over 8 nodes fetch the manifest cold 8 times, but all 64 wait
        # at the barrier.  Charging per cold fetch would report 8 x delta.
        self.assertEqual(value_ns(blocking(ranks=64), 1000), 64_000)

    def test_value_is_linear_in_gating_events_not_quadratic(self):
        # S1 reads a fresh manifest in each of R rounds, so accesses counts gating
        # events.  Squaring them would report 20 x 20 x 64 x delta.
        self.assertEqual(value_ns(blocking(ranks=64, accesses=20), 1000), 1_280_000)
        self.assertEqual(value_ns(blocking(ranks=64, accesses=20), 1000),
                         20 * value_ns(blocking(ranks=64, accesses=1), 1000))

    def test_value_does_not_square_the_rank_count(self):
        # Every rank reads the manifest once per round, so the multiplier is the
        # rank count.  accesses x rank-reads would give 64 x 64 x delta.
        self.assertEqual(value_ns(blocking(ranks=64, accesses=1), 1000), 64_000)

    def test_value_private_class_charges_one_rank_not_the_job_size(self):
        # A private sidecar serves only its owner.  At equal bytes the shared
        # manifest must outrank it by the rank count.
        self.assertEqual(value_ns(private(accesses=1), 1000), 1000)
        self.assertEqual(value_ns(blocking(ranks=64), 1000),
                         64 * value_ns(private(accesses=1), 1000))

    def test_a_hidden_class_is_worth_nothing_however_often_it_is_read(self):
        hot = FileClass("hot", 4 * KIB, 100_000, 1, False, Regime.HIDDEN)
        self.assertEqual(value_ns(hot, 1000), 0)

    def test_an_unsynchronised_class_cannot_claim_many_ranks(self):
        with self.assertRaises(ModelError):
            FileClass("bad", 4 * KIB, 1, 64, False, Regime.BLOCKING)

class Geometry(unittest.TestCase):

    def test_a_plain_layout_allocates_every_object_at_create(self):
        # touch on a -c 24 directory carries 24 obdidx lines: eager, not lazy.
        self.assertEqual(allocated_objects(plain(24, MIB), 0), 24)

    def test_pfl_components_instantiate_lazily(self):
        # Measured on the advisor's own layout: 1 / 2 / 3 components instantiated
        # at 64 KiB / 200 KiB / 64 MiB, so the -c 8 tail costs a small file nothing.
        self.assertEqual(allocated_objects(ADVISOR, 64 * KIB), 0)
        self.assertEqual(allocated_objects(ADVISOR, 200 * KIB), 1)
        self.assertEqual(allocated_objects(ADVISOR, 4 * MIB), 1)
        self.assertEqual(allocated_objects(ADVISOR, 64 * MIB), 9)

    def test_a_file_exactly_filling_a_component_does_not_reach_the_next(self):
        # Extent arithmetic, not measurement: a file of size S occupies offsets
        # [0, S) and component 0 covers [0, E), so S == E still fits entirely in
        # component 0.  Off by one here and every small file is charged the tail.
        self.assertEqual(allocated_objects(ADVISOR, 128 * KIB), 0)
        self.assertEqual(allocated_objects(ADVISOR, 128 * KIB + 1), 1)
        self.assertEqual(allocated_objects(ADVISOR, 8 * MIB), 1)
        self.assertEqual(allocated_objects(ADVISOR, 8 * MIB + 1), 9)

    def test_dom_bytes_capped_by_extent(self):
        # A 25 MB file in a 128 KiB extent costs 128 KiB of MDT, not 25 MB.
        self.assertEqual(dom_bytes(25 * MIB, 128 * KIB), 128 * KIB)
        self.assertEqual(dom_bytes(4 * KIB, 128 * KIB), 4 * KIB)


class Clamp(unittest.TestCase):

    def test_extent_rounds_up_to_the_64k_floor(self):
        self.assertEqual(extent_for(112 * KIB), 128 * KIB)
        self.assertEqual(extent_for(128 * KIB), 128 * KIB)

    def test_promotion_uses_the_limit_not_the_extent_it_rounds_to(self):
        # A 120 KiB file fits the 128 KiB extent but sits above a 112 KiB inline
        # limit: it would burn MDT bytes and still pay a full read round trip.
        self.assertFalse(is_promotable(120 * KIB, 112 * KIB))
        self.assertTrue(is_promotable(64 * KIB, 112 * KIB))
        self.assertFalse(is_promotable(0, 112 * KIB))


class Prediction(unittest.TestCase):

    def test_prediction_charges_only_instantiated_objects(self):
        # The claim A5 tests: a 256 KiB file in the advisor's layout has one
        # object, not nine, so it must not be charged the -c 8 tail.
        small = predict_fstat_ns(ADVISOR, 256 * KIB, 50_000, 4_400)
        wide = predict_fstat_ns(plain(24, MIB), 256 * KIB, 50_000, 4_400)
        self.assertEqual(small, 50_000 + 4_400)
        self.assertEqual(wide, 50_000 + 24 * 4_400)


if __name__ == "__main__":
    unittest.main()


class CachedReReads(unittest.TestCase):

    def test_a_shared_class_discounts_its_cached_re_reads(self):
        # The rank multiplier and the cold discount are different corrections.
        # Charging every access at full price because the class is shared says a
        # gating read served from page cache still gates the job.
        shared = FileClass("m", 4096, accesses=10, ranks_coupled=64,
                           synchronized=True, regime=Regime.BLOCKING, count=1)
        self.assertEqual(value_ns(shared, 1000, cold_fraction=0.1), 64_000)
        self.assertEqual(value_ns(shared, 1000, cold_fraction=1.0), 640_000)

    def test_the_rank_multiplier_survives_the_discount(self):
        one = FileClass("p", 4096, accesses=10, ranks_coupled=1,
                        synchronized=False, regime=Regime.BLOCKING, count=1)
        many = FileClass("m", 4096, accesses=10, ranks_coupled=64,
                         synchronized=True, regime=Regime.BLOCKING, count=1)
        self.assertEqual(value_ns(many, 1000, 0.1) / value_ns(one, 1000, 0.1), 64)


class Discounting(unittest.TestCase):

    def test_a_gating_class_read_once_is_not_discounted_to_zero(self):
        # Truncating the access count before multiplying sends every class with
        # an effective count below one to exactly zero -- and a file read once
        # by every rank is the class the tier exists for.
        fc = FileClass("m", 4096, accesses=1, ranks_coupled=32,
                       synchronized=True, regime=Regime.BLOCKING, count=10 ** 7)
        cold = cold_accesses(fc, 32 * 1024 ** 3) / fc.accesses
        self.assertLess(cold, 1.0)
        self.assertEqual(value_ns(fc, 650000, cold),
                         round(cold * 32 * 650000))

    def test_the_discount_still_applies(self):
        fc = FileClass("m", 4096, accesses=4, ranks_coupled=1,
                       synchronized=False, regime=Regime.BLOCKING)
        self.assertEqual(value_ns(fc, 1000, 0.5), 2000)
        self.assertEqual(value_ns(fc, 1000, 1.0), 4000)


class Discriminator(unittest.TestCase):

    def test_a_shared_class_outweighs_a_private_one_by_exactly_the_ranks(self):
        # The whole separation between WAIT and a count ranking, and it is a
        # property of the value model rather than a measurement: N x delta
        # against 1 x delta, with the access discount cancelling between them.
        for n in (4, 16, 32, 64):
            shared = FileClass("gate", 4096, accesses=1, ranks_coupled=n,
                               synchronized=True, regime=Regime.BLOCKING, count=200)
            private = FileClass("own", 4096, accesses=200, ranks_coupled=1,
                                synchronized=False, regime=Regime.BLOCKING,
                                count=200 * n)
            ratio = (value_ns(shared, 1000) / shared.accesses) / \
                    (value_ns(private, 1000) / private.accesses)
            self.assertAlmostEqual(ratio, n, places=6)
