import unittest

from wait import advisor, arms, policies
from wait.layout import KIB
from wait.model import Constants, FileClass, Regime, extent_for

LIMIT = 114688
CONSTANTS = Constants(inline_limit_bytes=LIMIT, saved_ns_per_access=651_000,
                      saved_write_ns_per_access=273_000)
EXTENT = extent_for(LIMIT)


def klass(name, regime, accesses=1, ranks=1, sync=False, count=200, size=4 * KIB,
          writes=False):
    return FileClass(name, size, accesses=accesses, ranks_coupled=ranks,
                     synchronized=sync, regime=regime, count=count, writes=writes)


DISCRIMINATING = (klass("gate", Regime.BLOCKING, ranks=32, sync=True),
                  klass("private", Regime.BLOCKING, accesses=200, count=320))
INDIFFERENT = (klass("blocks", Regime.BLOCKING),
               klass("hidden", Regime.HIDDEN))
BUDGET = 200 * 4 * KIB


class Derivation(unittest.TestCase):

    def test_the_default_arm_promotes_nothing(self):
        self.assertEqual(
            arms.allocation("default", DISCRIMINATING, BUDGET, CONSTANTS).promoted, {})

    def test_the_wait_arm_is_exactly_what_the_advisor_returns(self):
        # If the scenario could name its own answer, the paper would be measuring
        # its author rather than the advisor.
        self.assertEqual(
            arms.allocation("wait", DISCRIMINATING, BUDGET, CONSTANTS).promoted,
            advisor.allocate(DISCRIMINATING, BUDGET, CONSTANTS).promoted)

    def test_a_discriminating_heuristic_is_exactly_what_the_policy_returns(self):
        self.assertEqual(
            arms.allocation("heuristic", DISCRIMINATING, BUDGET, CONSTANTS).promoted,
            policies.access_count(DISCRIMINATING, BUDGET, EXTENT).promoted)

    def test_an_unknown_arm_is_refused(self):
        with self.assertRaises(ValueError):
            arms.allocation("oracle", DISCRIMINATING, BUDGET, CONSTANTS)


class Indifference(unittest.TestCase):

    def test_the_heuristic_takes_the_outcome_wait_did_not(self):
        # An indifferent rule flips a coin, so its expectation is the mean of the
        # outcomes and both must be measured.  Reporting whichever outcome
        # flatters the comparison would be the easiest way to say something
        # untrue.
        wait = arms.allocation("wait", INDIFFERENT, BUDGET, CONSTANTS)
        heuristic = arms.allocation("heuristic", INDIFFERENT, BUDGET, CONSTANTS)
        self.assertEqual(set(wait.promoted), {"blocks"})
        self.assertEqual(set(heuristic.promoted), {"hidden"})
        self.assertTrue(heuristic.indifferent)

    def test_the_two_outcomes_never_overlap(self):
        wait = arms.allocation("wait", INDIFFERENT, BUDGET, CONSTANTS)
        heuristic = arms.allocation("heuristic", INDIFFERENT, BUDGET, CONSTANTS)
        self.assertFalse(set(wait.promoted) & set(heuristic.promoted))


class Prediction(unittest.TestCase):

    def test_a_hidden_class_is_predicted_to_recover_nothing(self):
        self.assertEqual(
            arms.predicted_value_ns("heuristic", INDIFFERENT, BUDGET, CONSTANTS), 0)

    def test_a_written_class_is_charged_the_write_saving(self):
        # The read saving is 2.4x the durable-write one here, so charging a
        # written class the read constant overstates it by that much.
        written = (klass("results", Regime.BLOCKING, writes=True),)
        read = (klass("results", Regime.BLOCKING),)
        self.assertEqual(
            arms.predicted_value_ns("wait", written, BUDGET, CONSTANTS),
            200 * CONSTANTS.saved_write_ns_per_access)
        self.assertEqual(
            arms.predicted_value_ns("wait", read, BUDGET, CONSTANTS),
            200 * CONSTANTS.saved_ns_per_access)

    def test_a_synchronised_class_carries_the_rank_multiplier(self):
        gate = (klass("gate", Regime.BLOCKING, ranks=32, sync=True),)
        self.assertEqual(
            arms.predicted_value_ns("wait", gate, BUDGET, CONSTANTS),
            200 * 32 * CONSTANTS.saved_ns_per_access)


class Layouts(unittest.TestCase):

    def test_the_promoted_extent_follows_the_measured_limit(self):
        # Writing 128K into a scenario pins the extent to whatever the inline
        # limit happened to be the day it was written.
        wide = Constants(inline_limit_bytes=300_000, saved_ns_per_access=1)
        self.assertIn("-E 320K -L mdt", arms.layout_for(True, wide).spec())
        self.assertIn("-E 128K -L mdt", arms.layout_for(True, CONSTANTS).spec())

    def test_the_default_layout_carries_no_mdt_component(self):
        self.assertNotIn("mdt", arms.layout_for(False, CONSTANTS).spec())


class Baselines(unittest.TestCase):

    def test_every_baseline_decision_is_recorded(self):
        # Reporting only the rule that loses would be picking the opponent.
        decided = arms.baseline_decisions(DISCRIMINATING, BUDGET, CONSTANTS)
        self.assertEqual(set(decided), {"size_threshold", "access_count"})

    def test_a_count_of_application_reads_prefers_the_re_read_class(self):
        decided = arms.baseline_decisions(DISCRIMINATING, BUDGET, CONSTANTS)
        self.assertEqual(decided["access_count"]["promoted"], ["private"])

    def test_a_size_threshold_cannot_split_two_classes_of_one_size(self):
        decided = arms.baseline_decisions(DISCRIMINATING, BUDGET, CONSTANTS)
        self.assertTrue(decided["size_threshold"]["indifferent"])


class Eligibility(unittest.TestCase):

    def test_the_other_outcome_declines_the_band_above_the_inline_limit(self):
        # The extent rounds up to a 64 KiB multiple, so between the limit and the
        # extent sits a band that is on the MDT and still pays a full round trip.
        # The advisor declines it; a baseline arm built with the extent would
        # promote files the rule under test would never have taken.
        band = klass("band", Regime.BLOCKING, size=120 * KIB, count=200)
        blocks = klass("blocks", Regime.BLOCKING, size=4 * KIB, count=200)
        picked = arms.allocation("heuristic", (blocks, band), BUDGET, CONSTANTS)
        self.assertNotIn("band", picked.promoted)


class SizeArm(unittest.TestCase):

    def test_the_size_threshold_runs_as_its_own_arm(self):
        # Scoring it by an indifference argument assumes a coin flip between two
        # classes; the rule actually fills both proportionally, which is a
        # different allocation and a measurable one.
        got = arms.allocation("size", DISCRIMINATING, BUDGET, CONSTANTS)
        self.assertEqual(
            got.promoted,
            policies.size_threshold(DISCRIMINATING, BUDGET, EXTENT,
                                    CONSTANTS.inline_limit_bytes).promoted)

    def test_an_unknown_arm_is_still_refused(self):
        with self.assertRaises(ValueError):
            arms.allocation("whatever", DISCRIMINATING, BUDGET, CONSTANTS)
