import csv
import collections
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from wait.experiments import a7_concurrency as a7


def cell(arm="dom", share="distinct", c=8):
    return a7.Cell(arm, share, c)


class Grid(unittest.TestCase):

    def test_the_arms_of_a_comparison_run_back_to_back(self):
        # Whichever arm runs first wins, in both directions, so a comparison
        # must not have a whole sweep between its two halves.
        for i in range(0, len(a7.cells), 2):
            first, second = a7.cells[i], a7.cells[i + 1]
            self.assertEqual((first.share, first.concurrency),
                             (second.share, second.concurrency))
            self.assertNotEqual(first.arm, second.arm)

    def test_both_sharing_modes_at_every_concurrency(self):
        seen = {(c.share, c.concurrency) for c in a7.cells}
        self.assertEqual(seen, {(s, c) for s in ("distinct", "same")
                                for c in a7.CONCURRENCY})

    def test_the_sweep_reaches_a_single_reader(self):
        # Every constant in constants.json is a one-reader number; C=1 is what
        # anchors the curve to them.
        self.assertIn(1, a7.CONCURRENCY)


class Participants(unittest.TestCase):

    def test_participants_are_spread_over_both_nodes(self):
        # Served entirely by one client, a shared read is one cold fetch and the
        # rest page-cache hits, and the arms cannot separate.
        for c in a7.CONCURRENCY:
            if c < 2:
                continue
            counts = a7.per_node(c)
            self.assertEqual(sum(counts), c)
            self.assertTrue(all(n > 0 for n in counts), c)

    def test_the_mapping_is_a_bijection_onto_the_participants(self):
        for c in a7.CONCURRENCY:
            got = [a7.participant(cell(c=c), n, l)
                   for n in range(a7.READER_NODES) for l in range(32)]
            self.assertEqual(sorted(x for x in got if x is not None),
                             list(range(c)), c)

    def test_a_rank_beyond_the_participants_only_barriers(self):
        self.assertIsNone(a7.participant(cell(c=2), 0, 5))


class Sharing(unittest.TestCase):

    def test_distinct_gives_every_participant_its_own_file(self):
        c = cell(share="distinct", c=4)
        paths = {a7.path_for(c, "/w", p, 0) for p in range(4)}
        self.assertEqual(len(paths), 4)

    def test_same_puts_them_all_on_one(self):
        c = cell(share="same", c=4)
        paths = {a7.path_for(c, "/w", p, 0) for p in range(4)}
        self.assertEqual(len(paths), 1)

    def test_a_fresh_file_every_round(self):
        # A re-read is served from page cache and measures memory.
        c = cell(share="same")
        paths = {a7.path_for(c, "/w", 0, r) for r in range(c.rounds)}
        self.assertEqual(len(paths), c.rounds)


class Preparation(unittest.TestCase):

    def test_prepare_writes_exactly_what_measure_reads(self):
        c = a7.Cell("dom", "distinct", 4, rounds=3)
        with tempfile.TemporaryDirectory() as work:
            with mock.patch.object(a7.lustre, "setstripe", lambda p, l: None), \
                 mock.patch.object(a7.arms, "constants", lambda: None), \
                 mock.patch.object(a7.arms, "layout_for", lambda *a, **k: None):
                a7.prepare(c, work)
            wanted = {a7.path_for(c, work, p, r)
                      for p in range(c.concurrency) for r in range(c.rounds)}
            written = {os.path.join(root, f)
                       for root, _d, fs in os.walk(work) for f in fs}
            self.assertEqual(written, wanted)

    def test_the_shared_mode_writes_one_file_per_round(self):
        c = a7.Cell("dom", "same", 4, rounds=3)
        with tempfile.TemporaryDirectory() as work:
            with mock.patch.object(a7.lustre, "setstripe", lambda p, l: None), \
                 mock.patch.object(a7.arms, "constants", lambda: None), \
                 mock.patch.object(a7.arms, "layout_for", lambda *a, **k: None):
                a7.prepare(c, work)
            written = [f for _r, _d, fs in os.walk(work) for f in fs]
            self.assertEqual(len(written), c.rounds)


class SpreadGate(unittest.TestCase):

    def report_on(self, repeats):
        from analysis import report
        home = tempfile.mkdtemp()
        os.makedirs(os.path.join(home, "ledgers"))
        with open(os.path.join(home, "ledgers", "a7.jsonl"), "w") as fh:
            for arm, seen in repeats.items():
                for ns in seen:
                    fh.write(json.dumps({
                        "arm": arm, "cell_arm": arm, "cell_share": "distinct",
                        "cell_concurrency": 8, "median_ns": ns,
                        "read_ns": 10.0}) + "\n")
        ledgers, tables = report.LEDGERS, report.TABLES
        report.LEDGERS = os.path.join(home, "ledgers")
        report.TABLES = os.path.join(home, "tables")
        try:
            return report.a7_concurrency()[0][0]
        finally:
            report.LEDGERS, report.TABLES = ledgers, tables
            shutil.rmtree(home)

    def test_separated_repeat_ranges_resolve(self):
        row = self.report_on({"dom": [100.0, 110.0, 120.0],
                              "ost": [200.0, 210.0, 220.0]})
        self.assertEqual(row[7], "yes")
        self.assertEqual(row[6], 1.91)

    def test_a_ratio_above_one_drawn_from_overlapping_ranges_does_not(self):
        row = self.report_on({"dom": [100.0, 110.0, 300.0],
                              "ost": [120.0, 210.0, 220.0]})
        self.assertGreater(row[6], 1.0)
        self.assertEqual(row[7], "no")

    def test_the_gate_reads_the_spread_not_the_medians(self):
        row = self.report_on({"dom": [110.0, 110.0, 110.0],
                              "ost": [111.0, 111.0, 111.0]})
        self.assertEqual(row[7], "yes")

    def test_ranges_that_merely_touch_do_not_resolve(self):
        row = self.report_on({"dom": [100.0, 105.0, 110.0],
                              "ost": [110.0, 115.0, 120.0]})
        self.assertEqual(row[7], "no")


class ArmOrder(unittest.TestCase):

    def test_each_arm_leads_half_the_comparisons(self):
        # Whichever arm runs first wins, so a sweep that always puts the same
        # one first has the confound aligned with the conclusion.
        leads = collections.Counter(a7.cells[i].arm
                                    for i in range(0, len(a7.cells), 2))
        self.assertEqual(leads["dom"], leads["ost"])

    def test_the_pair_is_still_adjacent(self):
        for i in range(0, len(a7.cells), 2):
            first, second = a7.cells[i], a7.cells[i + 1]
            self.assertNotEqual(first.arm, second.arm)
            self.assertEqual((first.share, first.concurrency),
                             (second.share, second.concurrency))


class ScenarioGate(unittest.TestCase):

    def report_scenario(self, default_ns, wait_ns):
        from analysis import report
        home = tempfile.mkdtemp()
        os.makedirs(os.path.join(home, "ledgers"))
        arms = {"default": default_ns, "wait": wait_ns,
                "heuristic": default_ns, "size": default_ns}
        with open(os.path.join(home, "ledgers", "sx.jsonl"), "w") as fh:
            for arm, seen in arms.items():
                for ns in seen:
                    fh.write(json.dumps({
                        "arm": arm, "scale": 32, "metric": ns, "ranks": 32,
                        "measured_ns": 1, "git_rev": "x", "predicted_ns": 0,
                        "baselines": {}}) + "\n")
        ledgers, tables = report.LEDGERS, report.TABLES
        report.LEDGERS = os.path.join(home, "ledgers")
        report.TABLES = os.path.join(home, "tables")
        try:
            report.scenario("sx", "scale", "metric")
            with open(os.path.join(home, "tables", "sx_arms.csv")) as fh:
                return list(csv.DictReader(fh))[0]
        finally:
            report.LEDGERS, report.TABLES = ledgers, tables
            shutil.rmtree(home)

    def test_one_outlier_repeat_does_not_hide_a_clean_separation(self):
        # S1 at thirty-two ranks: the arms share no value at all, and a single
        # outlying default repeat widens the range past the difference.
        row = self.report_scenario([144.4, 155.7, 159.7, 162.9, 213.5],
                                   [103.6, 106.3, 107.8, 109.3, 118.1])
        self.assertEqual(row["resolved"], "no")
        self.assertEqual(row["separated"], "yes")

    def test_overlapping_arms_are_separated_by_neither_gate(self):
        # S4 at thirty-two ranks: wait's whole range sits inside default's.
        row = self.report_scenario([357.4, 360.0, 363.3, 400.0, 421.8],
                                   [403.1, 405.0, 414.0, 410.0, 409.0])
        self.assertEqual(row["resolved"], "no")
        self.assertEqual(row["separated"], "no")


class FigureTables(unittest.TestCase):

    def test_every_figure_series_is_long_form_and_carries_its_unit(self):
        # A plot wants one row per point with the value already in the axis's
        # unit; the checking tables are one row per cell with everything in ns.
        # Writing both means no figure needs a reshaping step nobody recorded.
        from analysis import report
        for stem, header in (("arms", "value_ms"), ("concurrency", "dom_us"),
                             ("deadline", "consume_ms"),
                             ("secondary_metric", "unit"),
                             ("tier_benefit_by_size", "saved_us")):
            path = os.path.join(report.FIGURES, stem + ".csv")
            self.assertTrue(os.path.exists(path), stem)
            with open(path) as fh:
                rows = list(csv.DictReader(fh))
            self.assertTrue(rows, stem)
            self.assertIn(header, rows[0], stem)

    def test_the_arms_series_names_every_arm_at_every_point(self):
        from analysis import report
        with open(os.path.join(report.FIGURES, "arms.csv")) as fh:
            rows = list(csv.DictReader(fh))
        seen = {}
        for r in rows:
            seen.setdefault((r["scenario"], r["point"]), set()).add(r["arm"])
        for key, arms_at in seen.items():
            self.assertEqual(arms_at, set(report.ARMS), key)
