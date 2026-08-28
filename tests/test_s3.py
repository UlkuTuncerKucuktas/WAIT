import ast
import inspect
import os
import tempfile
import unittest
from unittest import mock

from wait import arms
from wait.experiments import s3_deadline as s3
from wait.model import Constants, Regime, is_promotable

CONSTANTS = Constants(inline_limit_bytes=114688, saved_ns_per_access=651_178,
                      saved_write_ns_per_access=272_698)


def prepared(cell):
    work = tempfile.mkdtemp()
    with mock.patch.object(s3.lustre, "setstripe", lambda p, l: None), \
         mock.patch.object(arms, "constants", lambda: CONSTANTS):
        s3.prepare(cell, work)
    return work


class Classes(unittest.TestCase):

    def test_only_the_index_carries_a_deadline(self):
        index, statistics = s3.file_classes(s3.Cell("wait"))
        self.assertIs(index.regime, Regime.DEADLINE)
        self.assertIsNot(statistics.regime, Regime.DEADLINE)

    def test_the_classes_match_in_size_and_count(self):
        # A size threshold must have no basis, and the counts must be equal so
        # that only the re-reads separate them.
        index, statistics = s3.file_classes(s3.Cell("wait"))
        self.assertEqual(index.size_bytes, statistics.size_bytes)
        self.assertEqual(index.count, statistics.count)

    def test_the_statistics_are_re_read_and_the_index_is_not(self):
        index, statistics = s3.file_classes(s3.Cell("wait"))
        self.assertEqual(index.accesses, 1)
        self.assertGreater(statistics.accesses, index.accesses)

    def test_both_classes_are_small_enough_to_promote(self):
        for c in s3.file_classes(s3.Cell("wait")):
            self.assertTrue(is_promotable(c.size_bytes,
                                          CONSTANTS.inline_limit_bytes))


class Baselines(unittest.TestCase):

    def _decisions(self):
        cell = s3.Cell("wait")
        return arms.baseline_decisions(s3.file_classes(cell),
                                       s3.budget_bytes(cell), CONSTANTS)

    def test_the_advisor_takes_the_class_with_the_deadline(self):
        counts = s3.plan(s3.Cell("wait"), CONSTANTS)["counts"]
        self.assertEqual({n for n, k in counts.items() if k}, {"index"})

    def test_a_count_of_application_reads_takes_the_other_one(self):
        self.assertEqual(self._decisions()["access_count"]["promoted"],
                         ["statistics"])

    def test_both_baselines_fail_here(self):
        # The size threshold has nothing to see -- two classes of one size -- and
        # a count of application reads prefers the re-read statistics eight to
        # one, while the deadline sits on the index.
        decided = self._decisions()
        self.assertTrue(decided["size_threshold"]["indifferent"])
        self.assertEqual(decided["access_count"]["promoted"], ["statistics"])


class Roles(unittest.TestCase):

    def test_producers_and_the_consumer_are_on_different_nodes(self):
        # The writer must not be the reader within the run either, and a unit
        # read on the node that wrote it comes from page cache.
        # Ranged off PRODUCERS, not a literal: a hardcoded four silently stops
        # counting when the producer pool grows.
        localids = range(s3.PRODUCERS + 2)
        producers = [(n, l) for n in (0, 1) for l in localids
                     if s3.role(n, l) == "producer"]
        consumers = [(n, l) for n in (0, 1) for l in localids
                     if s3.role(n, l) == "consumer"]
        self.assertEqual(len(producers), s3.PRODUCERS)
        self.assertEqual(len(consumers), 1)
        self.assertFalse({n for n, _l in producers} & {n for n, _l in consumers})

    def test_a_unit_divides_evenly_across_the_producers(self):
        self.assertEqual(s3.FILES_PER_UNIT % s3.PRODUCERS, 0)
        self.assertEqual(s3.per_producer(s3.Cell("wait")) * s3.PRODUCERS,
                         s3.FILES_PER_UNIT)


class TimedRegion(unittest.TestCase):

    def test_only_the_index_read_is_timed(self):
        # The statistics have no deadline.  Charging their cost to the clock
        # would erase the distinction the scenario exists to measure.
        body = inspect.getsource(s3.measure)
        self.assertIn("_consume(", body)
        self.assertNotIn("_report(cell, workdir, unit)\n            consume", body)
        consume = inspect.getsource(s3._consume)
        self.assertIn('"index"', consume)
        self.assertNotIn('"statistics"', consume)

    def test_a_unit_is_published_only_after_it_is_made_durable(self):
        # An OST write returns buffered, so without the flush the consumer would
        # read from the producer's cache and the deadline would mean nothing.
        emit = inspect.getsource(s3._emit)
        self.assertIn("flush_paths", emit)
        tree = ast.parse(emit.lstrip())
        # By line, not by walk order: ast.walk is breadth-first, so a top-level
        # call comes back before one nested in a loop above it.
        at = {n.func.attr: n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertLess(at["write_paths"], at["flush_paths"])


class Sweep(unittest.TestCase):

    def test_the_period_is_not_a_cell_dimension(self):
        # It was, and the measurement did not reproduce: the consumer's time per
        # unit was bimodal across repeats -- the machine in two states -- and a
        # threshold sitting near the mean snapped whole cells between 0 % and
        # 100 % miss while the ratio between the arms held at 1.9x.  The run
        # records what is stable; the sweep is arithmetic afterwards.
        self.assertFalse(hasattr(s3.Cell("wait"), "period_ms"))
        self.assertEqual(len(s3.cells), 4)

    def test_every_arm_is_measured_once(self):
        self.assertEqual({c.arm for c in s3.cells},
                         {"default", "heuristic", "size", "wait"})

    def test_the_run_records_every_unit(self):
        # The miss rate at any period is arithmetic over these, so nothing in
        # the measurement depends on a period chosen before it ran.
        import inspect
        body = inspect.getsource(s3.measure)
        self.assertIn('"consume_ns": consumer["consume_ns"]', body)
        self.assertNotIn("period_ns", body)


class Directories(unittest.TestCase):

    def test_each_class_has_its_own_directory(self):
        cell = s3.Cell("wait", units=2)
        work = prepared(cell)
        for name in s3.CLASSES:
            self.assertTrue(os.path.isdir(os.path.join(work, name)))
        placing = s3.plan(cell)
        rooms = {os.path.dirname(s3.path_for(work, name, u, p, i, placing))
                 for name in s3.CLASSES for u in range(cell.units)
                 for p in range(s3.PRODUCERS)
                 for i in range(s3.per_producer(cell))}
        self.assertEqual(len(rooms), len(s3.CLASSES))


class Deadline(unittest.TestCase):

    def test_the_workload_acts_on_lateness(self):
        # A class has a deadline when the program does something different
        # because of it, so the consumer abandons a unit it cannot finish.  A
        # deadline carried only by the harness's scoring is not in the program,
        # and an agent reading the source answers "no deadline" -- correctly.
        body = inspect.getsource(s3._consume)
        self.assertIn("budget_ns", body)
        self.assertIn("return staged, True", body)

    def test_the_budget_does_not_come_from_the_consumer(self):
        # Timing it from the consumer's own pace would widen the budget exactly
        # as the consumer slowed, and nothing could ever be late.
        body = inspect.getsource(s3.measure)
        self.assertIn("barrier.broadcast(", body)
        self.assertIn("produce[-1]", body)

    def test_a_dropped_unit_skips_the_rest_of_its_work(self):
        body = inspect.getsource(s3.measure)
        self.assertIn("if not missed:", body)


class Budget(unittest.TestCase):

    def test_the_consumer_records_the_window_it_was_judged_against(self):
        # The slowest producer's time and the broadcast budget are different
        # quantities; reporting the first as the budget makes a consumer that
        # dropped everything look as if it had time to spare.
        body = inspect.getsource(s3.measure)
        self.assertIn("budgets.append(budget)", body)
        cut = body.index("budgets.append(budget)")
        self.assertLess(cut, body.index("_consume(cell, workdir, unit, budget, placing)"))

    def test_the_reported_budget_is_not_the_slowest_producer(self):
        body = inspect.getsource(s3.measure)
        self.assertIn('"budget_ns_p50": st.median(consumer["budget_ns"])', body)
