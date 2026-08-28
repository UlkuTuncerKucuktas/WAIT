import json
import os
import shutil
import tempfile
import unittest
from dataclasses import dataclass

from wait import campaign, ledger


@dataclass(frozen=True)
class Cell:
    size_bytes: int
    arm: str = "dom"
    same_node: bool = False


class Keys(unittest.TestCase):

    def test_the_same_cell_and_repeat_give_the_same_key(self):
        a = ledger.key("a2", Cell(4096), 0, "dom")
        self.assertEqual(a, ledger.key("a2", Cell(4096), 0, "dom"))

    def test_arm_repeat_and_cell_all_change_the_key(self):
        base = ledger.key("a2", Cell(4096), 0, "dom")
        self.assertNotEqual(base, ledger.key("a2", Cell(4096), 0, "ost"))
        self.assertNotEqual(base, ledger.key("a2", Cell(4096), 1, "dom"))
        self.assertNotEqual(base, ledger.key("a2", Cell(8192), 0, "dom"))


class Resume(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "l.jsonl")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_resume_skips_completed_cells(self):
        ledger.append(self.path, {"key": "abc", "total_ns": 1})
        self.assertEqual(ledger.done(self.path), {"abc"})

    def test_a_failed_cell_writes_a_row_and_is_not_counted_done(self):
        # Missing and failed must stay distinguishable, or a partially completed
        # grid silently lies about what it covered.
        ledger.append(self.path, {"key": "bad", "error": "LustreError: boom"})
        self.assertEqual(ledger.done(self.path), set())
        self.assertEqual(len(ledger.rows(self.path)), 1)


class Percentiles(unittest.TestCase):

    def test_percentile_returns_none_below_sample_size(self):
        # int(N*p) is one rank too high and saturates: p99 was literally the
        # maximum for every sample of a hundred or fewer.
        from wait.probe import nearest_rank
        self.assertIsNone(nearest_rank(list(range(50)), 0.99))
        self.assertIsNone(nearest_rank([], 0.5))
        self.assertIsNotNone(nearest_rank(list(range(200)), 0.99))
        self.assertEqual(nearest_rank([1, 2, 3, 4], 0.5), 2)


class Cleanup(unittest.TestCase):

    def setUp(self):
        self.base = tempfile.mkdtemp()
        self.path = os.path.join(self.base, "l.jsonl")

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def _run_one(self, measure):
        work = campaign.workdir(self.base, "x", "k1")
        campaign.write_provenance(work, {"host": "elsewhere"})
        record = json.load(open(os.path.join(work, campaign.PROVENANCE)))
        record["host"] = "elsewhere"
        json.dump(record, open(os.path.join(work, campaign.PROVENANCE), "w"))

        class Stub:
            @staticmethod
            def measure(cell, w):
                return measure()
        campaign._measure_one(Stub, "x", Cell(4096), 0, "k1", work, "env", self.path)
        return work

    def test_a_successful_cell_removes_its_workdir(self):
        work = self._run_one(lambda: {"total_ns": 1})
        self.assertFalse(os.path.exists(work))

    def test_a_failed_cell_keeps_its_workdir_as_evidence(self):
        # A failed cell's directory is the only evidence of why it failed.
        def boom():
            raise RuntimeError("boom")
        work = self._run_one(boom)
        self.assertTrue(os.path.exists(work))
        self.assertIn("error", ledger.rows(self.path)[0])


class WriterAndReader(unittest.TestCase):

    def setUp(self):
        self.work = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def test_measuring_on_the_writer_node_is_refused(self):
        campaign.write_provenance(self.work)
        with self.assertRaises(campaign.PhaseError):
            campaign.check_reader_is_not_the_writer(self.work, False)

    def test_a1_may_declare_same_node_deliberately(self):
        # Its wrong arm exists to show that measuring this way yields a null.
        campaign.write_provenance(self.work)
        writer, reader = campaign.check_reader_is_not_the_writer(self.work, True)
        self.assertEqual(writer, reader)

    def test_measuring_without_a_prepare_phase_is_refused(self):
        with self.assertRaises(campaign.PhaseError):
            campaign.check_reader_is_not_the_writer(self.work, False)

    def test_a_different_writer_passes(self):
        campaign.write_provenance(self.work)
        path = os.path.join(self.work, campaign.PROVENANCE)
        record = json.load(open(path))
        record["host"] = "some-other-node"
        json.dump(record, open(path, "w"))
        writer, reader = campaign.check_reader_is_not_the_writer(self.work, False)
        self.assertNotEqual(writer, reader)


if __name__ == "__main__":
    unittest.main()
