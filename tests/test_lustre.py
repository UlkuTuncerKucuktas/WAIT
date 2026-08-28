import inspect
import os
import pathlib
import tempfile
import unittest

from wait import lustre
from wait.layout import plain

from wait.lustre import (parse_cached_mb, parse_getstripe, parse_md_stats,
                         parse_mdt_used_kib, parse_rpc_stats)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def fixture(name):
    return (FIXTURES / (name + ".txt")).read_text()


class Getstripe(unittest.TestCase):

    def test_a_plain_layout_is_not_read_as_composite(self):
        g = parse_getstripe(fixture("plain_c4"))
        self.assertFalse(g.composite)
        self.assertFalse(g.is_dom)
        self.assertEqual(g.objects_allocated, 4)

    def test_composite_objects_come_from_lmm_objects_not_obdidx_rows(self):
        # A composite layout prints "- N: { l_ost_idx: ... }" rather than the
        # four-column obdidx table, so a parser counting obdidx rows reports
        # zero objects for every PFL file.
        raw = fixture("composite_3c_big")
        self.assertNotIn("obdidx", raw)
        self.assertEqual(parse_getstripe(raw).objects_allocated, 9)

    def test_a_blank_line_between_components_does_not_invent_one(self):
        # Components are separated by a blank line, so the split must not span
        # newlines or an empty block is read as a component.
        self.assertEqual(len(parse_getstripe(fixture("dom_64k")).components), 2)
        self.assertEqual(
            len(parse_getstripe(fixture("composite_3c_small")).components), 3)

    def test_instantiation_matches_the_measured_counts(self):
        small = parse_getstripe(fixture("composite_3c_small"))
        big = parse_getstripe(fixture("composite_3c_big"))
        self.assertEqual(sum(c.instantiated for c in small.components), 1)
        self.assertEqual(sum(c.instantiated for c in big.components), 3)
        self.assertEqual(small.objects_allocated, 0)

    def test_a_dom_component_reports_its_extent_and_no_objects(self):
        g = parse_getstripe(fixture("dom_64k"))
        self.assertTrue(g.is_dom)
        self.assertEqual(g.dom_extent_bytes, 131072)
        self.assertEqual(g.components[0].objects, 0)
        self.assertIsNone(g.components[-1].end_bytes)


class Counters(unittest.TestCase):

    def test_rpc_stats_counts_each_rpc_once(self):
        # Every histogram in rpc_stats -- pages per rpc, rpcs in flight, offset
        # -- has an rpcs column summing to the same total, so a parser that
        # reads them all reports a multiple of the RPC count.  The fixture's
        # pages-per-rpc read column sums to 672,651 and its rpcs-in-flight
        # column to 670,075; only the first is the answer.
        self.assertEqual(parse_rpc_stats(fixture("rpc_stats_head")), 672651)

    def test_rpc_stats_reads_every_osc_not_only_the_first(self):
        # lctl get_param on osc.*.rpc_stats concatenates one file per OSC, and
        # there are forty-eight of them.  A parser that stops after the first
        # block reports one forty-eighth of the traffic.
        block = ("\t\t\tread\t\t\twrite\n"
                 "pages per rpc         rpcs   % cum % |       rpcs   % cum %\n"
                 "1:\t\t         4   0   0   |          3   0   0\n"
                 "\n"
                 "\t\t\tread\t\t\twrite\n"
                 "rpcs in flight        rpcs   % cum % |       rpcs   % cum %\n"
                 "1:\t\t         4   0   0   |          3   0   0\n\n")
        self.assertEqual(parse_rpc_stats(block * 3), 12)

    def test_rpc_stats_stops_at_the_end_of_the_first_histogram(self):
        raw = ("\t\t\tread\t\t\twrite\n"
               "pages per rpc         rpcs   % cum % |       rpcs   % cum %\n"
               "1:\t\t         7   0   0   |          3   0   0\n"
               "2:\t\t         5   0   0   |          1   0   0\n"
               "\n"
               "\t\t\tread\t\t\twrite\n"
               "rpcs in flight        rpcs   % cum % |       rpcs   % cum %\n"
               "1:\t\t        12   0   0   |          4   0   0\n")
        self.assertEqual(parse_rpc_stats(raw), 12)

    def test_md_stats_reads_intent_lock_not_getattr(self):
        # stat() registers as intent_lock: 300 distinct stats moved intent_lock
        # and left getattr at zero.  Folding the two together would attribute
        # other clients' getattr traffic to our stats.
        raw = "getattr                   765635 samples [reqs]\n" \
              "intent_lock              8828653718 samples [reqs]\n"
        self.assertEqual(parse_md_stats(raw, "intent_lock"), 8828653718)
        self.assertEqual(parse_md_stats(raw, "getattr"), 765635)
        self.assertGreater(parse_md_stats(fixture("md_stats_head")), 0)

    def test_a_counter_delta_subtracts_field_by_field(self):
        from wait.lustre import Counters as C
        self.assertEqual(C(10, 20, 30) - C(4, 5, 6), C(6, 15, 24))

    def test_mdt_used_sums_only_mdt_rows(self):
        raw = fixture("lfs_df_mdt")
        self.assertIn("filesystem_summary", raw)
        # Four MDTs of 2.68 TiB; the summary line must not be counted.
        self.assertLess(parse_mdt_used_kib(raw), 11 * 1024 ** 3)

    def test_a_missing_binary_reads_as_absent_rather_than_raising(self):
        # The suite runs on a laptop as well as a client, where lctl is absent.
        from wait.lustre import _run
        result = _run(["lctl-that-does-not-exist", "get_param"])
        self.assertEqual(result.stdout, "")
        self.assertEqual(parse_rpc_stats(result.stdout), 0)

    def test_cached_mb_sums_every_osc(self):
        self.assertEqual(parse_cached_mb("used_mb: 10\nused_mb: 32\n"), 42)


if __name__ == "__main__":
    unittest.main()


class CreateWithLayout(unittest.TestCase):

    def test_the_write_neither_creates_nor_truncates(self):
        # setstripe has already created the file.  Truncating it costs a DoM
        # component its inlining without changing anything getstripe reports or
        # any RPC counter, so the read silently reverts to a full round trip.
        source = inspect.getsource(lustre.create_with_layout)
        self.assertNotIn("O_TRUNC", source)
        self.assertNotIn("O_CREAT", source)
        self.assertNotIn('"wb"', source)
        self.assertIn("os.O_WRONLY", source)

    def test_it_writes_exactly_the_payload(self):
        original = lustre.setstripe
        lustre.setstripe = lambda path, layout: open(path, "wb").close()
        try:
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "f")
                lustre.create_with_layout(path, plain(1, 1 << 20), b"abcd")
                with open(path, "rb") as fh:
                    self.assertEqual(fh.read(), b"abcd")
        finally:
            lustre.setstripe = original
