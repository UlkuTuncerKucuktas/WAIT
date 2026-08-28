import json
import os
import tempfile
import unittest
from unittest import mock

from agent import classify, stage3

REGIME_WORDS = ("DEADLINE", "BLOCKING", "HIDDEN", "deadline", "blocking",
                "hidden", "synchronized", "ranks_coupled")


class Redaction(unittest.TestCase):

    def _sources(self):
        for name, (module, names) in sorted(stage3.SCENARIOS.items()):
            with open(stage3.source_path(module)) as fh:
                yield name, fh.read(), names

    def test_no_scenario_hands_over_its_own_answers(self):
        # Every scenario declares file_classes -- regime, rank count, access
        # count -- and several docstrings state the labels in prose.  Sending the
        # file whole would be sending the answer key.
        for name, source, _names in self._sources():
            body = classify.redacted(source)
            found = [w for w in REGIME_WORDS if w in body]
            self.assertFalse(found, "%s leaks %s" % (name, found))

    def test_redaction_keeps_the_control_flow(self):
        # It has to remove the answers without removing the evidence.
        for name, source, _names in self._sources():
            body = classify.redacted(source)
            self.assertIn("def measure(", body, name)
            self.assertIn("def prepare(", body, name)
            self.assertGreater(len(body.splitlines()), 40, name)

    def test_comments_and_docstrings_go(self):
        source = 'def f():\n    """Doc."""\n    x = 1  # note\n    return x\n'
        body = classify.redacted(source)
        self.assertNotIn("Doc.", body)
        self.assertNotIn("note", body)
        self.assertIn("x = 1", body)

    def test_a_hash_inside_a_string_is_not_a_comment(self):
        body = classify.redacted('p = "a#b"\n')
        self.assertIn('"a#b"', body)


class Anonymisation(unittest.TestCase):

    def test_no_class_name_survives_anywhere(self):
        # Not word-bounded: the name also sits inside identifiers and row keys,
        # and one surviving occurrence is the whole answer.  Nor case-bound and
        # nor plural-bound -- a class called "results" is spelled RESULT_BYTES
        # in the constant and `result` in the payload, and either one standing
        # beside class_b names the class as plainly as the class name did.
        for name, (module, names) in sorted(stage3.SCENARIOS.items()):
            with open(stage3.source_path(module)) as fh:
                body, mapping = classify.anonymised(fh.read(), names)
            body = classify.redacted(body).lower()
            stems = {stem for n in names for stem in classify._stems(n)}
            self.assertFalse([stem for stem in stems if stem in body], name)
            self.assertEqual(len(set(mapping.values())), len(names), name)

    def test_the_singular_and_the_shouted_form_both_go(self):
        body, _ = classify.anonymised(
            'RESULT_BYTES = 4\nresult = b"x"\nResults = [result]\n',
            ["results"])
        self.assertNotIn("result", body.lower())
        self.assertIn("CLASS_A_BYTES", body)
        self.assertIn("Class_a", body)


class Truth(unittest.TestCase):

    def test_every_regime_is_represented(self):
        seen = {v["regime"] for name in stage3.SCENARIOS
                for v in stage3.truth(name).values()}
        self.assertEqual(seen, {"blocking", "hidden", "deadline"})

    def test_only_s3_carries_a_deadline(self):
        # The one positive example of the question that carries the sharpest
        # claim in the paper, which is why S3 had to exist before Stage 3.
        with_deadline = {name for name in stage3.SCENARIOS
                         if any(v["has_deadline"] for v in stage3.truth(name).values())}
        self.assertEqual(with_deadline, {"s3"})


class Grading(unittest.TestCase):

    def _answer(self, **over):
        base = {"name": "index", "regime": "deadline", "couples_all_ranks": False,
                "has_deadline": True, "accesses": 1, "reason": ""}
        base.update(over)
        return {"classes": [base]}

    def _expected(self):
        return {"index": {"regime": "deadline", "couples_all_ranks": False,
                          "has_deadline": True, "accesses": 1}}

    def test_a_correct_answer_scores_one_everywhere(self):
        marks = stage3.grade(self._answer(), self._expected())
        self.assertEqual(stage3.accuracy(marks),
                         {q: 1.0 for q in stage3.QUESTIONS})

    def test_rank_coupling_is_asked_the_way_it_is_graded(self):
        # It was asked as an integer and graded as one-or-all, so the model
        # answered -1 for "all ranks" -- a fair reading of a question the source
        # cannot answer numerically -- and the grader scored it wrong.  Ask the
        # binary the model actually consumes.
        self.assertIn("couples_all_ranks", classify.SCHEMA["properties"]["classes"]
                      ["items"]["properties"])
        self.assertNotIn("ranks_coupled", classify.SCHEMA["properties"]["classes"]
                         ["items"]["properties"])
        marks = stage3.grade(self._answer(couples_all_ranks=True), self._expected())
        self.assertEqual(stage3.accuracy(marks)["couples_all_ranks"], 0.0)
        marks = stage3.grade(self._answer(couples_all_ranks=False), self._expected())
        self.assertEqual(stage3.accuracy(marks)["couples_all_ranks"], 1.0)

    def test_a_class_the_agent_never_mentions_scores_zero(self):
        marks = stage3.grade({"classes": []}, self._expected())
        self.assertEqual(stage3.accuracy(marks),
                         {q: 0.0 for q in stage3.QUESTIONS})

    def test_agreement_counts_the_majority_answer(self):
        runs = [self._answer(), self._answer(), self._answer(regime="blocking")]
        got = stage3.agreement(runs, self._expected())
        self.assertAlmostEqual(got["regime"], 2 / 3)
        self.assertAlmostEqual(got["accesses"], 1.0)


class Failures(unittest.TestCase):

    class _Response:
        def __init__(self, blocks, stop_reason="max_tokens"):
            self.content = blocks
            self.stop_reason = stop_reason
            self.stop_details = None
            self.usage = type("U", (), {"input_tokens": 1, "output_tokens": 2})()

    class _Client:
        def __init__(self, response):
            self.messages = type("M", (), {"create": lambda _s, **_k: response})()

    def test_a_response_with_no_text_says_why(self):
        # Adaptive thinking spends from the same budget as the answer, so a call
        # can return a thinking block and nothing else.  A bare StopIteration
        # names neither the cause nor the call.
        client = self._Client(self._Response([]))
        with self.assertRaises(classify.ClassifyError) as caught:
            classify.classify("x = 1\n", ["a"], client=client)
        self.assertIn("max_tokens", str(caught.exception))

    def test_a_refusal_is_not_parsed_as_an_answer(self):
        client = self._Client(self._Response([], stop_reason="refusal"))
        with self.assertRaises(classify.ClassifyError):
            classify.classify("x = 1\n", ["a"], client=client)

    def test_a_failed_call_scores_zero_rather_than_vanishing(self):
        # It has to be a record: a study that silently drops its failures
        # reports the accuracy of the calls that happened to work.
        expected = {"index": {"regime": "deadline", "couples_all_ranks": False,
                              "has_deadline": True, "accesses": 1}}
        marks = stage3.grade({"classes": [], "error": "boom"}, expected)
        self.assertEqual(stage3.accuracy(marks),
                         {q: 0.0 for q in stage3.QUESTIONS})


class Baseline(unittest.TestCase):

    def test_the_trivial_baseline_is_reported(self):
        # Eight of nine classes carry no deadline, so answering "no" every time
        # already scores 89 % on that question.  An accuracy figure means little
        # without this beside it, and the gap is the only part that is evidence.
        truths = {name: stage3.truth(name) for name in stage3.SCENARIOS}
        base, items = stage3.majority_baseline(truths)
        self.assertEqual(items, 9)
        self.assertAlmostEqual(base["has_deadline"], 8 / 9)
        # Six of nine couple one rank, so this question is nearly balanced and
        # the trivial score on it is low.
        self.assertAlmostEqual(base["couples_all_ranks"], 6 / 9)
        self.assertLess(base["regime"], 0.7)

    def test_the_deadline_question_is_not_independent_of_the_regime(self):
        # truth() derives one from the other, so it cannot disagree.
        for name in stage3.SCENARIOS:
            for item in stage3.truth(name).values():
                self.assertEqual(item["has_deadline"],
                                 item["regime"] == "deadline")


class Schema(unittest.TestCase):

    def test_every_required_field_exists(self):
        # A `required` naming a property that does not exist, together with
        # `additionalProperties: false`, makes the schema unsatisfiable -- and
        # the model degrades instead of erroring: one class instead of two,
        # "x1" for a reason, 285 output tokens instead of 700.  A whole study
        # of that scores as an accuracy drop rather than as a fault.
        item = classify.SCHEMA["properties"]["classes"]["items"]
        self.assertEqual(set(item["required"]) - set(item["properties"]), set())
        self.assertFalse(item["additionalProperties"])

    def test_the_questions_asked_are_the_questions_graded(self):
        asked = set(classify.SCHEMA["properties"]["classes"]["items"]["properties"])
        self.assertEqual(set(stage3.QUESTIONS) - asked, set())


class Retry(unittest.TestCase):

    def test_a_stored_answer_is_never_re_bought(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json")
            with open(path, "w") as fh:
                json.dump({"classes": [], "model": "m"}, fh)
            got = stage3._one(("s2", "m", False, 0, path), client=None)
            self.assertEqual(got["model"], "m")

    def test_a_stored_failure_is_retried(self):
        # Fourteen calls failed on an exhausted credit balance.  Treating that
        # record as an answer would make a transient permanent and skip exactly
        # the calls that need retrying.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json")
            with open(path, "w") as fh:
                json.dump({"classes": [], "error": "boom"}, fh)
            with mock.patch.object(stage3.classify, "label",
                                   lambda *a, **k: {"classes": [], "model": "fresh"}):
                got = stage3._one(("s2", "m", False, 0, path), client=None)
            self.assertEqual(got["model"], "fresh")


class Helpers(unittest.TestCase):

    def test_the_helper_bodies_are_sent(self):
        # Without them the only thing separating S4's two classes in the source
        # is the names `write_staged` and `write_buffered` -- the answer spelled
        # out rather than derived.  With them the model sees an fsync in one and
        # not the other, which is control flow.
        helpers = classify.helper_source()
        staged = helpers[helpers.index("def write_staged"):helpers.index("def write_buffered")]
        buffered = helpers[helpers.index("def write_buffered"):]
        self.assertIn("fsync", staged)
        self.assertNotIn("fsync", buffered.split("def ")[1])

    def test_the_prompt_carries_them(self):
        self.assertIn("{helpers}", classify.PROMPT)


class PromptHygiene(unittest.TestCase):

    def test_no_question_quotes_a_scenario_verbatim(self):
        # Q2's example was "a file every rank reads before a barrier", which is
        # S1's manifest described exactly -- while S1 is claimed held out.
        self.assertNotIn("every rank reads before a barrier", classify.SYSTEM)

    def test_the_access_question_forbids_multiplying_by_ranks(self):
        # The model was answering per-class totals with the mechanism right and
        # scoring zero, because the ground truth counts one rank's accesses and
        # the rank multiplier is question 2.
        self.assertIn("Do not multiply by the number of ranks", classify.SYSTEM)


class Provenance(unittest.TestCase):

    def test_an_answer_records_what_it_was_bought_against(self):
        # The scenario sources were renamed after some runs, so without this
        # nothing proves which source text an answer was bought against.
        class _R:
            content = [type("B", (), {"type": "text", "text": '{"classes": []}'})()]
            stop_reason = "end_turn"
            stop_details = None
            usage = type("U", (), {"input_tokens": 1, "output_tokens": 1})()
        client = type("C", (), {"messages": type("M", (), {
            "create": lambda _s, **_k: _R()})()})()
        got = classify.classify("x = 1\n", ["a"], client=client)
        for field in ("git_rev", "prompt_sha", "source_sha"):
            self.assertIn(field, got)
        self.assertTrue(got["source_sha"])


class EndToEnd(unittest.TestCase):

    def test_the_agent_s_labels_reach_an_allocation(self):
        # Stage 2 measures the allocation given correct labels and Stage 3
        # measures whether the labels are correct.  Neither shows the two
        # composing, and the composition is what the paper claims.
        rows, agree = stage3.allocation_agreement()
        self.assertEqual(len(rows), len(stage3.SCENARIOS)
                         * len(stage3.MODELS) * stage3.RUNS * 2)
        self.assertEqual(agree, len(rows))

    def test_only_the_three_asked_fields_come_from_the_label(self):
        # Size, count and writes are structural; the agent never sees them and
        # substituting them from a label would be measuring the scenario.
        answer = {"classes": [{"name": "results", "regime": "hidden",
                               "couples_all_ranks": True, "accesses": 99},
                              {"name": "diagnostics", "regime": "hidden",
                               "couples_all_ranks": True, "accesses": 99}]}
        classes, _ = stage3.as_classes("s4", answer)
        from wait.experiments import s4_ensemble as s4
        declared = {c.name: c for c in s4.file_classes(s4.cells[-1])}
        for got in classes:
            self.assertEqual(got.size_bytes, declared[got.name].size_bytes)
            self.assertEqual(got.count, declared[got.name].count)
            self.assertEqual(got.writes, declared[got.name].writes)
            self.assertEqual(got.accesses, 99)
