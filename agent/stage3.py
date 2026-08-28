"""Score what the agent read against what the workload actually is.

Ground truth exists by construction: we wrote the workloads, so each one's
`file_classes` is the answer key -- which is exactly why the classifier never
sees it.

Question 2 is scored as a binary.  The value model only ever asks whether a class
couples one rank or all of them; the job's rank count is a runtime quantity that
appears nowhere in the source, so grading an exact integer would penalise the
agent for not knowing something the source does not contain.
"""
import dataclasses
import json
import os
import statistics as st
from concurrent.futures import ThreadPoolExecutor

from agent import classify
from wait import arms
from wait.model import Regime

SCENARIOS = {
    "s1": ("wait.experiments.s1_barrier", ("manifest", "sidecar", "shard")),
    "s2": ("wait.experiments.s2_hidden", ("tiles", "masks")),
    "s3": ("wait.experiments.s3_deadline", ("index", "statistics")),
    "s4": ("wait.experiments.s4_ensemble", ("results", "diagnostics")),
}
QUESTIONS = ("regime", "couples_all_ranks", "has_deadline", "accesses")
RANKS = 32
MODELS = ("claude-opus-5", "claude-sonnet-5")
RUNS = 5
# The prompt's wording was fixed against S2 and S3: the first live call showed
# that question 4's "read or written" let a set-up write count as an access, and
# S3 showed question 2's examples did not cover a file one rank writes and
# another reads.  Both were definitions rather than answers, but those two are no
# longer held out.  S1 and S4 are.
DEVELOPED_ON = ("s2", "s3")


def source_path(module):
    return os.path.join(*module.split(".")) + ".py"


def truth(name):
    """The four answers, from the scenario's own declaration."""
    module_name, _names = SCENARIOS[name]
    os.environ["WAIT_SCALE"] = str(RANKS)
    module = __import__(module_name, fromlist=["file_classes"])
    cell = module.cells[-1]
    out = {}
    for fc in module.file_classes(cell):
        out[fc.name] = {"regime": fc.regime.value,
                        "couples_all_ranks": fc.ranks_coupled > 1,
                        "has_deadline": fc.regime.value == "deadline",
                        "accesses": fc.accesses}
    return out


def grade(answer, expected):
    """Per-question marks for one run, over the classes the truth declares."""
    got = {c["name"]: c for c in answer["classes"]}
    marks = {q: [] for q in QUESTIONS}
    for name, want in expected.items():
        said = got.get(name)
        for q in QUESTIONS:
            if said is None:
                marks[q].append(False)
            elif q == "couples_all_ranks":
                marks[q].append(said["couples_all_ranks"] == want[q])
            else:
                marks[q].append(said[q] == want[q])
    return marks


def accuracy(marks):
    return {q: (sum(v) / len(v) if v else 0.0) for q, v in marks.items()}


def agreement(answers, expected):
    """How often the runs said the same thing, right or wrong."""
    out = {}
    for q in QUESTIONS:
        per_class = []
        for name in expected:
            said = []
            for a in answers:
                got = {c["name"]: c for c in a["classes"]}.get(name)
                said.append(None if got is None else got[q])
            per_class.append(said.count(max(set(said), key=said.count)) / len(said))
        out[q] = st.mean(per_class)
    return out


def label_path(directory, name, model, condition, run):
    return os.path.join(directory, "%s_%s_%s_%d.json"
                        % (name, model.replace(".", ""), condition, run))


def _jobs(directory):
    for name in sorted(SCENARIOS):
        for model in MODELS:
            for anonymise in (False, True):
                condition = "anon" if anonymise else "named"
                for i in range(RUNS):
                    yield (name, model, anonymise, i,
                           label_path(directory, name, model, condition, i))


def _one(job, client):
    """Fetch one label, or read the committed one back."""
    name, model, anonymise, i, path = job
    if os.path.exists(path):
        with open(path) as fh:
            stored = json.load(fh)
        # An answer is committed and never re-bought.  A failure is not an
        # answer: keeping it would make a transient -- an exhausted credit
        # balance took fourteen calls -- permanent, and every later run would
        # skip exactly the calls that need retrying.
        if not stored.get("error"):
            return stored
    module_name, names = SCENARIOS[name]
    # A failure is a record, not a gap, and not the end of the other
    # seventy-nine: one call returning nothing must not cost the whole study.
    try:
        got = classify.label(source_path(module_name), names, model=model,
                             client=client, anonymise=anonymise)
    except Exception as exc:
        got = {"model": model, "classes": [],
               "error": "%s: %s" % (type(exc).__name__, exc)}
    got.update({"scenario": name, "condition": "anon" if anonymise else "named",
                "run": i})
    with open(path, "w") as fh:
        json.dump(got, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return got


def fetch(directory="labels", client=None, workers=10):
    """Every label, in one pass.

    The calls are independent and each takes the better part of a minute with
    adaptive thinking on, so they go out together.
    """
    os.makedirs(directory, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda j: _one(j, client), list(_jobs(directory))))


def marks_over(answers, truths):
    marks = {q: [] for q in QUESTIONS}
    for answer in answers:
        for q, got in grade(answer, truths[answer["scenario"]]).items():
            marks[q] += got
    return marks


def per_cell(answers, truths):
    cells = {}
    for answer in answers:
        key = (answer["scenario"], answer["model"], answer["condition"])
        cells.setdefault(key, []).append(answer)
    return {key: {"accuracy": accuracy(marks_over(got, truths)),
                  "agreement": agreement(got, truths[key[0]]),
                  "failures": sum(1 for a in got if a.get("error"))}
            for key, got in cells.items()}


def headlines(answers, truths):
    """The rows the paper reports, computed here rather than by hand.

    A non-answer counts as wrong in every figure except the one that separates
    it out -- a study that drops its failures reports the accuracy of the calls
    that happened to work.
    """
    opus, sonnet = MODELS
    rows = [
        ("%s, all" % opus, [a for a in answers if a["model"] == opus]),
        ("%s, held out" % opus,
         [a for a in answers if a["model"] == opus
          and a["scenario"] not in DEVELOPED_ON]),
        ("%s, named" % opus,
         [a for a in answers if a["model"] == opus and a["condition"] == "named"]),
        ("%s, anonymised" % opus,
         [a for a in answers if a["model"] == opus and a["condition"] == "anon"]),
        ("%s, all" % sonnet, [a for a in answers if a["model"] == sonnet]),
        ("%s, answered only" % sonnet,
         [a for a in answers if a["model"] == sonnet and not a.get("error")]),
    ]
    return [(label, len(got), sum(1 for a in got if a.get("error")),
             accuracy(marks_over(got, truths))) for label, got in rows if got]


def majority_baseline(truths):
    """What always giving the commonest answer would score.

    There are nine class-items in total, and the repeats multiply them rather
    than add to them.  Several truths are lopsided -- eight of nine classes
    couple one rank, eight of nine carry no deadline -- so an accuracy near
    ninety per cent on those two questions is close to what answering "no" every
    time would achieve, and the number means little without this beside it.
    """
    items = [v for t in truths.values() for v in t.values()]
    out = {}
    for q in QUESTIONS:
        counts = {}
        for item in items:
            counts[item[q]] = counts.get(item[q], 0) + 1
        out[q] = max(counts.values()) / len(items)
    return out, len(items)



def as_classes(name, answer):
    """The scenario's classes with the agent's three answers substituted.

    Size, count and whether a class is written are structural -- the agent is
    never asked for them and could not see them in the redacted source.  Regime,
    coupling and access count are what it is asked for, so those are what the
    label replaces.
    """
    module_name, _names = SCENARIOS[name]
    os.environ["WAIT_SCALE"] = str(RANKS)
    module = __import__(module_name, fromlist=["file_classes"])
    cell = module.cells[-1]
    said = {c["name"]: c for c in answer["classes"]}
    out = []
    for fc in module.file_classes(cell):
        got = said.get(fc.name)
        if got is None:
            return None, (module, cell)
        couples = bool(got["couples_all_ranks"])
        out.append(dataclasses.replace(
            fc,
            regime=Regime(got["regime"]),
            ranks_coupled=RANKS if couples else 1,
            synchronized=couples,
            accesses=max(1, abs(int(got["accesses"])))))
    return tuple(out), (module, cell)


def allocation_agreement(directory="labels"):
    """Whether the agent's labels allocate what the declarations allocate.

    Stage 2 measures the allocation given correct labels, and Stage 3 measures
    whether the labels are correct.  Neither shows the two composing, and the
    composition is the claim.  This drives every committed label through
    `advisor.allocate` and compares the class it promotes with the class the
    scenario's own declarations promote.
    """
    consts = arms.constants()
    rows, agree = [], 0
    for name, model, anonymise, run, path in _jobs(directory):
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            answer = json.load(fh)
        if "error" in answer:
            continue
        classes, where = as_classes(name, answer)
        module, cell = where
        budget = module.budget_bytes(cell)
        want = _promoted("wait", module.file_classes(cell), budget, consts)
        got = ([] if classes is None
               else _promoted("wait", classes, budget, consts))
        same = got == want
        agree += bool(same)
        rows.append([name, model.replace("claude-", ""),
                     "anon" if anonymise else "named", run,
                     " ".join(got) or "-", " ".join(want) or "-",
                     "yes" if same else "NO"])
    return rows, agree


def _promoted(arm, classes, budget_bytes, consts):
    counts = arms.promoted_counts(arm, classes, budget_bytes, consts)
    return sorted(k for k, v in counts.items() if v)


def report(answers, truths):
    cells = per_cell(answers, truths)
    print("%-4s %-10s %-6s | %s | %5s %s"
          % ("", "model", "names", "  ".join("%9s" % q[:9] for q in QUESTIONS),
             "agree", "fail"))
    for key in sorted(cells):
        name, model, condition = key
        cell = cells[key]
        held = "" if name in DEVELOPED_ON else "  (held out)"
        print("%-4s %-10s %-6s | %s | %4.0f%% %4d%s"
              % (name, model.replace("claude-", ""), condition,
                 "  ".join("%8.0f%%" % (100 * cell["accuracy"][q])
                           for q in QUESTIONS),
                 100 * sum(cell["agreement"].values()) / len(QUESTIONS),
                 cell["failures"], held))
    print()
    print("%-28s %5s %5s | %s" % ("", "calls", "fail",
                                  "  ".join("%9s" % q[:9] for q in QUESTIONS)))
    for label, n, failures, acc in headlines(answers, truths):
        print("%-28s %5d %5d | %s"
              % (label.replace("claude-", ""), n, failures,
                 "  ".join("%8.0f%%" % (100 * acc[q]) for q in QUESTIONS)))
    base, items = majority_baseline(truths)
    print("%-28s %5d %5s | %s"
          % ("always the commonest answer", items, "-",
             "  ".join("%8.0f%%" % (100 * base[q]) for q in QUESTIONS)))
    print()
    rows, agree = allocation_agreement()
    if rows:
        print()
        print("%-28s %5d %5s | %s"
              % ("labels -> advisor -> layout", len(rows), "-",
                 "%d of %d reproduce the declared allocation"
                 % (agree, len(rows))))
    print()
    print("%d distinct class-items; the repeats multiply them, not add to them."
          % items)
    print("has_deadline is defined as regime == deadline, so it is a consistency")
    print("check on regime rather than a fourth independent question -- the")
    print("advisor consumes three labels, not four.")


if __name__ == "__main__":
    got = fetch()
    report(got, {name: truth(name) for name in SCENARIOS})
