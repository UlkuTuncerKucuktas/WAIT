"""Read a workload's source and answer the four questions the advisor needs.

The source is redacted first, and the redaction is the experiment.  A scenario
declares its own `file_classes` -- regime, rank count, access count, all of it --
so sending the file whole would be handing over the answer key.  Docstrings and
comments go too: they were written for a reader who already knows the design and
several of them state the labels outright.  What is left is control flow, which
is what the paper claims is sufficient.
"""
import ast
import hashlib
import inspect
import io
import json
import re
import os
import subprocess
import tokenize

import anthropic


class ClassifyError(RuntimeError):
    """The model returned no usable answer, and which way it failed."""

MODEL = "claude-opus-5"
# Adaptive thinking spends from the same budget as the answer, and one call in
# eighty returned a thinking block and no text -- the ceiling reached before it
# wrote anything.  The answer itself is a few hundred tokens.
MAX_TOKENS = 16000
# What the scenario says about itself.  Everything here either names a label or
# derives from one.
ANSWER_KEY = ("file_classes", "promoted_class", "budget_bytes", "promoted_paths",
              "promoted_ranks", "budget_files")

# The workload calls these and their bodies are the difference between a durable
# write and a buffered one.  Without them the only thing separating S4's classes
# in the source is the names `write_staged` and `write_buffered`, which is the
# answer spelled out rather than derived.
HELPERS = ("write_staged", "write_buffered", "read_staged", "write_paths")

SCHEMA = {
    "type": "object",
    "properties": {
        "classes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "regime": {"type": "string",
                               "enum": ["blocking", "hidden", "deadline"]},
                    "couples_all_ranks": {"type": "boolean"},
                    "has_deadline": {"type": "boolean"},
                    "accesses": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "regime", "couples_all_ranks",
                             "has_deadline", "accesses", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["classes"],
    "additionalProperties": False,
}

SYSTEM = """You are reading the source of one HPC workload and classifying the \
file classes it uses, so that a storage advisor can decide which of them belongs \
on a scarce fast tier.

Answer four questions per class, from the control flow alone.

1. regime -- which of these the class is:
   * blocking: a rank stops and waits for this access before it can continue.
   * hidden: the access is overlapped with computation, so no rank waits on it.
   * deadline: the access must complete within a time window or the work is lost.
2. couples_all_ranks -- true if every rank in the job is blocked waiting while \
one access to one file completes, false if only the rank making the access is. \
Decide it from what the code makes the other ranks do while the access happens: \
if they are held at a synchronisation point until it completes, it blocks them \
all; if they are doing their own work, it blocks only the rank making it. Where \
one rank writes a file and a different rank later reads it, each access blocks \
only the rank making it -- the writer is not waiting on the reader's access, nor \
the reader on the writer's.
3. has_deadline -- whether there is a time window the access must fit inside.
4. accesses -- how many times a single rank accesses a single file of that \
class: the reads, for a class the workload reads; the writes, for a class it \
writes. Count repeats by the same rank. **Do not multiply by the number of ranks** \
-- how many ranks are involved is question 2, and counting them here would count \
them twice. Do not count files being created during set-up, and do not add a \
write and a read together for the same file.

Give a one-sentence reason per class citing the specific code that decides it."""

PROMPT = """Workload source (comments, docstrings and the class declarations have \
been removed):

```python
{source}
```

Helpers the workload calls, same treatment:

```python
{helpers}
```

Classify these file classes: {names}."""


def _comment_free(source):
    blanked = source.splitlines()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for kind, _text, (row, col), _end, _line in tokens:
            if kind == tokenize.COMMENT:
                blanked[row - 1] = blanked[row - 1][:col]
    except (tokenize.TokenError, IndentationError):
        pass
    return blanked


def redacted(source, hide=ANSWER_KEY):
    """The workload without anything that states its own labels."""
    tree = ast.parse(source)
    cut = set()
    for node in ast.walk(tree):
        # Imports go: they carry no control flow the call sites do not already
        # show, and a module whose name collides with a class name -- `import
        # statistics as st` beside a class called statistics -- reverses the
        # anonymisation for free.
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            cut.update(range(node.lineno, node.end_lineno + 1))
        if isinstance(node, ast.FunctionDef) and node.name in hide:
            cut.update(range(node.lineno, node.end_lineno + 1))
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and body:
            first = body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                cut.update(range(first.lineno, first.end_lineno + 1))
    lines = _comment_free(source)
    kept = [line.rstrip() for i, line in enumerate(lines, 1)
            if i not in cut and line.strip()]
    return "\n".join(kept)


def anonymised(source, names):
    """Replace the class names with neutral tokens wherever they are quoted.

    A class called `results` beside one called `diagnostics` tells a reader
    something before any control flow is read, and the paper's claim is about
    control flow.  Running both ways measures how much of the answer the names
    were carrying.
    """
    mapping = {name: "class_%s" % chr(ord("a") + i)
               for i, name in enumerate(sorted(names))}
    # Plain substring, not a word boundary: the name also turns up inside
    # identifiers and row keys -- "results_are_dom" keeps it through any
    # boundary-anchored pattern, and one surviving occurrence is the whole
    # answer.  Identifiers come out odd and consistently renamed, which is the
    # right trade.
    #
    # Case-insensitively and over the singular stem, because the declaration is
    # plural and lower case while the source that carries the same word is
    # neither: a class called "results" leaves RESULT_BYTES and a payload named
    # `result` standing beside class_b, which names the class as plainly as the
    # class name did.
    stems = {}
    for real, token in mapping.items():
        for stem in _stems(real):
            stems[stem] = token
    pattern = re.compile("|".join(re.escape(stem) for stem in
                                  sorted(stems, key=len, reverse=True)),
                         re.IGNORECASE)

    def swap(match):
        return _cased(stems[match.group(0).lower()], match.group(0))

    return pattern.sub(swap, source), mapping


def _stems(name):
    """The forms of a class name a source file actually spells it in."""
    forms = {name.lower()}
    if name.lower().endswith("s"):
        forms.add(name.lower()[:-1])
    return forms


def _cased(token, sample):
    if sample.isupper():
        return token.upper()
    if sample[:1].isupper():
        return token.capitalize()
    return token


def helper_source():
    """The bodies of the helpers the workloads call, redacted the same way."""
    from wait import probe
    return redacted("\n\n".join(inspect.getsource(getattr(probe, name))
                                for name in HELPERS), hide=())


def _revision():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def classify(source, names, model=MODEL, client=None, anonymise=False):
    client = client or anthropic.Anthropic()
    body, mapping = anonymised(source, names) if anonymise else (source, None)
    asked = sorted(mapping.values()) if mapping else list(names)
    sent = PROMPT.format(source=redacted(body), helpers=helper_source(),
                         names=", ".join(asked))
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": sent}],
    )
    if response.stop_reason == "refusal":
        raise ClassifyError("refused: %s" % response.stop_details)
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise ClassifyError(
            "no text block, stop_reason=%s, output_tokens=%d"
            % (response.stop_reason, response.usage.output_tokens))
    got = json.loads(text)["classes"]
    if mapping:
        back = {token: real for real, token in mapping.items()}
        for c in got:
            c["name"] = back.get(c["name"], c["name"])
    # What the answer was bought against, so a later reader can tell whether the
    # source has moved under it.
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()[:12]
    return {"model": model, "anonymised": bool(mapping),
            "git_rev": _revision(),
            "prompt_sha": digest(SYSTEM), "source_sha": digest(sent),
            "usage": {"input": response.usage.input_tokens,
                      "output": response.usage.output_tokens},
            "classes": got}


def label(path, names, model=MODEL, client=None, anonymise=False):
    with open(path) as fh:
        return classify(fh.read(), names, model=model, client=client,
                        anonymise=anonymise)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--names", required=True,
                    help="comma-separated file class names")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out")
    ap.add_argument("--anonymise", action="store_true")
    args = ap.parse_args()
    got = label(args.source, [n.strip() for n in args.names.split(",")],
                model=args.model, anonymise=args.anonymise)
    text = json.dumps(got, indent=2, sort_keys=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
