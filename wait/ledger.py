import hashlib
import json
import os
from dataclasses import asdict, is_dataclass


def cell_fields(cell):
    return asdict(cell) if is_dataclass(cell) else dict(cell)


def key(experiment, cell, repeat, arm=None, scale=None):
    payload = {"experiment": experiment, "repeat": repeat, "arm": arm,
               "scale": scale, "cell": cell_fields(cell)}
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def append(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def rows(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def done(path):
    # Resume is set membership, and a failed cell is a row with an error field --
    # so "never ran" and "ran and failed" stay distinguishable.  Re-running a
    # failure is deliberate: drop its row first.
    return {r["key"] for r in rows(path) if "error" not in r}
