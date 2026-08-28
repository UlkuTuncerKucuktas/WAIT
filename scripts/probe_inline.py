"""Which act of creating a file costs it its inlining.

A file can be on the MDT -- zero OST objects, no bulk RPC on the read -- and still
pay a full round trip instead of a memcpy, because inlining is a property of the
open reply that no structural check reports.  Five ways of creating the same 4 KiB
file in the same layout, read cold from another node, say which ones keep it.
"""
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wait import lustre, probe
from wait.layout import KIB, MIB, dom, plain

SIZE = 4 * KIB
COUNT = 30
PROMOTED = dom(128 * KIB, 1, MIB)
WAYS = ("inherit", "inherit_reopen", "perfile", "perfile_one_open", "ost")


def build(base, way):
    d = os.path.join(base, way)
    os.makedirs(d, exist_ok=True)
    lustre.setstripe(d, PROMOTED if way != "ost" else plain(1, MIB))
    payload = b"x" * SIZE
    for i in range(COUNT):
        path = os.path.join(d, "f%d" % i)
        if way == "inherit" or way == "ost":
            with open(path, "wb") as fh:          # one open, layout inherited
                fh.write(payload)
        elif way == "inherit_reopen":
            open(path, "wb").close()              # create, close, then reopen
            with open(path, "wb") as fh:
                fh.write(payload)
        elif way == "perfile":
            lustre.setstripe(path, PROMOTED)      # setstripe creates and closes
            with open(path, "wb") as fh:          # ... and this reopens it
                fh.write(payload)
        elif way == "perfile_one_open":
            lustre.setstripe(path, PROMOTED)
            fd = os.open(path, os.O_WRONLY)       # no truncate, single handle
            os.write(fd, payload)
            os.close(fd)


def write_phase(base):
    for way in WAYS:
        build(base, way)
    lustre.flush(base)


def read_phase(base):
    print("%-18s %-6s %8s %8s %8s   %s" %
          ("way", "dom", "open", "fstat", "read", "signature"))
    for way in WAYS:
        d = os.path.join(base, way)
        doms = sum(1 for i in range(COUNT)
                   if lustre.getstripe(os.path.join(d, "f%d" % i)).is_dom)
        rows = [probe.read_staged(os.path.join(d, "f%d" % i)) for i in range(COUNT)]
        o, f, r = (st.median([x[k] for x in rows]) / 1000
                   for k in ("open_ns", "fstat_ns", "first_read_ns"))
        print("%-18s %2d/%-3d %8.1f %8.1f %8.1f   %s" %
              (way, doms, COUNT, o, f, r,
               "inlined" if r < 50 else "not inlined"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("write", "read"))
    ap.add_argument("--base", required=True)
    args = ap.parse_args()
    (write_phase if args.phase == "write" else read_phase)(args.base)
