import argparse
import os
import sys

from wait import campaign


def main(argv=None):
    parser = argparse.ArgumentParser(prog="wait.run")
    parser.add_argument("module", nargs="?", help="e.g. wait.experiments.a2")
    parser.add_argument("phase", nargs="?", choices=("prepare", "measure"))
    parser.add_argument("--base", default=os.environ.get(
        "WAIT_BASE", "/arf/scratch/%s/wait" % os.environ.get("USER", "user")))
    parser.add_argument("--ledger", default=os.environ.get(
        "WAIT_LEDGER", "out/ledgers/campaign.jsonl"))
    parser.add_argument("--reap", action="store_true",
                        help="remove workdirs with no ledger row")
    args = parser.parse_args(argv)

    if args.reap:
        print("reaped %d orphaned workdirs" % campaign.reap(args.base, args.ledger))
        return 0
    if not args.module or not args.phase:
        parser.error("module and phase are required unless --reap")
    campaign.run(args.module, args.phase, args.base, args.ledger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
