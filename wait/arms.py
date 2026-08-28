import os
import json

from wait import advisor, policies
from wait.layout import MIB
from wait.model import (Constants, cold_accesses, dom_bytes, extent_for,
                        is_promotable, saving_for, value_ns)

CONSTANTS = "constants.json"


def constants(path=CONSTANTS):
    with open(path) as fh:
        return Constants(**json.load(fh))


def _other_outcome(classes, chosen, budget_bytes, extent, limit):
    # An indifferent rule has no basis to prefer one class over another, so it
    # flips a coin and its expectation is the mean of the outcomes.  Both are
    # measured: WAIT's arm is one outcome and this is the other.  Choosing for it
    # -- reporting whichever outcome flatters the comparison -- would be the
    # easiest way to make this paper say something untrue.
    # Against the inline limit, not the extent it rounds up to.  The band
    # between them is on the MDT and still pays a full read round trip, and the
    # advisor declines it -- so admitting it here would build the baseline arm
    # out of files the rule under test would never have promoted.
    eligible = [c for c in classes
                if is_promotable(c.size_bytes, limit) and chosen.files(c.name) == 0]
    promoted, left = {}, budget_bytes
    for c in eligible:
        per_file = dom_bytes(c.size_bytes, extent)
        if per_file <= 0:
            continue
        taken = min(c.count, left // per_file)
        if taken:
            promoted[c.name] = taken
            left -= taken * per_file
    return type(chosen)(promoted, indifferent=True)


def allocation(arm, classes, budget_bytes, consts, heuristic=None):
    """What an arm promotes, derived from the classes rather than named.

    A scenario that names its own answer measures its author.  The advisor and
    the baseline policies are the things under test, so they decide here and the
    scenario only applies what comes back.
    """
    extent = extent_for(consts.inline_limit_bytes)
    if arm == "default":
        return policies.nothing(classes, budget_bytes, extent)
    chosen = advisor.allocate(classes, budget_bytes, consts)
    if arm == "wait":
        return chosen
    if arm == "size":
        # Run as its own arm rather than scored by assumption.  A threshold
        # that cannot rank two classes of one size still produces a concrete
        # allocation -- it fills them proportionally -- so what it captures is
        # measurable and does not have to be inferred from the other arms.
        return policies.size_threshold(classes, budget_bytes, extent,
                                       consts.inline_limit_bytes)
    if arm != "heuristic":
        raise ValueError("unknown arm %r" % arm)
    picked = (heuristic or policies.access_count)(classes, budget_bytes, extent)
    if not picked.indifferent:
        return picked
    return _other_outcome(classes, chosen, budget_bytes, extent,
                          consts.inline_limit_bytes)



def predicted_value_ns(arm, classes, budget_bytes, consts, heuristic=None):
    """What the model says this arm recovers, before it is run.

    Reported beside the measurement whether or not it lands.  S1's prediction was
    an order of magnitude low -- right in sign and ordering, wrong in size -- and
    a prediction that is only reported when it succeeds is not one.
    """
    allocated = allocation(arm, classes, budget_bytes, consts, heuristic)
    total = 0
    for c in classes:
        promoted = allocated.files(c.name)
        if not promoted:
            continue
        cold = cold_accesses(c, consts.client_cache_bytes) / max(1, c.accesses)
        total += promoted * value_ns(c, saving_for(c, consts), cold)
    return total


def promoted_counts(arm, classes, budget_bytes, consts, heuristic=None):
    """How many files of each class the arm promotes, by name.

    An allocation is a count per class, not a single name.  A threshold that
    cannot rank two classes of one size fills both proportionally, so
    collapsing the allocation to one name promotes whichever class sorts first
    and measures some other arm than the one asked for.
    """
    allocated = allocation(arm, classes, budget_bytes, consts, heuristic)
    return {c.name: min(allocated.files(c.name), c.count) for c in classes}


def tier_dir(base, index, promoted, count):
    """Where one file lives under `base`, given how many of them are promoted.

    One directory per layout, because a layout is inherited from the directory
    and naming it per file costs `open` 2.7x.  A set promoted whole or not at
    all keeps its single directory, so an arm that promotes one class is laid
    out exactly as it was before a split was possible.
    """
    if promoted <= 0 or promoted >= count:
        return base
    return os.path.join(base, "dom" if index < promoted else "ost")


def tier_dirs(base, promoted, count):
    """Every directory a set needs, with the layout each one carries."""
    if promoted <= 0 or promoted >= count:
        return [(base, promoted >= count)]
    return [(os.path.join(base, "dom"), True),
            (os.path.join(base, "ost"), False)]


def layout_for(promoted, consts, stripe_count=1, stripe_bytes=MIB):
    """The layout an arm applies, from the advisor rather than a literal.

    The DoM extent follows the measured inline limit; writing 128K into the
    scenario would pin it to whatever the limit happened to be the day it was
    written.
    """
    if promoted:
        return advisor.promoted_layout(consts, stripe_count, stripe_bytes)
    return advisor.default_layout(stripe_count, stripe_bytes)


def baseline_decisions(classes, budget_bytes, consts):
    """What every baseline rule would promote, recorded beside the measurement.

    Two rules, and they do not agree: a size threshold and a profile of
    application reads.  Reporting only the one that loses would be picking the
    opponent.
    """
    extent = extent_for(consts.inline_limit_bytes)
    picked = {
        "size_threshold": policies.size_threshold(
            classes, budget_bytes, extent, consts.inline_limit_bytes),
        "access_count": policies.access_count(classes, budget_bytes, extent),
    }
    return {name: {"promoted": sorted(k for k, v in a.promoted.items() if v),
                   "indifferent": a.indifferent}
            for name, a in picked.items()}
