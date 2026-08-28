from wait.model import Allocation, bytes_for, dom_bytes


def nothing(classes, budget_bytes, dom_extent_bytes):
    return Allocation({})


def size_threshold(classes, budget_bytes, dom_extent_bytes, threshold_bytes):
    eligible = [c for c in classes if c.size_bytes <= threshold_bytes]
    # A threshold decides one file at a time, so under a binding budget its share
    # lands proportionally across everything eligible -- it has no way to prefer
    # one 4 KiB file over another.  Where every eligible class is the same size it
    # cannot discriminate at all, and the arm it picks is a coin flip.
    sizes = {c.size_bytes for c in eligible}
    return Allocation(_fill_proportional(eligible, budget_bytes, dom_extent_bytes),
                      indifferent=len(sizes) <= 1 and len(eligible) > 1)


def access_count(classes, budget_bytes, dom_extent_bytes):
    # Ties break on size, not on the order the scenario happened to list its
    # classes: a stable sort alone makes the decision an artifact of the source
    # the baseline is not supposed to be reading.  Smaller first is the rule a
    # count ranking would use anyway -- it buys more files per budget byte.
    #
    # Eligibility is the extent, not the inline limit.  The limit is a
    # reply-buffer property that has to be measured (A2); a practitioner
    # ranking by count knows dom_stripesize and nothing finer, so charging this
    # baseline for files that fit the tier without benefiting from it is the
    # comparison, not a handicap applied to it.
    ranked = sorted(classes, key=lambda c: (-c.accesses, c.size_bytes, c.name))
    keys = {(c.accesses, c.size_bytes) for c in classes}
    return Allocation(_fill(ranked, budget_bytes, dom_extent_bytes),
                      indifferent=len(keys) <= 1 and len(classes) > 1)


def _fill(ordered, budget_bytes, dom_extent_bytes):
    promoted, left = {}, budget_bytes
    for c in ordered:
        per_file = dom_bytes(c.size_bytes, dom_extent_bytes)
        if per_file <= 0:
            continue
        taken = min(c.count, left // per_file)
        if taken:
            promoted[c.name] = taken
            left -= taken * per_file
    return promoted


def _fill_proportional(classes, budget_bytes, dom_extent_bytes):
    wanted = sum(bytes_for(c, dom_extent_bytes) for c in classes)
    if wanted <= 0:
        return {}
    share = min(1.0, budget_bytes / wanted)
    promoted = {}
    for c in classes:
        taken = int(c.count * share)
        if taken:
            promoted[c.name] = taken
    return promoted
