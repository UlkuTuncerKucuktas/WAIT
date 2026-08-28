from wait.layout import Component, Layout, Tier
from wait.model import (Allocation, Regime, cold_accesses, density_ns_per_byte,
                        dom_bytes, extent_for, is_promotable, saving_for)


def _density(fc, constants, extent):
    # Cold accesses are derived from what the agent reports plus a site constant.
    # The advisor never sees a measured saving; that is what it is scored against.
    cold = cold_accesses(fc, constants.client_cache_bytes) / max(1, fc.accesses)
    return density_ns_per_byte(fc, saving_for(fc, constants), extent, cold)


def allocate(classes, budget_bytes, constants):
    extent = extent_for(constants.inline_limit_bytes)
    # Anything above the inline limit still costs MDT bytes and still pays a full
    # read round trip, because the boundary is a reply-buffer property that does
    # not follow the extent the layout rounds up to.
    eligible = [c for c in classes
                if is_promotable(c.size_bytes, constants.inline_limit_bytes)]

    deadline = [c for c in eligible if c.regime is Regime.DEADLINE]
    blocking = [c for c in eligible if c.regime is Regime.BLOCKING]
    # Hidden classes are declined, never demoted below the site default: a wrong
    # hidden label then costs foregone benefit rather than added stall.

    deadline.sort(key=lambda c: dom_bytes(c.size_bytes, extent))
    blocking.sort(key=lambda c: _density(c, constants, extent), reverse=True)

    promoted, left = {}, budget_bytes
    for c in deadline + blocking:
        per_file = dom_bytes(c.size_bytes, extent)
        if per_file <= 0:
            continue
        taken = min(c.count, left // per_file)
        if taken:
            promoted[c.name] = taken
            left -= taken * per_file
    return Allocation(promoted)


def promoted_layout(constants, stripe_count, stripe_bytes, pool=None):
    return Layout((
        Component(extent_for(constants.inline_limit_bytes), Tier.MDT),
        Component(None, Tier.OST, stripe_count, stripe_bytes, pool),
    ))


def default_layout(stripe_count, stripe_bytes, pool=None):
    return Layout((Component(None, Tier.OST, stripe_count, stripe_bytes, pool),))
