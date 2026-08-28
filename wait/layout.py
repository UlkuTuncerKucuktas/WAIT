from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

KIB = 1024
MIB = 1024 * KIB
MIN_STRIPE_BYTES = 64 * KIB


class Tier(Enum):
    MDT = "mdt"
    OST = "ost"


class LayoutError(ValueError):
    pass


@dataclass(frozen=True)
class Component:
    end_bytes: Optional[int]
    tier: Tier
    stripe_count: Optional[int] = None
    stripe_bytes: Optional[int] = None
    pool: Optional[str] = None


@dataclass(frozen=True)
class Layout:
    components: Tuple[Component, ...]

    def __post_init__(self):
        if not self.components:
            raise LayoutError("a layout needs at least one component")

        for i, c in enumerate(self.components):
            last = i == len(self.components) - 1
            if c.end_bytes is None and not last:
                raise LayoutError("only the final component may run to EOF")
            if c.end_bytes is not None and last:
                raise LayoutError("the final component must run to EOF")

        for a, b in zip(self.starts(), self.components):
            if b.end_bytes is not None and b.end_bytes <= a:
                raise LayoutError("component ends must increase")

        for i, c in enumerate(self.components):
            if c.tier is Tier.MDT:
                self._check_mdt(i, c)
            else:
                self._check_ost(c)

    def _check_mdt(self, index, c):
        if index != 0:
            raise LayoutError("the MDT component must come first")
        # lfs rejects the whole layout: "Option 'stripe-size' can't be specified
        # with Data-on-MDT component".  The extent end IS its stripe size.
        if c.stripe_bytes is not None or c.stripe_count is not None:
            raise LayoutError("an MDT component takes neither -S nor -c")
        if c.end_bytes is None or c.end_bytes % MIN_STRIPE_BYTES:
            raise LayoutError("an MDT extent must be a multiple of %d bytes"
                              % MIN_STRIPE_BYTES)

    def _check_ost(self, c):
        if not c.stripe_count or c.stripe_count < 1:
            raise LayoutError("an OST component needs a stripe count")
        # Omit -S on a component that follows a DoM one and it silently inherits
        # the DoM extent as its stripe size: -E 128K -L mdt -E -1 -c 4 granted
        # stripe_size 131072 on the 4-wide component.
        if not c.stripe_bytes:
            raise LayoutError("an OST component needs an explicit stripe size")
        if c.stripe_bytes % MIN_STRIPE_BYTES:
            raise LayoutError("stripe size must be a multiple of %d bytes"
                              % MIN_STRIPE_BYTES)
        if c.end_bytes is not None and c.end_bytes % c.stripe_bytes:
            raise LayoutError("a component end must be a multiple of its stripe size")

    def starts(self):
        offsets, running = [], 0
        for c in self.components:
            offsets.append(running)
            running = c.end_bytes if c.end_bytes is not None else running
        return tuple(offsets)

    def has_dom(self):
        return self.components[0].tier is Tier.MDT

    def dom_extent_bytes(self):
        return self.components[0].end_bytes if self.has_dom() else 0

    def spec(self):
        if len(self.components) == 1 and not self.has_dom():
            c = self.components[0]
            return _flags(c)
        parts = []
        for c in self.components:
            end = "-1" if c.end_bytes is None else format_size(c.end_bytes)
            parts.append("-E " + end)
            parts.append("-L mdt" if c.tier is Tier.MDT else _flags(c))
        return " ".join(parts)


def _flags(c):
    out = "-c %d -S %s" % (c.stripe_count, format_size(c.stripe_bytes))
    return out + (" -p %s" % c.pool if c.pool else "")


def format_size(n):
    for suffix, unit in (("M", MIB), ("K", KIB)):
        if n % unit == 0:
            return "%d%s" % (n // unit, suffix)
    return str(n)


def dom(extent_bytes, stripe_count, stripe_bytes, pool=None):
    return Layout((
        Component(extent_bytes, Tier.MDT),
        Component(None, Tier.OST, stripe_count, stripe_bytes, pool),
    ))


def plain(stripe_count, stripe_bytes, pool=None):
    return Layout((Component(None, Tier.OST, stripe_count, stripe_bytes, pool),))
