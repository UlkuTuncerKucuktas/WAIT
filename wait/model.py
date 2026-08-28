import math
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Dict

from wait.layout import MIN_STRIPE_BYTES, Tier


class Regime(Enum):
    HIDDEN = "hidden"
    BLOCKING = "blocking"
    DEADLINE = "deadline"


class ModelError(ValueError):
    pass


@dataclass(frozen=True)
class Constants:
    inline_limit_bytes: int
    saved_ns_per_access: int
    client_cache_bytes: int = 32 * 1024 ** 3
    base_fstat_ns: int = 0
    per_object_fstat_ns: int = 0
    # Write and fsync move in opposite directions, so the durable pair is the
    # only figure that reports the tier honestly on the write path.
    saved_write_ns_per_access: int = 0
    # A2 and A6 measure the saving at every size on a grid.  The scalars above
    # are the median over that grid, which mixes sizes a scenario never uses --
    # every scenario here writes 4 KiB files, where the read saving is a third
    # smaller than the grid median.  Keyed by size, the scalar is the fallback.
    saved_ns_by_size: dict = dataclasses.field(default_factory=dict)
    saved_write_ns_by_size: dict = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class Allocation:
    promoted: Dict[str, int]
    indifferent: bool = False

    def files(self, name):
        return self.promoted.get(name, 0)


@dataclass(frozen=True)
class FileClass:
    name: str
    size_bytes: int
    accesses: int
    ranks_coupled: int
    synchronized: bool
    regime: Regime
    count: int = 1
    # The read saving and the durable-write saving are different constants and
    # differ by more than a factor of two; charging a written class the read one
    # overstates it.
    writes: bool = False

    def __post_init__(self):
        if self.accesses < 0 or self.size_bytes < 0 or self.count < 0:
            raise ModelError("counts and sizes cannot be negative")
        if self.ranks_coupled < 1:
            raise ModelError("ranks_coupled is at least 1")
        # A private file serves only its owner however large the job.  Charging
        # it by rank count gives it the shared multiplier and the discriminator
        # between the two -- exactly N per promoted byte, by construction --
        # disappears.
        if not self.synchronized and self.ranks_coupled != 1:
            raise ModelError("an unsynchronised class couples exactly one rank")


def saving_for(fc, constants):
    """The saving measured at this class's file size, not a median over sizes."""
    table = (constants.saved_write_ns_by_size if fc.writes
             else constants.saved_ns_by_size)
    scalar = (constants.saved_write_ns_per_access if fc.writes
              else constants.saved_ns_per_access)
    at = {int(k): int(v) for k, v in (table or {}).items()}
    if not at:
        return scalar
    # The largest measured size at or below this one; below the grid, its foot.
    under = [s for s in sorted(at) if s <= fc.size_bytes]
    return at[under[-1]] if under else at[min(at)]


def value_ns(fc, saved_ns_per_access, cold_fraction=1.0):
    if fc.regime is Regime.HIDDEN:
        return 0
    if fc.synchronized:
        # Co-resident ranks share the page cache, so a shared file is fetched cold
        # once per node -- but a barrier makes every rank wait for that fetch.
        # Charging per cold fetch predicts M x delta while the machine loses N x
        # delta, so the multiplier is ranks.  The *accesses* still discount: a
        # gating read served from cache gates nobody, and ignoring cold_fraction
        # here charged a re-read shared file at full price on every round.
        # Rounded once, at the end: truncating the access count first sends a
        # gating class that is read once and does not fit cache to exactly
        # zero, which is the one class the tier exists for.
        return round(fc.accesses * cold_fraction * fc.ranks_coupled
                     * saved_ns_per_access)
    return round(fc.accesses * cold_fraction * saved_ns_per_access)


def cold_accesses(fc, cache_bytes):
    # Re-reads come from the client page cache and are worth nothing -- promoting
    # a re-read file measured indistinguishable from zero.  A class whose whole
    # population fits in cache is fetched cold exactly once per file.
    working_set = fc.count * fc.size_bytes
    if working_set <= cache_bytes:
        return 1.0
    return fc.accesses * (1.0 - cache_bytes / working_set)


def cold_fraction(cache_bytes, working_set_bytes):
    if working_set_bytes <= 0:
        raise ModelError("working set must be positive")
    return max(0.0, 1.0 - cache_bytes / working_set_bytes)


def allocated_objects(layout, size_bytes):
    total = 0
    for i, (start, c) in enumerate(zip(layout.starts(), layout.components)):
        # Components instantiate lazily, so a component the file never reaches has
        # no objects and nothing to glimpse -- unlike a plain layout, where touch
        # on -c 24 already carries all 24.  The first component always exists.
        if i and size_bytes <= start:
            continue
        if c.tier is Tier.OST:
            total += c.stripe_count
    return total


def dom_bytes(size_bytes, dom_extent_bytes):
    # A 25 MB file in a 128 KiB DoM extent puts 128 KiB on the MDT, not 25 MB.
    return min(size_bytes, dom_extent_bytes)


def extent_for(inline_limit_bytes):
    units = math.ceil(inline_limit_bytes / MIN_STRIPE_BYTES)
    return units * MIN_STRIPE_BYTES


def is_promotable(size_bytes, inline_limit_bytes):
    # Against the limit, not the extent it rounds up to.  A file in the band
    # between them burns MDT bytes and still pays a full read round trip, because
    # the inline boundary is a reply-buffer property and does not follow the extent.
    return 0 < size_bytes <= inline_limit_bytes


def predict_fstat_ns(layout, size_bytes, base_ns, per_object_ns):
    return base_ns + per_object_ns * allocated_objects(layout, size_bytes)


def bytes_for(fc, dom_extent_bytes):
    return fc.count * dom_bytes(fc.size_bytes, dom_extent_bytes)


def density_ns_per_byte(fc, saved_ns_per_access, dom_extent_bytes, cold=1.0):
    # Per file on both sides, so the class population cancels: what ranks classes
    # is what one promoted byte buys, not how many bytes the class could consume.
    cost = dom_bytes(fc.size_bytes, dom_extent_bytes)
    if cost <= 0:
        return 0.0
    return value_ns(fc, saved_ns_per_access, cold) / cost
