import os
import time

NS = time.perf_counter_ns
MIB = 1 << 20


def now_ns():
    return time.time_ns()


def monotonic_s():
    return time.monotonic()


def sleep_seconds(seconds):
    time.sleep(seconds)


def read_staged(path, chunk_bytes=MIB):
    # The stages are not interchangeable.  A wide layout's per-object cost lands
    # in the size query, not in open, and below the inline limit DoM's cost lands
    # in open with a memcpy for the read.  Summing the stages hides both.
    t0 = NS()
    fd = os.open(path, os.O_RDONLY)
    t1 = NS()
    os.fstat(fd)
    t2 = NS()
    block = os.pread(fd, chunk_bytes, 0)
    t3 = NS()
    read = len(block)
    while block:
        block = os.pread(fd, chunk_bytes, read)
        read += len(block)
    t4 = NS()
    os.close(fd)
    return {"open_ns": t1 - t0, "fstat_ns": t2 - t1, "first_read_ns": t3 - t2,
            "rest_read_ns": t4 - t3, "total_ns": t4 - t0, "bytes": read}


def write_staged(path, payload):
    # Write and fsync are read together, never apart: an OST write returns
    # cheaply because it is buffered and defers its work, so write alone reports
    # DoM as the slower tier while write+fsync reports it as the faster one.
    t0 = NS()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    t1 = NS()
    os.write(fd, payload)
    t2 = NS()
    os.fsync(fd)
    t3 = NS()
    os.close(fd)
    return {"create_ns": t1 - t0, "write_ns": t2 - t1, "fsync_ns": t3 - t2,
            "durable_ns": t3 - t1, "total_ns": t3 - t0}


def write_buffered(path, payload):
    # No fsync.  A class nobody waits on defers its cost to the filesystem's own
    # writeback, which is exactly what makes it worth nothing to promote.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def write_paths(paths, size_bytes):
    # Named paths rather than a count and a prefix, for a caller whose reader
    # names files itself: a prefix that drifts writes a set nothing opens, and
    # the miss surfaces at measure time with the allocation already spent.
    payload = b"x" * size_bytes
    for path in paths:
        with open(path, "wb") as fh:
            fh.write(payload)


def write_files(directory, count, size_bytes, prefix="f"):
    write_paths([os.path.join(directory, "%s%d" % (prefix, i))
                 for i in range(count)], size_bytes)


def nearest_rank(values, quantile):
    # int(N*p) is one rank too high and saturates: p99 was literally the maximum
    # for every sample of 100 or fewer.  Refuse a quantile the sample cannot support.
    if not values:
        return None
    if quantile > 0 and len(values) < int(round(1 / (1 - quantile))):
        return None
    ordered = sorted(values)
    index = int(round(quantile * len(ordered) + 0.5)) - 1
    return ordered[max(0, min(len(ordered) - 1, index))]
