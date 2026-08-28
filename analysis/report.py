import csv
import glob
import json
import os
import statistics as st
import sys

LEDGERS = "out/ledgers"
TABLES = "out/tables"
CONSTANTS = "constants.json"
INLINE_SHARE = 0.25


def rows(name):
    path = os.path.join(LEDGERS, name + ".jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def group(records, *fields):
    out = {}
    for r in records:
        out.setdefault(tuple(r["cell_" + f] for f in fields), []).append(r)
    return out


def median(records, field):
    return st.median([r[field] for r in records])


def median_if(records, field):
    """The median where every row carries the field, and None where they do not.

    A ledger written before a field existed is still a valid ledger; a table
    that crashes on it hides every other row it could have reported.
    """
    seen = [r[field] for r in records if field in r]
    return st.median(seen) if len(seen) == len(records) and seen else None


def write_table(name, header, body):
    os.makedirs(TABLES, exist_ok=True)
    with open(os.path.join(TABLES, name + ".csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(body)


def fit_line(points):
    xs = [x for x, _ in points]
    mx, my = st.mean(xs), st.mean([y for _, y in points])
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in points) / denom if denom else 0.0
    return my - slope * mx, slope


def a2_tier_benefit():
    by = group(rows("a2"), "arm", "size_bytes")
    if not by:
        return None
    body, benefits, inlined, by_size = [], [], [], {}
    for size in sorted({s for a, s in by if a != "width"}):
        d2, d3, ost = by[("dom2", size)], by[("dom3", size)], by[("ost", size)]
        total = median(d2, "total_ns_p50")
        share = median(d2, "first_read_ns_p50") / total
        saved = median(ost, "total_ns_p50") - total
        # Data riding the open reply leaves the read a memcpy; past the limit the
        # read costs a round trip and the share steps by an order of magnitude.
        if share < INLINE_SHARE:
            inlined.append(size)
            benefits.append(saved)
            by_size[size] = round(saved)
        body.append([size, round(total), round(median(d3, "total_ns_p50")),
                     round(median(ost, "total_ns_p50")), round(saved), round(share, 4),
                     round(median(d2, "ost_bulk_rpcs") / median(d2, "cell_files"), 2)])
    write_table("a2_tier_benefit",
                ["size_bytes", "dom2_ns", "dom3_ns", "ost_ns", "saved_ns",
                 "read_share", "ost_rpcs_per_file"], body)

    # The width arm varies stripe count at one file size, so it has to be
    # regrouped: keying on size alone collapses all three cells into one.
    wide = group([r for r in rows("a2") if r["cell_arm"] == "width"], "stripe_count")
    width = sorted((median(rs, "granted_objects"), median(rs, "fstat_ns_p50"))
                   for rs in wide.values())
    base, per_object = fit_line(width)
    write_table("a2_width_tax", ["objects", "fstat_ns"],
                [[int(o), round(n)] for o, n in width])
    return {"inline_limit_bytes": max(inlined),
            "saved_ns_per_access": round(st.median(benefits)),
            # Keyed by size as well, because every scenario reads one size and
            # the median over the grid is not that size's saving.
            "saved_ns_by_size": by_size,
            "base_fstat_ns": round(base), "per_object_fstat_ns": round(per_object)}


def a6_write_benefit():
    by = group(rows("a6"), "tier", "size_bytes")
    if not by:
        return None
    body, durable, by_size = [], [], {}
    for size in sorted({s for t, s in by}):
        dom, ost = by[("dom", size)], by[("ost", size)]
        line = [size]
        for stage in ("create_ns", "write_ns", "fsync_ns"):
            d, o = median(dom, stage + "_sum"), median(ost, stage + "_sum")
            line += [round(d), round(o), round(o - d)]
        files = median(dom, "cell_files")
        # Write and fsync move in opposite directions, so the durable pair is the
        # only figure that reports the tier honestly.
        saved = ((median(ost, "write_ns_sum") + median(ost, "fsync_ns_sum")) -
                 (median(dom, "write_ns_sum") + median(dom, "fsync_ns_sum"))) / files
        durable.append(saved)
        by_size[size] = round(saved)
        body.append(line + [round(saved)])
    write_table("a6_write_benefit",
                ["size_bytes", "create_dom_ns", "create_ost_ns", "create_saved_ns",
                 "write_dom_ns", "write_ost_ns", "write_saved_ns", "fsync_dom_ns",
                 "fsync_ost_ns", "fsync_saved_ns", "durable_saved_ns_per_file"], body)
    return round(st.median(durable)), by_size


def a5_model_check(constants):
    from wait.experiments.a5_model_check import LAYOUTS
    from wait.model import allocated_objects, predict_fstat_ns
    by = group(rows("a5"), "layout", "size_bytes")
    body = []
    for (name, size), rs in sorted(by.items()):
        layout = LAYOUTS[name]
        predicted_objects = allocated_objects(layout, size)
        predicted_ns = predict_fstat_ns(layout, size, constants["base_fstat_ns"],
                                        constants["per_object_fstat_ns"])
        measured_ns = median(rs, "fstat_ns_p50")
        body.append([name, size, predicted_objects, int(median(rs, "granted_objects")),
                     round(predicted_ns), round(measured_ns),
                     round(measured_ns / predicted_ns, 3)])
    write_table("a5_model_check",
                ["layout", "size_bytes", "predicted_objects", "granted_objects",
                 "predicted_fstat_ns", "measured_fstat_ns", "ratio"], body)
    return body


def a1_validity():
    by = group(rows("a1"), "arm", "tier")
    write_table("a1_validity",
                ["arm", "tier", "open_ns", "fstat_ns", "read_ns", "total_ns",
                 "ost_bulk_rpcs"],
                [[arm, tier, round(median(rs, "open_ns_p50")),
                  round(median(rs, "fstat_ns_p50")),
                  round(median(rs, "first_read_ns_p50")),
                  round(median(rs, "total_ns_p50")), int(median(rs, "ost_bulk_rpcs"))]
                 for (arm, tier), rs in sorted(by.items())])


def a4_tier_bytes():
    by = group(rows("a4"), "extent_bytes", "size_bytes")
    body = []
    for (extent, size), rs in sorted(by.items()):
        predicted = median(rs, "predicted_kib")
        observed = median(rs, "used_kib")
        drift = st.median([abs(r["drift_kib"]) for r in rs])
        body.append([extent, size, int(median(rs, "cell_files")), int(predicted),
                     int(observed), int(drift),
                     round(observed / predicted, 3) if predicted else 0,
                     "yes" if observed > 0 and drift < 0.5 * predicted else "no"])
    write_table("a4_tier_bytes",
                ["extent_bytes", "size_bytes", "files", "predicted_kib",
                 "observed_kib", "drift_kib", "ratio", "resolvable"], body)
    return body


SCENARIOS = {
    # ledger stem -> (the cell field swept, the metric the arms are compared on)
    "s1": ("scale", "barrier_release_ns"),
    "s2": ("cell_compute_ratio", "phase_ns"),
    "s3": (None, "consume_ns_p50"),
    "s4": ("scale", "generation_release_ns"),
}
ARMS = ("default", "heuristic", "size", "wait")


def _indifferent(records, rule="access_count"):
    """Whether the baseline had any basis to choose, as recorded at run time."""
    flags = [r["baselines"][rule]["indifferent"] for r in records
             if isinstance(r.get("baselines"), dict) and rule in r["baselines"]]
    return bool(flags) and all(flags)


def scenario_rows(name):
    """Every ledger for a scenario, narrowed to one code revision.

    A scenario is run once per rank count and re-run whenever the harness
    changes, so its rows are spread over several ledgers and several revisions.
    Mixing revisions would put an arm measured before a fix beside one measured
    after -- which is how a table says something no single run ever did.  The
    newest revision present wins and the rest are ignored.
    """
    found = []
    for path in sorted(glob.glob(os.path.join(LEDGERS, name + "*.jsonl"))):
        with open(path) as fh:
            found += [json.loads(l) for l in fh if l.strip()]
    found = [r for r in found if not r.get("error")]
    if not found:
        return [], None
    newest = max(found, key=lambda r: r["measured_ns"])["git_rev"]
    return [r for r in found if r["git_rev"] == newest], newest


def at(got, metric):
    """Medians and spreads for one metric over arms already grouped."""
    seen = {a: median(got[a], metric) for a in ARMS}
    spreads = {a: max(g[metric] for g in got[a]) - min(g[metric] for g in got[a])
               for a in ARMS}
    return seen, spreads


def separated(got, metric, first="default", second="wait"):
    """Whether the two arms' repeat ranges fail to overlap at all.

    The spread gate asks whether a difference clears a range, and a range is
    unbounded -- its expectation grows with the repeat count, so more evidence
    makes that gate stricter.  This asks the question the claim needs: could
    these two arms be one distribution?
    """
    low, high = sorted((got[first], got[second]),
                       key=lambda rs: median(rs, metric))
    return max(r[metric] for r in low) < min(r[metric] for r in high)


def arms_at(name, sweep, metric):
    """Every sweep point of a scenario, with each arm's values at that point.

    Three readers wanted the same shape -- the arms table, the prediction table
    and the whole-workload table -- and each had rebuilt the grouping, the
    median and the spread for itself.
    """
    records, rev = scenario_rows(name)
    if not records or metric not in records[0]:
        return [], None
    by = {}
    for r in records:
        by.setdefault((r.get(sweep), r["arm"]), []).append(r)
    out = []
    for point in sorted({k[0] for k in by}, key=lambda v: (v is None, v)):
        got = {a: by.get((point, a)) for a in ARMS}
        if not all(got.values()):
            continue
        # Per arm, so a comparison is gated by the noise of the two arms it
        # involves.  Taking the widest spread over all four gates the
        # default-against-wait claim on an arm that is not in it: S1 at
        # thirty-two ranks has default spread 20.4 ms and wait 50.1, and was
        # called unresolved by the size arm's 155.8.
        seen, spreads = at(got, metric)
        out.append((point, got, seen, spreads))
    return out, rev


def scenario(name, sweep, metric):
    """One table per scenario: the arms at each point of its sweep.

    Read straight out of the ledger so the numbers in the paper and the numbers
    in the rows are the same numbers.
    """
    points, rev = arms_at(name, sweep, metric)
    if not points:
        return None
    body = []
    for point, by_arm, arms_seen, spreads in points:
        pair = lambda a, b: max(spreads[a], spreads[b])
        spread = pair("default", "wait")
        line = [point]
        for arm in ARMS:
            got = by_arm[arm]
            line += [round(arms_seen[arm], 6),
                     round(median(got, "predicted_ns")) if "predicted_ns" in got[0] else ""]
        available = arms_seen["default"] - arms_seen["wait"]
        captured = arms_seen["default"] - arms_seen["heuristic"]
        # Where the rule has no basis to discriminate it flips a coin, so
        # what it captures in expectation is the mean of the two promotable
        # arms -- one of which is WAIT's.  Scoring it by the arm it lost on
        # is scoring the tail, and it makes this paper's margin look larger
        # than its own stated rule allows.  Both are reported.
        indifferent = _indifferent(by_arm["heuristic"])
        # The size threshold runs as its own arm, so what it captures is
        # measured rather than inferred from an indifference argument.  Its
        # recorded flag is still reported: a rule that captured something
        # while having no basis to rank is capturing it by spending the
        # budget on everything, which is a different claim from ranking well.
        size_blind = _indifferent(by_arm["heuristic"], "size_threshold")
        captured_size = arms_seen["default"] - arms_seen["size"]
        expected = ((arms_seen["heuristic"] + arms_seen["wait"]) / 2
                    if indifferent else arms_seen["heuristic"])
        # A share is a ratio whose denominator is the arm difference, so a
        # difference inside the within-arm spread produces a number with no
        # content -- S4 at thirty-two ranks came out at 1.63 that way.  The
        # cells are reported; the ratio is not.
        # Both ends of the ratio, not just the denominator.  A numerator
        # inside the spread prints a capture rate built on noise -- S1's
        # 1 % and 4 % were heuristic-minus-default differences of 0.8 and
        # 1.5 ms against spreads of 20.5 and 31.6.
        resolved = abs(available) > spread
        # The range gate asks whether the arm difference clears the widest
        # within-arm range, and a range is unbounded: its expectation grows
        # with the repeat count, so adding evidence makes the gate stricter.
        # Disjointness asks the question the claim needs -- whether the two
        # arms could be one distribution -- and it is what A7 already uses.
        # S1 at thirty-two ranks separates completely, 103.6-118.1 against
        # 144.4-213.5, while one outlying default repeat widens the range
        # past the difference.  Both are reported; neither is dropped.
        low, high = sorted((by_arm["default"], by_arm["wait"]),
                           key=lambda rs: median(rs, metric))
        separated = (max(r[metric] for r in low)
                     < min(r[metric] for r in high))
        # Whether the heuristic differs from promoting nothing by more than
        # the run-to-run spread -- gated on its own arm and the default, not on
        # the default-and-wait pair.  Gating it on another pair's noise is how
        # S4 at sixteen ranks printed a resolved capture for a heuristic that
        # is 55.4 ms *worse* than the default and whose own range is 96.3.
        counted = abs(captured) > pair("default", "heuristic")
        # share_arm needs both ends of the ratio to clear the spread.
        # share_expected does not: for an indifferent rule it is an
        # expectation over two measured arms, and half of it is structural
        # -- a coin flip between them captures half the available benefit
        # whatever the arm difference turns out to be.
        line += [round(spread, 6), "yes" if resolved else "no",
                 "yes" if separated else "no",
                 "yes" if counted else "no",
                 round(available, 6),
                 round(captured / available, 4)
                 if available and resolved and counted else "",
                 "yes" if indifferent else "no",
                 round((arms_seen["default"] - expected) / available, 4)
                 if available and resolved and indifferent else "",
                 "yes" if size_blind else "no",
                 round(captured_size / available, 4)
                 if available and resolved
                 and abs(captured_size) > pair("default", "size")
                 else ""]
        body.append(line)
    header = [sweep]
    for arm in ARMS:
        header += ["%s_%s" % (arm, metric), "%s_predicted_ns" % arm]
    header += ["spread", "resolved", "separated", "captured_resolved", "available",
               "share_arm", "indifferent", "share_expected",
               "size_indifferent", "share_size"]
    write_table("%s_arms" % name, header, body)
    return len(body), rev, sum(len(g) for _, by_arm, _, _ in points
                                for g in by_arm.values())


# A second metric beside the one the arms were compared on, where the headline
# statistic does not tell the whole story.
#
# S1's arms are compared on the barrier release, the gating read, and an effect
# that resolves there need not resolve on the whole-workload clock.
#
# S4's classes couple one rank each, so the value model prices them per file --
# while generation_release_ns is a sum of maxima over ranks, whose noise grows
# with the rank count although S4's signal does not.  The per-file median is
# taken over every file every rank wrote, so it separates where the wall clock
# cannot, and it is the quantity the model actually predicts.
# metric, label, scale, unit, and whether it is a per-file cost.  Only a
# per-file metric can be compared with a per-file prediction; the whole-workload
# clock is a wall time and dividing it by a file count means nothing.
SECONDARY = {"s1": ("phase_ns_max", "whole-workload clock", 1e6, "ms", False),
             "s4": ("durable_ns", "per-file durable cost", 1e3, "us", True)}


def secondary():
    body = []
    for name, (metric, _label, _scale, _unit, _pf) in sorted(SECONDARY.items()):
        points, _ = arms_at(name, SCENARIOS[name][0], metric)
        for point, got, seen, spreads in points:
            spread = max(spreads["default"], spreads["wait"])
            apart = separated(got, metric)
            gain = seen["default"] - seen["wait"]
            spent = (median(got["wait"], "promoted_files")
                     if "promoted_files" in got["wait"][0] else "")
            body.append([name, point, round(seen["default"]), round(seen["wait"]),
                         round(gain), round(spread),
                         "yes" if abs(gain) > spread else "no",
                         "yes" if apart else "no", spent])
    write_table("secondary", ["scenario", "point", "default_ns", "wait_ns",
                              "gain_ns", "spread_ns", "resolved", "separated",
                              "promoted_files"], body)
    return body


def host_separation():
    """Every cold-read row must have been written on a different node.

    The two srun steps enforce it -- prepare on one node, measure on another --
    but a cell that skips prepare and writes inside measure reads its own page
    cache instead, and the arm then looks like the tier did nothing.  A1's
    `wrong` and `floor` arms do exactly that on purpose, so they are named here
    rather than silently passing.
    """
    declared, offenders = {("a1", "wrong"), ("a1", "floor")}, []
    for path in sorted(glob.glob(os.path.join(LEDGERS, "*.jsonl"))):
        stem = os.path.basename(path)[:-6]
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("prepare_host") or r.get("error"):
                    continue
                if r["prepare_host"] != r.get("measure_host"):
                    continue
                key = (stem.rstrip("0123456789"), r.get("cell_arm") or r.get("arm"))
                if key not in declared:
                    offenders.append((stem, r.get("cell_arm") or r.get("arm")))
    return sorted(set(offenders))


def predictions():
    """The model against the measurement, from the ledgers the tables use.

    `predicted_value_ns` sums the value over promoted files and multiplies a
    synchronised class by the ranks it couples, so it is core time.  The arm
    difference is one rank's wall clock, and every rank pays it, so the two are
    comparable only after multiplying the measurement by the rank count.  A
    ratio whose denominator is inside the spread is left blank rather than
    printed: an unresolved cell has no measurement to predict.
    """
    body = []
    for name, (sweep, metric) in sorted(SCENARIOS.items()):
        points, _ = arms_at(name, sweep, metric)
        for point, got, seen, spreads in points:
            available = seen["default"] - seen["wait"]
            resolved = abs(available) > max(spreads["default"], spreads["wait"])
            ranks = median(got["wait"], "ranks") if "ranks" in got["wait"][0] else 1
            core = available * ranks
            predicted = median(got["wait"], "predicted_ns")
            body.append([name, point, int(ranks), metric, "core-s", 1e9,
                         round(predicted), round(core),
                         "yes" if resolved else "no",
                         round(predicted / core, 2) if resolved and core else ""])
            # Where a scenario's result is reported on a second metric, the
            # prediction is scored there too.  S4's reversal is claimed on the
            # per-file cost, so excusing the model on the wall clock -- where
            # those cells do not resolve -- would score the two on whichever
            # metric flatters each.
            if name in SECONDARY and SECONDARY[name][4]:
                other = SECONDARY[name][0]
                seen2, spreads2 = at(got, other)
                files = median(got["wait"], "promoted_files") or 1
                per_file = predicted / files
                saved = seen2["default"] - seen2["wait"]
                gate = max(spreads2["default"], spreads2["wait"])
                body.append([name, point, int(ranks), other, "us/file", 1e3,
                             round(per_file), round(saved),
                             "yes" if abs(saved) > gate else "no",
                             round(per_file / saved, 2)
                             if abs(saved) > gate and saved else ""])
    write_table("predictions",
                ["scenario", "point", "ranks", "metric", "unit", "scale",
                 "predicted_ns", "measured_ns", "resolved",
                 "predicted_over_measured"], body)
    return body


def s3_band():
    """The deadline result: what fraction of units the consumer had to abandon.

    The workload carries the deadline -- a unit it cannot finish inside the
    window the producer took to make it is dropped -- so this is a behavioural
    outcome rather than a threshold applied to a latency afterwards.  An earlier
    version swept an absolute period and did not reproduce: the machine sat in
    two states across repeats and a threshold near the mean snapped whole cells
    between 0 % and 100 % while the ratio between the arms never moved.
    """
    records, rev = scenario_rows("s3")
    if not records:
        return None
    by = {}
    for rec in records:
        by.setdefault(rec["arm"], []).append(rec)
    body = []
    for arm in ARMS:
        got = by.get(arm)
        if not got:
            continue
        drops = [g["drop_rate"] for g in got]
        budget = median_if(got, "budget_ns_p50")
        body.append([arm, len(got),
                     round(st.median(drops), 4),
                     round(max(drops) - min(drops), 4),
                     round(median(got, "consume_ns_p50")),
                     round(median(got, "consume_ns_p90")),
                     (round(budget) if budget is not None else "")])
    write_table("s3_deadline", ["arm", "repeats", "drop_rate", "drop_spread",
                                "consume_ns_p50", "consume_ns_p90",
                                "budget_ns_p50"], body)
    return body, rev


def a7_concurrency():
    records = rows("a7")
    if not records:
        return None
    by = group(records, "share", "concurrency", "arm")
    body, decay = [], {}
    for share in sorted({s for s, _, _ in by}):
        widths = sorted({c for s, c, _ in by if s == share})
        for conc in widths:
            spans = {}
            for arm in ("dom", "ost"):
                seen = sorted(r["median_ns"] for r in by[(share, conc, arm)])
                spans[arm] = (st.median(seen), seen[0], seen[-1])
            dom, ost = spans["dom"], spans["ost"]
            ratio = ost[0] / dom[0]
            # The arms are separated only when their repeat ranges do not touch;
            # a ratio drawn from overlapping ranges is within the noise.
            resolved = ost[1] > dom[2]
            reads = {a: median(by[(share, conc, a)], "read_ns")
                     for a in ("dom", "ost")}
            body.append([share, conc, round(dom[0]), round(dom[2] - dom[1]),
                         round(ost[0]), round(ost[2] - ost[1]), round(ratio, 2),
                         "yes" if resolved else "no",
                         round(reads["dom"]), round(reads["ost"])])
            decay.setdefault(share, {})[conc] = ratio
    write_table("a7_concurrency",
                ["share", "concurrency", "dom_ns", "dom_spread_ns", "ost_ns",
                 "ost_spread_ns", "ratio", "resolved", "dom_read_ns",
                 "ost_read_ns"], body)
    return body, decay


FIGURES = "out/figures"


def figure_tables():
    """One tidy CSV per figure the paper is likely to draw.

    The tables above are shaped for checking a claim -- one row per cell, every
    column a reader might want.  A plot wants the opposite: long form, one row
    per point, named series, values already in the unit the axis will carry.
    Writing both means a figure never needs a reshaping step nobody recorded.
    """
    os.makedirs(FIGURES, exist_ok=True)
    made = []

    # fig: what the tier saves against file size, and where it stops
    body = []
    for r in _read("a2_tier_benefit"):
        size = int(r["size_bytes"])
        body.append([size, size / 1024.0, float(r["dom2_ns"]) / 1000,
                     float(r["ost_ns"]) / 1000, float(r["saved_ns"]) / 1000,
                     float(r["ost_ns"]) / float(r["dom2_ns"]),
                     float(r["read_share"]), float(r["ost_rpcs_per_file"])])
    made += _emit("tier_benefit_by_size",
                  ["size_bytes", "size_kib", "dom_us", "ost_us", "saved_us",
                   "speedup", "read_share", "ost_rpcs_per_file"], body)

    # fig: the width tax, one point per allocated object count
    made += _emit("width_tax", ["objects", "fstat_us"],
                  [[int(r["objects"]), float(r["fstat_ns"]) / 1000]
                   for r in _read("a2_width_tax")])

    # fig: DoM against OST as readers multiply, both statistics
    body = []
    for r in _read("a7_concurrency"):
        body.append([r["share"], int(r["concurrency"]),
                     float(r["dom_ns"]) / 1000, float(r["ost_ns"]) / 1000,
                     float(r["ratio"]), r["resolved"],
                     float(r["dom_read_ns"]) / 1000,
                     float(r["ost_read_ns"]) / 1000])
    made += _emit("concurrency", ["share", "concurrency", "dom_us", "ost_us",
                                  "ost_over_dom", "resolved", "dom_read_us",
                                  "ost_read_us"], body)

    # fig: every scenario's arms, long form -- the paper's main bar chart
    body = []
    for name, (sweep, metric) in sorted(SCENARIOS.items()):
        for r in _read("%s_arms" % name) if os.path.exists(
                os.path.join(TABLES, "%s_arms.csv" % name)) else []:
            point = r[sweep] if sweep else ""
            for arm in ARMS:
                body.append([name, point, arm, metric,
                             float(r["%s_%s" % (arm, metric)]) / 1e6,
                             float(r["spread"]) / 1e6, r["resolved"],
                             r["separated"]])
    made += _emit("arms", ["scenario", "point", "arm", "metric", "value_ms",
                           "pair_spread_ms", "resolved", "separated"], body)

    # fig: the deadline scenario, one bar per arm
    made += _emit("deadline", ["arm", "repeats", "drop_rate", "drop_spread",
                               "consume_ms", "budget_ms"],
                  [[r["arm"], int(r["repeats"]), float(r["drop_rate"]),
                    float(r["drop_spread"]),
                    float(r["consume_ns_p50"]) / 1e6,
                    float(r["budget_ns_p50"]) / 1e6 if r["budget_ns_p50"] else ""]
                   for r in _read("s3_deadline")])

    # fig: the write path reversing with rank count
    body = []
    for r in _read("secondary"):
        scale = SECONDARY[r["scenario"]][2]
        body.append([r["scenario"], r["point"],
                     float(r["default_ns"]) / scale, float(r["wait_ns"]) / scale,
                     float(r["gain_ns"]) / scale, float(r["spread_ns"]) / scale,
                     SECONDARY[r["scenario"]][3], r["resolved"], r["separated"]])
    made += _emit("secondary_metric",
                  ["scenario", "point", "default", "wait", "gain", "spread",
                   "unit", "resolved", "separated"], body)

    # fig: model against measurement
    made += _emit("prediction",
                  ["scenario", "point", "ranks", "metric", "unit", "predicted",
                   "measured", "resolved", "predicted_over_measured"],
                  [[r["scenario"], r["point"], int(r["ranks"]), r["metric"],
                    r["unit"], float(r["predicted_ns"]) / float(r["scale"]),
                    float(r["measured_ns"]) / float(r["scale"]),
                    r["resolved"], r["predicted_over_measured"]]
                   for r in _read("predictions")])

    return made


def _read(stem):
    path = os.path.join(TABLES, stem + ".csv")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _emit(stem, header, body):
    if not body:
        return []
    path = os.path.join(FIGURES, stem + ".csv")
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(body)
    return [(stem, len(body))]


def main():
    sys.path.insert(0, os.getcwd())
    constants = a2_tier_benefit()
    if not constants:
        print("no A2 ledger; nothing to fit")
        return 1
    constants["client_cache_bytes"] = 32 * 1024 ** 3

    durable = a6_write_benefit()
    if durable:
        constants["saved_write_ns_per_access"] = durable[0]
        constants["saved_write_ns_by_size"] = durable[1]
    a1_validity()
    resolvable = a4_tier_bytes()
    checks = a5_model_check(constants)
    scenarios = {name: scenario(name, *spec) for name, spec in SCENARIOS.items()
                 if spec[0] is not None}
    band = s3_band()
    conc = a7_concurrency()
    preds = predictions()
    same_host = host_separation()
    second = secondary()
    figures = figure_tables()

    with open(CONSTANTS, "w") as fh:
        json.dump(constants, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("constants.json")
    for k, v in sorted(constants.items()):
        print("  %-24s %s" % (k, v))
    print()
    print("A5 model check:")
    for name, size, po, go, pn, mn, ratio in checks:
        print("  %-8s %7d KiB   objects %d/%d %-9s   fstat measured/predicted %.2f"
              % (name, size // 1024, po, go,
                 "match" if po == go else "DIFFER", ratio))
    ok = sum(1 for r in resolvable if r[-1] == "yes")
    print()
    print("A4: %d of %d cells resolvable against drift" % (ok, len(resolvable)))
    print()
    if band:
        rows_, rev = band
        print("  s3  deadline at %s -> tables/s3_deadline.csv" % rev)
        for arm, n, drop, spread, p50, p90, budget in rows_:
            print("      %-10s %d repeats  dropped %5.1f%% (spread %.1f pp)"
                  "  consume %5.1f ms  budget %s"
                  % (arm, n, 100 * drop, 100 * spread, p50 / 1e6,
                     ("%.1f ms" % (budget / 1e6)) if budget != "" else "not recorded"))
    if conc:
        crows, decay = conc
        print()
        print("A7: DoM against OST as readers multiply -> tables/a7_concurrency.csv")
        for share in sorted(decay):
            widths = sorted(decay[share])
            print("      %-8s %.2fx at %d reader -> %.2fx at %d readers"
                  % (share, decay[share][widths[0]], widths[0],
                     decay[share][widths[-1]], widths[-1]))
        held = sum(1 for r in crows if r[7] == "yes")
        print("      DoM ahead in %d of %d cells, all resolved against spread"
              % (held, len(crows)))
    print()
    if same_host:
        print("WRITER AND READER ON ONE NODE -- these rows read their own cache:")
        for stem, arm in same_host:
            print("      %s  arm=%s" % (stem, arm))
    else:
        print("Writer and reader on different nodes in every undeclared cell.")
    if second:
        print()
        print("A second metric beside the headline one -> tables/secondary.csv")
        for name, point, base, w, gain, spread, resolved, separated, spent in second:
            metric, label, scale, unit, _per_file = SECONDARY[name]
            print("      %-3s %-6s %-22s default %7.1f  wait %7.1f %s  "
                  "gain %7.1f (spread %6.1f)  %s"
                  % (name, point, label, base / scale, w / scale, unit,
                     gain / scale, spread / scale,
                     "resolved" if resolved == "yes"
                     else ("separated" if separated == "yes" else "UNRESOLVED")))
    if preds:
        print()
        print("Prediction against measurement -> tables/predictions.csv")
        for row in preds:
            name, point, nranks, metric, unit, scale = row[:6]
            predicted, measured, _resolved, ratio = row[6:]
            print("      %-3s %-6s %2d ranks  %-22s predicted %9.3f  "
                  "measured %9.3f %-8s %s"
                  % (name, point, nranks, metric, predicted / scale,
                     measured / scale, unit,
                     ("%.2fx" % ratio) if ratio else "unresolved"))
    if figures:
        print()
        print("Plot-ready series -> %s/" % FIGURES)
        for stem, n in figures:
            print("      %-22s %d rows" % (stem + ".csv", n))
    for name in sorted(SCENARIOS):
        got = scenarios.get(name)
        if not got:
            continue
        points, rev, kept = got
        print("  %-3s %d sweep points from %d rows at %s -> tables/%s_arms.csv"
              % (name, points, kept, rev, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
