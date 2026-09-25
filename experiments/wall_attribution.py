"""Which constraints put more than 1% of N on one record in a VB draw?

For each fitted VB family (in practice ``sumdiff``, one per PUMA and alpha),
draws ``lambda`` and finds every (block group, record) cell whose share of
``p(lambda)`` -- its share of ``N`` -- exceeds ``--threshold``: an *event*.
Each event's logit shift relative to the MAP is split over the constraints
the cell loads on, exactly as part 4 of ``laplace_diagnostic.py`` does:
with ``delta = lambda - lambda*``, constraint row ``k`` contributes
``-x_ik delta_k`` for a unit ``i`` in that row's area.

To ask whether the constraints behind the events share a signature -- block
group cells statistically indistinguishable from zero, carried by few or
bulk-loaded PUMS records -- the same features are computed for **every**
constraint category, implicated or not, so that the two groups can be
compared:

- block group side: cells, the share with ``|Y| < 2 SE`` (SE floored at the
  zero-count variance, as the fits use it), the share published as zero, the
  summed published count; the same for the tract side where the category is
  constrained there;
- PUMS side: carriers (units with a nonzero loading), the largest loading,
  the largest carrier's and top five carriers' shares of the weighted PUMS
  total, the Herfindahl index of the weighted loadings, and the PUMS total over
  the published total.

Writes to ``--out``:

- ``events.csv``: one row per event -- PUMA, alpha, draw, block group, record,
  share of N, household size, group quarters, the cell's total logit shift;
- ``event_constraints.csv``: each event's top contributing constraint rows --
  level, area, category, loading, published, floored SE, MAP multiplier,
  ``delta``, contribution and share of the positive shift;
- ``constraint_features.csv``: one row per (PUMA, category) with the features
  above and, per alpha, how many events it led (top contributor) and was in
  the top five of, and the largest weight any of its carriers takes -- in any
  one block group -- at the MAP (``map_max_w``) and in any VB draw
  (``vb_max_w``), with the largest design weight among its carriers
  (``design_max_w``) for scale. Plotted against ``carriers`` this asks
  whether few-carrier categories are the ones whose carriers reach large
  weights. A record carries many categories, so a common category's
  carriers can reach large weights through a rare one they also carry;
  ``events_led`` says which category drove an event, and ``vb_max_w_led``
  is the largest weight among the events a category led;
- ``summary.txt``: per (PUMA, alpha) event counts and the leading categories,
  and implicated against not-implicated categories' features.

Several fits in one call, each ``--fit PUMA ALPHA VB_RUN_DIR``::

    $CONDA_PREFIX/bin/python experiments/wall_attribution.py --variance-floor zero \\
        --fit 4701502 1.0 $RUNS/6277113 --fit 4701502 0.1 $RUNS/6277117 ... \\
        --out $RUNS/wall_attribution
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logsumexp

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.progress import logger, record_run
from pmedm_vb.solvers.base import ConstraintOperator

from laplace_diagnostic import cell_loadings, load_vb

#: Standard errors within which a published cell counts as zero-equivalent.
ZERO_SE = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fit", nargs=3, action="append", required=True,
                        metavar=("PUMA", "ALPHA", "VB_RUN"),
                        help="one fitted VB family; repeat for each (PUMA, alpha)")
    parser.add_argument("--taper", default="tract")
    parser.add_argument("--area", default="knox-2024-5yr")
    parser.add_argument("--variance-floor", default="zero",
                        help="'zero', 'none' or a number; sets the SE used for zero-equivalence")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=500, help="draws formed at once")
    parser.add_argument("--threshold", type=float, default=0.01, help="share of N")
    parser.add_argument("--top", type=int, default=5, help="constraints kept per event")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def floor_spec(text: str):
    return None if text == "none" else "zero" if text == "zero" else float(text)


# -- features -----------------------------------------------------------------


def level_features(Y: np.ndarray, se: np.ndarray, prefix: str) -> dict:
    """Published-side features of one category at one level (``Y``, ``se``: per area)."""
    return {
        f"{prefix}_areas": Y.size,
        f"{prefix}_zero_equiv_share": float(np.mean(np.abs(Y) < ZERO_SE * se)),
        f"{prefix}_zero_share": float(np.mean(Y == 0)),
        f"{prefix}_published_total": float(Y.sum()),
    }


def pums_features(column: np.ndarray, weight: np.ndarray, published_total: float) -> dict:
    """PUMS-side features of one category's loadings."""
    carriers = column > 0
    weighted = weight[carriers] * column[carriers]
    total = weighted.sum()
    shares = np.sort(weighted / total)[::-1] if total > 0 else np.zeros(0)
    return {
        "carriers": int(carriers.sum()),
        "max_loading": float(column.max()) if column.size else 0.0,
        "top1_share": float(shares[0]) if shares.size else np.nan,
        "top5_share": float(shares[:5].sum()) if shares.size else np.nan,
        "hhi": float(np.square(shares).sum()) if shares.size else np.nan,
        "pums_total": float(total),
        "pums_over_published": float(total / published_total) if published_total > 0 else np.nan,
    }


def category_carriers(inputs: PMEDMInputs) -> dict[str, np.ndarray]:
    """Per category, the units carrying it (block group column, else tract)."""
    X_T, X_B = sp.csc_matrix(inputs.X_T), sp.csc_matrix(inputs.X_B)
    out = {}
    for name in dict.fromkeys([*inputs.bg_constraints, *inputs.tract_constraints]):
        if name in inputs.bg_constraints:
            column = X_B[:, inputs.bg_constraints.index(name)]
        else:
            column = X_T[:, inputs.tract_constraints.index(name)]
        out[name] = column.indices[column.data > 0]
    return out


def category_features(inputs: PMEDMInputs, floor) -> pd.DataFrame:
    """One row per constraint category (block group or tract-only) of one PUMA."""
    split = inputs.Y_T.size
    se_all = np.sqrt(inputs.floored_variances(floor))
    se_T = se_all[:split].reshape(inputs.Y_T.shape, order="F")
    se_B = se_all[split:].reshape(inputs.Y_B.shape, order="F")
    X_T, X_B = sp.csc_matrix(inputs.X_T), sp.csc_matrix(inputs.X_B)
    weight = inputs.units["weight"].to_numpy(float)
    tract_index = {name: k for k, name in enumerate(inputs.tract_constraints)}
    rows = []
    for name in dict.fromkeys([*inputs.bg_constraints, *inputs.tract_constraints]):
        row = {"puma": inputs.puma, "category": name}
        if name in inputs.bg_constraints:
            k = inputs.bg_constraints.index(name)
            row.update(level_features(inputs.Y_B[:, k], se_B[:, k], "bg"))
            column, published = X_B[:, k].toarray().ravel(), row["bg_published_total"]
        else:
            column = None
        if name in tract_index:
            k = tract_index[name]
            row.update(level_features(inputs.Y_T[:, k], se_T[:, k], "tract"))
            if column is None:
                column, published = X_T[:, k].toarray().ravel(), row["tract_published_total"]
        row["levels"] = "+".join(level for level, present in
                                 (("bg", name in inputs.bg_constraints), ("tract", name in tract_index))
                                 if present)
        row.update(pums_features(column, weight, published))
        rows.append(row)
    return pd.DataFrame(rows)


# -- events -------------------------------------------------------------------


def row_labels(inputs: PMEDMInputs) -> pd.DataFrame:
    """Per stacked constraint row: level, area and category."""
    n_t, n_b = inputs.Y_T.shape[0], inputs.Y_B.shape[0]
    return pd.DataFrame({
        "level": ["tract"] * inputs.Y_T.size + ["bg"] * inputs.Y_B.size,
        "geoid": list(inputs.tracts.iloc[:, 0].astype(str)) * len(inputs.tract_constraints)
                 + list(inputs.block_groups.iloc[:, 0].astype(str)) * len(inputs.bg_constraints),
        "category": [c for c in inputs.tract_constraints for _ in range(n_t)]
                    + [c for c in inputs.bg_constraints for _ in range(n_b)],
    })


def find_events(args, puma: str, alpha: float, vb_run: Path, rng) -> tuple[list[dict], list[dict], int]:
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / puma)
    name = f"{puma}_{args.taper}_a{alpha:g}"
    q, _, _ = load_vb(vb_run / f"{name}.npz")
    with np.load(vb_run / f"{name}.npz") as saved:
        map_lam = saved["map_lam"]
        family = str(saved["family"]) if "family" in saved.files else "?"
    op = ConstraintOperator(inputs)
    with np.errstate(divide="ignore"):
        log_q = np.log(inputs.q)
    labels = row_labels(inputs)
    targets = inputs.targets()
    se = np.sqrt(inputs.floored_variances(floor_spec(args.variance_floor)))
    X_B = sp.csc_matrix(inputs.X_B)
    age_sex = [k for k, c in enumerate(inputs.bg_constraints) if c.startswith("B01001.")]
    household_size = np.asarray(X_B[:, age_sex].sum(axis=1)).ravel()
    zones = inputs.zones.iloc[:, 0].astype(str).to_numpy()
    serials = inputs.units["SERIALNO"].to_numpy()
    gq = inputs.units["is_group_quarters"].to_numpy(bool)
    log_threshold = np.log(args.threshold)
    logits = log_q - op.adjoint(map_lam)
    map_w = inputs.N * np.exp(logits - logsumexp(logits))
    unit_max_map = map_w.max(axis=0)
    unit_max_vb = np.zeros(inputs.n_units)

    events, contributions = [], []
    started = time.perf_counter()
    for start in range(0, args.draws, args.batch):
        lam = q.sample(rng, min(args.batch, args.draws - start))
        for j in range(lam.shape[1]):
            draw = start + j
            logits = log_q - op.adjoint(lam[:, j])
            log_p = logits - logsumexp(logits)
            np.maximum(unit_max_vb, inputs.N * np.exp(log_p.max(axis=0)), out=unit_max_vb)
            for zone, unit in zip(*np.nonzero(log_p > log_threshold)):
                delta = lam[:, j] - map_lam
                index, loading = cell_loadings(inputs, zone, unit)
                contribution = -loading * delta[index]
                positive = contribution[contribution > 0].sum()
                event = len(events)
                events.append({
                    "event": event, "puma": puma, "alpha": alpha, "family": family, "draw": draw,
                    "block_group": zones[zone], "serialno": serials[unit],
                    "share_of_N": float(np.exp(log_p[zone, unit])),
                    "household_size": float(household_size[unit]), "is_gq": bool(gq[unit]),
                    "logit_shift": float(contribution.sum()),
                })
                for rank, i in enumerate(np.argsort(contribution)[::-1][:args.top], start=1):
                    k = index[i]
                    contributions.append({
                        "event": event, "puma": puma, "alpha": alpha, "rank": rank,
                        "level": labels["level"].iat[k], "geoid": labels["geoid"].iat[k],
                        "category": labels["category"].iat[k], "loading": float(loading[i]),
                        "published": float(targets[k]), "se_floored": float(se[k]),
                        "map_lambda": float(map_lam[k]), "delta": float(delta[k]),
                        "contribution": float(contribution[i]),
                        "share_of_positive": float(contribution[i] / positive) if positive > 0 else np.nan,
                    })
    logger.info("%s a=%g: %d events in %d draws (%.0fs)", puma, alpha, len(events), args.draws,
                time.perf_counter() - started)
    return events, contributions, inputs, unit_max_map, unit_max_vb


# -- summary ------------------------------------------------------------------


FEATURES = ("bg_zero_equiv_share", "carriers", "max_loading", "top1_share", "hhi",
            "pums_over_published")


def summarise(events: pd.DataFrame, parts: pd.DataFrame, features: pd.DataFrame,
              fits: list[tuple[str, float]], draws: int) -> str:
    lines = [f"Events: a (block group, record) cell over the threshold share of N in a VB draw; "
             f"{draws:,} draws per fit.", ""]
    for puma, alpha in fits:
        ev = events[(events.puma == puma) & (events.alpha == alpha)] if len(events) else events
        pt = parts[(parts.puma == puma) & (parts.alpha == alpha)] if len(parts) else parts
        n_draws = ev["draw"].nunique() if len(ev) else 0
        lines.append(f"== {puma} alpha {alpha:g}: {len(ev):,} events in {n_draws:,} draws "
                     f"({100 * n_draws / draws:.2f}% of draws); "
                     f"{ev['serialno'].nunique() if len(ev) else 0} records, "
                     f"{ev['block_group'].nunique() if len(ev) else 0} block groups")
        if not len(ev):
            continue
        lines.append(f"   share of N: median {ev.share_of_N.median():.3f}, max {ev.share_of_N.max():.3f}")
        lead = pt[pt["rank"] == 1]
        table = (lead.groupby(["level", "category"]).size().rename("events_led")
                 .sort_values(ascending=False).head(10).reset_index())
        table["median_share_of_positive"] = [
            lead[(lead.level == r.level) & (lead.category == r.category)].share_of_positive.median()
            for r in table.itertuples()]
        lines += ["   leading constraint (level, category) by events led:"]
        lines += ["     " + r for r in table.to_string(index=False, float_format="%.2f").splitlines()]
        records = ev.groupby("serialno").agg(events=("event", "size"), max_share=("share_of_N", "max"),
                                             household_size=("household_size", "first"))
        lines += ["   records by events:"]
        lines += ["     " + r for r in records.sort_values("events", ascending=False).head(8)
                  .to_string(float_format="%.3f").splitlines()]

    lines += ["", "== Implicated (led >= 1 event at any alpha) against not implicated, "
                  "per PUMA, medians [quartiles]"]
    for puma, group in features.groupby("puma"):
        led = group.filter(like="events_led_").sum(axis=1) > 0
        lines.append(f"   {puma}: {int(led.sum())} implicated of {len(group)} categories")
        for feature in FEATURES:
            def cell(x):
                x = x.dropna()
                if not len(x):
                    return "n/a"
                q1, q2, q3 = np.quantile(x, [0.25, 0.5, 0.75])
                return f"{q2:.3g} [{q1:.3g}, {q3:.3g}]"
            lines.append(f"     {feature:<22} implicated {cell(group.loc[led, feature]):<28} "
                         f"not {cell(group.loc[~led, feature])}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    rng = np.random.default_rng(args.seed)
    floor = floor_spec(args.variance_floor)

    all_events, all_parts, features, fits, maxima = [], [], {}, [], {}
    for puma, alpha, vb_run in args.fit:
        alpha = float(alpha)
        fits.append((puma, alpha))
        events, parts, inputs, unit_map, unit_vb = find_events(args, puma, alpha, Path(vb_run), rng)
        all_events += events
        all_parts += parts
        if puma not in features:
            features[puma] = category_features(inputs, floor)
            features[puma].attrs["N"] = float(inputs.N)
            carriers = category_carriers(inputs)
            weight = inputs.units["weight"].to_numpy(float)
            features[puma]["design_max_w"] = [
                float(weight[carriers[c]].max()) if carriers[c].size else np.nan
                for c in features[puma].category]
        for c in features[puma].category:
            rows = carriers[c]
            maxima[(puma, c, alpha)] = (
                float(unit_map[rows].max()) if rows.size else np.nan,
                float(unit_vb[rows].max()) if rows.size else np.nan,
            )
    # Event ids are per fit; make them global.
    events = pd.DataFrame(all_events)
    parts = pd.DataFrame(all_parts)
    if len(events):
        key = events[["puma", "alpha", "event"]].reset_index().rename(columns={"index": "id"})
        parts = parts.merge(key, on=["puma", "alpha", "event"]).drop(columns="event").rename(
            columns={"id": "event"})
        events["event"] = np.arange(len(events))

    frame = pd.concat(features.values(), ignore_index=True)
    for alpha in sorted({a for _, a in fits}, reverse=True):
        pair = [maxima.get((p, c, alpha), (np.nan, np.nan)) for p, c in zip(frame.puma, frame.category)]
        frame[f"map_max_w_a{alpha:g}"] = [m for m, _ in pair]
        frame[f"vb_max_w_a{alpha:g}"] = [v for _, v in pair]
    for alpha in sorted({a for _, a in fits}, reverse=True):
        led = parts[(parts.alpha == alpha) & (parts["rank"] == 1)] if len(parts) else parts
        if len(led):
            led = led.merge(events[["event", "share_of_N"]], on="event")
            led_max = led.groupby(["puma", "category"])["share_of_N"].max()
        else:
            led_max = pd.Series(dtype=float)
        n_of = {p: features[p].attrs.get("N", np.nan) for p in features}
        frame[f"vb_max_w_led_a{alpha:g}"] = [
            float(led_max.get((p, c), np.nan)) * n_of[p] for p, c in zip(frame.puma, frame.category)]
    for _, alpha in sorted(set(fits), key=lambda f: -f[1]):
        for rank_filter, label in ((1, "events_led"), (None, "events_top")):
            sub = parts[parts.alpha == alpha] if len(parts) else parts
            if rank_filter is not None and len(sub):
                sub = sub[sub["rank"] == rank_filter]
            counts = (sub.groupby(["puma", "category"])["event"].nunique()
                      if len(sub) else pd.Series(dtype=float))
            frame[f"{label}_a{alpha:g}"] = [
                int(counts.get((p, c), 0)) for p, c in zip(frame.puma, frame.category)]

    events.to_csv(args.out / "events.csv", index=False)
    parts.to_csv(args.out / "event_constraints.csv", index=False)
    frame.to_csv(args.out / "constraint_features.csv", index=False)
    text = summarise(events, parts, frame, fits, args.draws)
    (args.out / "summary.txt").write_text(text)
    print(text)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
