"""What a statewide support changes: one PUMA's HMC posterior on its own records against one on every state record.

Reads the short-HMC draws of two experiments for the same PUMA and alpha -- a
*base* fitted on the PUMA's own PUMS records (e.g. ``tn01_gpu``) and an *alt*
fitted on a statewide support (e.g. ``tn02_state_e0.1``, ``model.epsilon``) --
and reports:

1. **Race and ethnicity by household income** (``B03002 x B19001``), which no
   table constrains, so the prior decides it. For the PUMA as a whole: each
   posterior's mean and 90% interval, the PUMA's own PUMS (*local*) and the
   state's PUMS (*state*), both scaled to the PUMA's ``N``, and
   ``t = (posterior - local) / (state - local)``: 0 where the posterior keeps
   the local joint structure, 1 where it takes the state's. Per block group,
   the alt posterior's shift in base posterior sds
   (:func:`pmedm_vb.compare.compare_outcomes`).
2. **One constrained category** (``--focus``, default ``B03002.nh_nhpi``): both
   posteriors per block group and for the PUMA, the published counts, and the
   rows carrying it in each support.
3. **Borrowed weight**: the share of ``N`` the alt posterior puts on records
   from outside the PUMA, overall and per block group, against the prior's.
4. **The largest cell of each draw**: its share of ``N``, and whether the alt
   posterior's sits on a borrowed record.
5. **Held-out tables**, where both inputs folders have them (``run_map.py
   heldout``, with ``--epsilon`` for the statewide one): each posterior's
   prediction of every held-out tract and block group cell against the
   published estimate and its standard error -- the error, the predictive
   ``z = (mean - Y) / sqrt(SE^2 + sd^2)``, 90% predictive coverage and a
   Gaussian log score -- by level and by relation to the fitted tables, and
   how often the statewide posterior is closer (``heldout_scores.py``, shared
   with ``compare_variants.py``).

Section 1 also gives, for each race and ethnicity, the distribution of its
persons over household income bands (shares within the race): under each
posterior, the local PUMS and the state PUMS, with the total variation
distance between them. For rare groups this is the right measure, since the
tables, not the PUMS, set how many people the group has.

The state PUMS comes from the alt prior itself: its weights are
``(1 - epsilon) local + epsilon state`` (:func:`pmedm_vb.assemble.design.support_weights`),
so ``state = (prior - (1 - epsilon) local) / epsilon``. Within a merged row of
the statewide support, the weight splits between the PUMA's own records and
borrowed ones in proportion to their prior weights, since records with
identical rows of ``X`` are tilted alike.

Usage, on a CPU node::

    sbatch experiments/run_python.sbatch compare_support.py --puma 4701502 \\
        --base-exp tn01_gpu --alt-exp tn02_state_e0.1

Writes ``puma_totals.csv``, ``income_within_race.csv``, ``block_groups.csv``,
``borrowed.csv``, ``largest_cells.csv``, ``heldout_cells.csv`` and ``summary.txt`` to ``--out`` (default
``<root>/<alt-exp>/<puma>/compare_<base-exp>``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx  # noqa: E402
from heldout_scores import (HeldoutPredictor, area_names, level_frame, score_cells,  # noqa: E402
                            score_groups)

from pmedm_vb import compare  # noqa: E402
from pmedm_vb.assemble.heldout import HeldOut  # noqa: E402
from pmedm_vb.assemble.inputs import PMEDMInputs  # noqa: E402
from pmedm_vb.progress import record_run  # noqa: E402
from pmedm_vb.solvers.base import ConstraintOperator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--puma", required=True)
    parser.add_argument("--base-exp", required=True, help="experiment fitted on the PUMA's own records")
    parser.add_argument("--alt-exp", required=True, help="experiment fitted on a statewide support")
    parser.add_argument("--root", type=Path, default=rx.DEFAULT_ROOT,
                        help=f"folder holding the experiment folders (default {rx.DEFAULT_ROOT})")
    parser.add_argument("--alpha", type=float, default=None,
                        help="default: the alt experiment's first alpha")
    parser.add_argument("--kind", default="short", help="the HMC run to read (hmc_<kind>)")
    parser.add_argument("--draws", type=int, default=1000,
                        help="draws used from each run, evenly spaced over its kept draws")
    parser.add_argument("--focus", default="B03002.nh_nhpi",
                        help="a block group constraint column to report in detail")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_run(root: Path, exp_name: str, puma: str, alpha: float | None, kind: str):
    """The experiment record, its inputs for ``puma``, the alpha and the kept HMC draws."""
    spec = json.loads((root / exp_name / "experiment.json").read_text())
    alpha = spec["alphas"][0] if alpha is None else alpha
    name = rx.fit_name(spec, puma, alpha)
    trace = root / exp_name / puma / f"hmc_{kind}" / f"{name}_trace.npz"
    with np.load(trace) as saved:
        kept = saved["lam"]  # (kept, chains, m): the multipliers the data see
    path = rx.inputs_dir(spec) / puma
    inputs = PMEDMInputs.load(path)
    heldout = HeldOut.load(path) if HeldOut.exists(path) else None
    return spec, inputs, alpha, kept, trace, heldout


def thinned(kept: np.ndarray, draws: int) -> np.ndarray:
    """``(m, <= draws)`` evenly spaced draws from a ``(kept, chains, m)`` trace."""
    flat = kept.reshape(-1, kept.shape[-1])
    return flat[np.unique(np.linspace(0, len(flat) - 1, min(draws, len(flat))).astype(int))].T


def borrowed_fraction(inputs: PMEDMInputs) -> np.ndarray:
    """Per row of a statewide support, the share of its prior weight on records
    from outside the PUMA (0 for a PUMA-only problem)."""
    members = inputs.unit_members
    if members is None:
        return np.zeros(inputs.n_units)
    members = members[members["row"] >= 0]
    rows = members["row"].to_numpy()
    weight = members["weight"].to_numpy(dtype=float)
    outside = ~members["in_puma"].to_numpy(dtype=bool)
    total = np.bincount(rows, weights=weight, minlength=inputs.n_units)
    borrowed = np.bincount(rows, weights=weight * outside, minlength=inputs.n_units)
    return np.divide(borrowed, total, out=np.zeros_like(total), where=total > 0)


def draw_summaries(inputs: PMEDMInputs, lam: np.ndarray, outcomes: sp.spmatrix,
                   borrowed: np.ndarray, heldout: HeldOut | None = None) -> dict[str, np.ndarray]:
    """Per draw (columns of ``lam``): outcome counts per zone ``(draws, zones, k)``,
    borrowed weight per zone ``(draws, zones)``, the largest cell's share of
    ``N``, zone and row, and with ``heldout`` its predicted cells per block
    group and per tract."""
    op = ConstraintOperator(inputs)
    with np.errstate(divide="ignore"):
        log_q = np.log(inputs.q)
    outcomes = sp.csr_matrix(outcomes)
    d = lam.shape[1]
    out = {"outcomes": np.empty((d, inputs.n_zones, outcomes.shape[1]), dtype=np.float32),
           "borrowed": np.empty((d, inputs.n_zones)),
           "max_share": np.empty(d), "max_zone": np.empty(d, int), "max_row": np.empty(d, int)}
    if heldout is not None:
        held = HeldoutPredictor(inputs, heldout)
        out["held_bg"] = np.empty((d, inputs.n_zones, held.h_b.shape[1]), dtype=np.float32)
        out["held_tract"] = np.empty((d, held.A_T.shape[0], held.h_t.shape[1]), dtype=np.float32)
    for k in range(d):
        logits = log_q - op.adjoint(lam[:, k])
        log_z = logsumexp(logits)
        W = inputs.N * np.exp(logits - log_z)
        out["outcomes"][k] = (outcomes.T @ W.T).T
        out["borrowed"][k] = W @ borrowed
        flat = int(np.argmax(logits))
        out["max_share"][k] = np.exp(logits.flat[flat] - log_z)
        out["max_zone"][k], out["max_row"][k] = divmod(flat, inputs.n_units)
        if heldout is not None:
            out["held_bg"][k], out["held_tract"][k] = held(W)
    return out


def income_within_race(totals: pd.DataFrame, base_draws: np.ndarray, alt_draws: np.ndarray
                       ) -> pd.DataFrame:
    """Shares of each race's persons over household income bands: posterior
    means and 90% intervals under base and alt, and the local and state PUMS.
    ``*_draws`` are ``(draws, k)`` PUMA totals in ``totals``' column order."""
    cross = totals[totals["kind"] == "crosstab"]
    parts = cross["outcome"].str.split(" x ", expand=True)
    rows = []
    for race, idx in cross.groupby(parts[0]).groups.items():
        cols = [totals.index.get_loc(i) for i in idx]
        bands = parts.loc[idx, 1].to_numpy()

        def shares(x):
            total = x.sum(-1, keepdims=True)
            return np.divide(x, total, out=np.full_like(x, np.nan, dtype=float), where=total > 0)

        b, a = shares(base_draws[:, cols].astype(float)), shares(alt_draws[:, cols].astype(float))
        local = shares(totals.loc[idx, "local"].to_numpy(float))
        state = shares(totals.loc[idx, "state"].to_numpy(float))
        for j, band in enumerate(bands):
            rows.append({
                "race": race, "income": band, "local": local[j], "state": state[j],
                "base_mean": np.nanmean(b[:, j]), "base_q05": np.nanquantile(b[:, j], 0.05),
                "base_q95": np.nanquantile(b[:, j], 0.95),
                "alt_mean": np.nanmean(a[:, j]), "alt_q05": np.nanquantile(a[:, j], 0.05),
                "alt_q95": np.nanquantile(a[:, j], 0.95),
                "base_persons": base_draws[:, cols].sum(1).mean(),
                "alt_persons": alt_draws[:, cols].sum(1).mean(),
            })
    return pd.DataFrame(rows)


def tvd(a: np.ndarray, b: np.ndarray) -> float:
    """Total variation distance between two distributions (NaN if either is undefined)."""
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    return 0.5 * float(np.abs(a - b).sum())


def interval(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean, sd, 5% and 95% quantiles over the first axis."""
    q05, q95 = np.quantile(draws, [0.05, 0.95], axis=0)
    return draws.mean(0), draws.std(0, ddof=1), q05, q95


def main() -> None:
    args = parse_args()
    base_spec, base, alpha, base_kept, base_trace, base_held = load_run(
        args.root, args.base_exp, args.puma, args.alpha, args.kind)
    alt_spec, alt, alt_alpha, alt_kept, alt_trace, alt_held = load_run(
        args.root, args.alt_exp, args.puma, alpha, args.kind)
    if base_held is None or alt_held is None:
        base_held = alt_held = None  # held-out scoring needs both
    elif base_held.names != alt_held.names:
        raise SystemExit("the two runs' held-out tables differ")
    if base.unit_members is not None or alt.unit_members is None:
        raise SystemExit("--base-exp must be fitted on the PUMA's own records and --alt-exp "
                         "on a statewide support (model.epsilon)")
    if list(base.bg_constraints) != list(alt.bg_constraints) or base.n_zones != alt.n_zones:
        raise SystemExit("the two runs' block groups or constraint columns differ")
    epsilon = float(alt.support["epsilon"])
    out = args.out or (args.root / args.alt_exp / args.puma / f"compare_{args.base_exp}")
    out.mkdir(parents=True, exist_ok=True)
    record_run(out, args)

    base_X, meta = compare.outcome_matrix(base)
    alt_X, alt_meta = compare.outcome_matrix(alt)
    assert list(meta["outcome"]) == list(alt_meta["outcome"])
    zones = list(base.zones["block_group_geoid"])
    base_s = draw_summaries(base, thinned(base_kept, args.draws), base_X, np.zeros(base.n_units),
                            base_held)
    frac = borrowed_fraction(alt)
    alt_s = draw_summaries(alt, thinned(alt_kept, args.draws), alt_X, frac, alt_held)

    # 1-2. PUMA totals: both posteriors, the local and state PUMS, the published counts.
    weights = lambda inputs: inputs.units["weight"].to_numpy(dtype=float)  # noqa: E731
    local = sp.csr_matrix(base_X).T @ weights(base)
    prior = sp.csr_matrix(alt_X).T @ weights(alt)
    state = (prior - (1 - epsilon) * local) / epsilon
    published = np.full(len(meta), np.nan)
    constrained = (meta["kind"] == "constrained").to_numpy()
    published[constrained] = np.asarray(base.Y_B).sum(0)
    b_mean, b_sd, b_q05, b_q95 = interval(base_s["outcomes"].sum(1))
    a_mean, a_sd, a_q05, a_q95 = interval(alt_s["outcomes"].sum(1))
    gap = state - local
    # t only where local and state differ by more than the base posterior's own spread.
    usable = np.abs(gap) > np.maximum(b_sd, 1.0)
    totals = meta.assign(
        local=local, state=state, prior_alt=prior, published=published,
        base_mean=b_mean, base_q05=b_q05, base_q95=b_q95, base_sd=b_sd,
        alt_mean=a_mean, alt_q05=a_q05, alt_q95=a_q95, alt_sd=a_sd,
        shift_sds=(a_mean - b_mean) / np.maximum(b_sd, compare.SD_FLOOR),
        t_base=np.where(usable, (b_mean - local) / np.where(usable, gap, 1), np.nan),
        t_alt=np.where(usable, (a_mean - local) / np.where(usable, gap, 1), np.nan),
    )
    totals.to_csv(out / "puma_totals.csv", index=False)
    within = income_within_race(totals, base_s["outcomes"].sum(1), alt_s["outcomes"].sum(1))
    within.to_csv(out / "income_within_race.csv", index=False)

    # Per block group, alt against base (in base sds), with the published counts.
    by_bg = compare.compare_outcomes(compare.summarise(base_s["outcomes"], meta, zones),
                                     compare.summarise(alt_s["outcomes"], meta, zones))
    published_bg = pd.DataFrame({
        "zone": np.repeat(zones, len(base.bg_constraints)),
        "outcome": np.tile(list(base.bg_constraints), len(zones)),
        "published": np.asarray(base.Y_B).ravel(),
    })
    by_bg = by_bg.merge(published_bg, on=["zone", "outcome"], how="left")
    by_bg.to_csv(out / "block_groups.csv", index=False)

    # 3. Borrowed weight, overall and per block group.
    prior_borrowed = float(weights(alt) @ frac / alt.N)
    total_borrowed = alt_s["borrowed"].sum(1) / alt.N
    bg_mean, _, bg_q05, bg_q95 = interval(alt_s["borrowed"])
    borrowed = pd.DataFrame({"zone": zones, "borrowed_mean": bg_mean, "borrowed_q05": bg_q05,
                             "borrowed_q95": bg_q95})
    borrowed.to_csv(out / "borrowed.csv", index=False)

    # 4. Largest cells: the alt posterior's, with whether they are borrowed.
    units = alt.units.reset_index(drop=True)
    cells = pd.DataFrame({"zone": np.asarray(zones)[alt_s["max_zone"]], "row": alt_s["max_row"],
                          "share_pct": 100 * alt_s["max_share"]})
    cells["borrowed_fraction"] = frac[cells["row"]]
    for column in ("SERIALNO", "n_records", "n_own", "is_group_quarters"):
        if column in units:
            cells[column] = units[column].to_numpy()[cells["row"]]
    top = (cells.groupby(["zone", "row"], as_index=False)
           .agg(max_share_pct=("share_pct", "max"), draws=("share_pct", "size"),
                **{c: (c, "first") for c in ("borrowed_fraction", "SERIALNO", "n_records", "n_own")
                   if c in cells})
           .sort_values("max_share_pct", ascending=False))
    top.to_csv(out / "largest_cells.csv", index=False)

    # The summary.
    lines = [
        f"PUMA {args.puma}, alpha {alpha:g}: base {args.base_exp} ({base.n_units:,} rows, the "
        f"PUMA's own records) against alt {args.alt_exp} ({alt.n_units:,} rows, statewide, "
        f"epsilon {epsilon:g}); N = {base.N:,.0f}",
        f"draws: {base_s['max_share'].size:,} from {base_trace}",
        f"       {alt_s['max_share'].size:,} from {alt_trace}",
        "",
        "1. Race and ethnicity by household income (B03002 x B19001), PUMA totals",
        "   local and state: the PUMS, the PUMA's own and the state's, scaled to N;",
        "   t = (posterior - local) / (state - local): 0 keeps the local joint structure, "
        "1 takes the state's;",
        "   t only where |state - local| exceeds the base posterior sd (and 1).",
    ]
    cross = totals[totals["kind"] == "crosstab"]
    t_cells = cross.dropna(subset=["t_alt"])
    if len(t_cells):
        lines.append(f"   cells with a usable t: {len(t_cells)} of {len(cross)}; t_base median "
                     f"{t_cells['t_base'].median():.2f}, t_alt median {t_cells['t_alt'].median():.2f}, "
                     f"t_alt p10-p90 {t_cells['t_alt'].quantile(0.1):.2f} to "
                     f"{t_cells['t_alt'].quantile(0.9):.2f}")
    lines.append(f"   PUMA-total shift, alt - base in base sds: median |shift| "
                 f"{cross['shift_sds'].abs().median():.2f}, over 2 sds in "
                 f"{int((cross['shift_sds'].abs() > 2).sum())} of {len(cross)} cells")
    lines.append("   twelve largest shifts:")
    show = cross.reindex(cross["shift_sds"].abs().sort_values(ascending=False).index).head(12)
    lines += ["   " + line for line in show[[
        "outcome", "local", "state", "base_mean", "base_q05", "base_q95", "alt_mean", "alt_q05",
        "alt_q95", "shift_sds", "t_base", "t_alt"]].round(2).to_string(index=False).splitlines()]
    bg_cross = by_bg[by_bg["kind"] == "crosstab"]
    lines += [
        f"   per block group ({len(bg_cross):,} cells): median |mean shift| "
        f"{bg_cross['mean_diff'].abs().median():.2f} base sds, over 2 in "
        f"{100 * (bg_cross['mean_diff'].abs() > 2).mean():.1f}%; 90% interval width alt/base "
        f"median {bg_cross['width_ratio'].median():.2f}",
        "",
        "1b. Income distribution within each race and ethnicity (persons in households, PUMA)",
        "    TVD: total variation distance between income distributions (0 identical, 1 disjoint);",
        "    pull = TVD(alt, local) / TVD(state, local): how far alt moved toward the state's shape",
    ]
    race_rows = []
    for race, g in within.groupby("race", sort=False):
        loc, sta = g["local"].to_numpy(), g["state"].to_numpy()
        bm, am = g["base_mean"].to_numpy(), g["alt_mean"].to_numpy()
        d_ls = tvd(loc, sta)
        race_rows.append({
            "race": race.split(".", 1)[1], "persons_base": g["base_persons"].iloc[0],
            "persons_alt": g["alt_persons"].iloc[0], "TVD_local_state": d_ls,
            "TVD_base_local": tvd(bm, loc), "TVD_alt_local": tvd(am, loc),
            "TVD_alt_state": tvd(am, sta), "TVD_alt_base": tvd(am, bm),
            "pull": tvd(am, loc) / d_ls if d_ls and d_ls == d_ls else float("nan"),
        })
    lines += ["    " + line for line in pd.DataFrame(race_rows).round(3).to_string(index=False)
              .splitlines()]
    lines += ["    (shares per band with 90% intervals: income_within_race.csv)", "",
              f"2. {args.focus}"]
    if args.focus in set(totals["outcome"]):
        k = list(base.bg_constraints).index(args.focus)
        row = totals[totals["outcome"] == args.focus].iloc[0]
        carriers = lambda X: np.asarray(sp.csc_matrix(X)[:, k].todense()).ravel() > 0  # noqa: E731
        base_c, alt_c = carriers(base.X_B), carriers(alt.X_B)
        lines += [
            f"   PUMA total: published {row['published']:.0f}; base {row['base_mean']:.1f} "
            f"[{row['base_q05']:.1f}, {row['base_q95']:.1f}]; alt {row['alt_mean']:.1f} "
            f"[{row['alt_q05']:.1f}, {row['alt_q95']:.1f}]; local PUMS {row['local']:.1f}, "
            f"state PUMS {row['state']:.1f}",
            f"   rows carrying it: base {int(base_c.sum())}; alt {int(alt_c.sum())}, of them "
            f"{int((alt_c & (frac >= 1 - 1e-12)).sum())} wholly borrowed",
        ]
        focus = by_bg[by_bg["outcome"] == args.focus][[
            "zone", "published", "mean_ref", "q05_ref", "q95_ref", "mean_alt", "q05_alt", "q95_alt"]]
        lines += ["   per block group (ref = base, alt = statewide):"]
        lines += ["   " + line for line in focus.round(2).to_string(index=False).splitlines()]
    else:
        lines.append("   not a block group constraint column here")
    q05, q50, q95 = np.quantile(total_borrowed, [0.05, 0.5, 0.95])
    lines += [
        "",
        "3. Weight on records from outside the PUMA (alt)",
        f"   prior share {100 * prior_borrowed:.2f}%; posterior median {100 * q50:.2f}% "
        f"[{100 * q05:.2f}%, {100 * q95:.2f}%] of N",
        f"   per block group, posterior mean of borrowed households: min "
        f"{borrowed['borrowed_mean'].min():.0f}, median {borrowed['borrowed_mean'].median():.0f}, "
        f"max {borrowed['borrowed_mean'].max():.0f} (borrowed.csv)",
        "",
        "4. Each draw's largest cell (one row in one block group), share of N",
    ]
    for label, s in (("base", base_s), ("alt", alt_s)):
        lines.append(f"   {label}: median {100 * np.median(s['max_share']):.3f}%, "
                     f"p99 {100 * np.quantile(s['max_share'], 0.99):.3f}%, "
                     f"max {100 * s['max_share'].max():.3f}%")
    lines.append(f"   alt draws whose largest cell is wholly borrowed: "
                 f"{100 * (cells['borrowed_fraction'] >= 1 - 1e-12).mean():.1f}%, partly borrowed: "
                 f"{100 * ((cells['borrowed_fraction'] > 0) & (cells['borrowed_fraction'] < 1 - 1e-12)).mean():.1f}%")
    lines.append("   ten largest (largest_cells.csv):")
    lines += ["   " + line for line in top.head(10).round(4).to_string(index=False).splitlines()]

    lines += ["", "5. Held-out tables (not fitted): each posterior's prediction against the published estimates"]
    if base_held is None:
        lines.append("   skipped: build them in both inputs folders first (run_map.py heldout, "
                     "with --epsilon for the statewide one)")
    else:
        frames, summary = [], []
        areas = area_names(base)
        for level, key in (("block group", "held_bg"), ("tract", "held_tract")):
            Y, v = base_held.Y[level], base_held.v[level]
            scores = {tag: score_cells(s[key], Y, v) for tag, s in (("base", base_s), ("alt", alt_s))}
            frame = level_frame(base_held, level, areas[level], scores)
            frames.append(frame)
            # By relation, and race x poverty (B17010A-I) on its own too: the joint
            # structure borrowing changes.
            for rel, g in score_groups(frame):
                ok = g.dropna(subset=["base_pred_z", "alt_pred_z"])
                summary.append({
                    "level": level, "relation": rel, "cells": len(ok),
                    "rmse_base": np.sqrt((ok["base_error"] ** 2).mean()),
                    "rmse_alt": np.sqrt((ok["alt_error"] ** 2).mean()),
                    "pred_z2_base": (ok["base_pred_z"] ** 2).mean(),
                    "pred_z2_alt": (ok["alt_pred_z"] ** 2).mean(),
                    "cover90_base": ok["base_covers"].mean(), "cover90_alt": ok["alt_covers"].mean(),
                    "logscore_alt_minus_base": (ok["alt_log_score"] - ok["base_log_score"]).sum(),
                    "alt_closer": (ok["alt_error"].abs() < ok["base_error"].abs()).mean(),
                })
        pd.concat(frames).to_csv(out / "heldout_cells.csv", index=False)
        lines += [
            "   rmse: root mean squared error of the posterior mean; pred_z2: mean squared predictive z",
            "   (about 1 if calibrated); cover90: share of cells within the 90% predictive interval",
            "   (about 0.9 if calibrated); logscore_alt_minus_base: summed Gaussian log score, alt minus",
            "   base (positive favours the statewide support); alt_closer: share of cells where alt's",
            "   mean is nearer the published estimate",
        ]
        lines += ["   " + line for line in pd.DataFrame(summary).round(3).to_string(index=False)
                  .splitlines()]
        lines.append("   (every cell: heldout_cells.csv)")
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
