"""Where does VB differ from the HMC reference, and does a simulator user see it?

Reads one fit's VB family (a ``run_map.py vb`` result) and the kept ``lambda``
draws of an HMC run on the same problem (``run_mcmc.py sample``), draws
``--draws`` from each, and reports:

1. **PSIS k-hat** of VB against the posterior (Yao et al. 2018): a one-number
   verdict on the joint approximation.
2. **Whitened coordinates.** Both samplers' draws mapped through ``G'(lambda -
   mu)`` of VB's Gaussian part, where HMC's would be standard normal if that
   Gaussian were the posterior. Quantiles over coordinates of HMC's mean and
   sd, the sd ratio VB/HMC, and the coordinates whose 0.1% or 99.9% quantile
   VB puts furthest beyond HMC's -- the directions VB over-reaches.
3. **Walls.** The (block group, record) cells that hold over 1% of ``N`` in
   some VB draw, with their share of ``N`` under each sampler.
4. **Outcomes** (:mod:`pmedm_vb.compare`): per block group, every constrained
   cell and the race-by-income cross-tabulation, VB's mean, median, sd and 90%
   interval against HMC's, in HMC sds. Summarised by kind, with the worst cells.
   The 90% width ratio is summarised only over cells whose HMC interval is at
   least ``--min-width`` people wide; below that the ratio is mostly the floor.
5. **Tract and block group pairs** (:func:`pmedm_vb.compare.pair_comparison`):
   for every category constrained at both levels, the tract and block group
   multipliers' correlation and the sd and skewness of their sum (the direction
   the data see) and difference, VB against HMC -- for all pairs, and for pairs
   whose block group count is published at ``--rare-count`` or less.

Writes ``<out>/<name>_compare.txt`` and three CSVs beside it: ``_outcomes``,
``_coordinates``, ``_walls``, ``_pairs``. ``--out`` defaults to ``<mcmc>/compare``.
With ``--alt-mcmc DIR`` the sampler compared is a second HMC run instead of
VB -- a short run against the long reference, say. The whitening is still
VB's Gaussian part, section 1 is skipped (it needs VB's density), and the
labels read "alt" for the compared run. Needs torch (for ``log pi``) and arviz. Takes minutes on a compute node::

    $CONDA_PREFIX/bin/python experiments/compare_vb_hmc.py --puma 4701502 --alpha 1.0 \\
        --variance-floor zero --vb-run $RUNS/6272357 --mcmc $RUNS/<hmc jobid>/mcmc
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb import compare
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.progress import record_run

from laplace_diagnostic import batched_f, constraint_table, load_vb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--puma", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--taper", choices=["tract", "none"], default="tract")
    parser.add_argument("--area", default="knox-2024-5yr", help="assembled-inputs directory name")
    parser.add_argument("--variance-floor", default="none",
                        help="'none', 'zero' or a number; must match both runs")
    parser.add_argument("--vb-run", type=Path, required=True, help="VB results directory")
    parser.add_argument("--mcmc", type=Path, required=True, help="run_mcmc.py --out directory")
    parser.add_argument("--alt-mcmc", type=Path, default=None,
                        help="compare this HMC run's draws instead of VB's")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=4000, help="draws from each sampler")
    parser.add_argument("--crosstab", nargs=2, action="append", metavar=("PREFIX_A", "PREFIX_B"),
                        help="column-name prefixes to cross; default B03002. B19001.")
    parser.add_argument("--min-width", type=float, default=5.0,
                        help="people; HMC 90%% width below which a cell is left out of the width ratio")
    parser.add_argument("--rare-count", type=float, default=6.0,
                        help="block group published count at or below which a pair counts as rare")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def floor_spec(text: str) -> str | float | None:
    return None if text == "none" else "zero" if text == "zero" else float(text)


def quantile_row(x: np.ndarray, fmt: str = "{:.3f}") -> str:
    x = np.asarray(x)[np.isfinite(x)]
    qs = np.quantile(x, [0.01, 0.1, 0.5, 0.9, 0.99])
    return "  ".join(f"{label} {fmt.format(v)}" for label, v in zip(["p1", "p10", "p50", "p90", "p99"], qs))


def thinned(kept: np.ndarray, draws: int) -> np.ndarray:
    """``(m, <= draws)`` evenly spaced draws from a ``(kept, chains, m)`` trace."""
    flat = kept.reshape(-1, kept.shape[-1])
    return flat[np.unique(np.linspace(0, len(flat) - 1, min(draws, len(flat))).astype(int))].T


def main() -> None:
    import torch

    from pmedm_vb.solvers.vb import _DualTarget

    args = parse_args()
    name = f"{args.puma}_{args.taper}_a{args.alpha:g}"
    out = args.out or (args.mcmc / "compare")
    out.mkdir(parents=True, exist_ok=True)
    record_run(out, args)
    rng = np.random.default_rng(args.seed)

    inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / args.puma)
    taper = None if args.taper == "none" else args.taper
    sigma = inputs.sigma(args.alpha, taper, floor_spec(args.variance_floor))
    q, _, _ = load_vb(args.vb_run / f"{name}.npz")
    with np.load(args.mcmc / f"{name}_trace.npz") as saved:
        kept = saved["lam"]  # (kept, chains, m)
    hmc = thinned(kept, args.draws)
    if args.alt_mcmc:
        with np.load(args.alt_mcmc / f"{name}_trace.npz") as saved:
            vb = thinned(saved["lam"], args.draws)
        alt, alt_source = "alt", args.alt_mcmc
    else:
        vb = q.sample(rng, args.draws)
        alt, alt_source = "VB", args.vb_run
    torch.set_num_threads(max(1, torch.get_num_threads()))
    target = _DualTarget(inputs, sigma, "cpu")
    n = inputs.n

    lines = [f"{alt} ({alt_source}) against HMC ({args.mcmc}) for {name}, variance floor "
             f"{args.variance_floor}; whitened by VB {args.vb_run}",
             f"{hmc.shape[1]:,} HMC draws, {vb.shape[1]:,} {alt} draws; "
             f"{inputs.n_zones} zones x {inputs.n_units:,} units, N = {inputs.N:,.0f}"]

    # 1. PSIS k-hat
    if args.alt_mcmc:
        lines += ["", "1. PSIS k-hat: skipped, the compared draws are not VB's"]
    else:
        log_ratio = -n * batched_f(target, vb, 100) - q.log_density(vb)
        khat = compare.psis_khat(log_ratio)
        lines += ["", "1. PSIS k-hat of VB (under 0.5 good, 0.5-0.7 acceptable, over 0.7 unreliable)",
                  f"   k-hat {khat:.2f}"]

    # 2. Whitened coordinates
    labels = constraint_table(inputs)[["level", "geoid", "constraint", "published", "se"]]
    coords = compare.coordinate_comparison(compare.whitened(q, hmc), compare.whitened(q, vb))
    coords = pd.concat([labels, coords], axis=1)
    coords["lower_overreach"] = coords["q001_ref"] - coords["q001_alt"]
    coords["upper_overreach"] = coords["q999_alt"] - coords["q999_ref"]
    lines += [
        "", "2. Whitened coordinates G'(lambda - mu) of VB's Gaussian part "
            "(HMC's would be N(0, 1) if that Gaussian were the posterior)",
        f"   HMC mean        {quantile_row(coords['mean_ref'])}",
        f"   HMC sd          {quantile_row(coords['sd_ref'])}",
        f"   HMC skewness    {quantile_row(coords['skew_ref'])}",
        f"   {alt:<3} skewness    {quantile_row(coords['skew_alt'])}",
        f"   sd {alt} / HMC     {quantile_row(coords['sd_ratio'])}",
        f"   {alt} 0.1% beyond HMC's (whitened units)   {quantile_row(coords['lower_overreach'])}",
        f"   {alt} 99.9% beyond HMC's (whitened units)  {quantile_row(coords['upper_overreach'])}",
    ]
    for column, label in (("lower_overreach", "lower"), ("upper_overreach", "upper")):
        worst = coords.sort_values(column, ascending=False).head(10)
        lines += [f"   ten coordinates where {alt}'s {label} tail reaches furthest past HMC's:"]
        shown = worst[["level", "geoid", "constraint", "published", "se", "sd_ref", "sd_ratio",
                       "skew_ref", "skew_alt", column]]
        lines += ["     " + row for row in shown.to_string(index=False, float_format="%.3f").splitlines()]

    # 3. Walls
    share, zone, unit = compare.largest_cells(inputs, vb)
    over = share > 0.01
    walls = pd.DataFrame({"zone": zone[over], "unit": unit[over]}).value_counts().rename("vb_draws_over_1pct")
    walls = walls.reset_index().head(20)
    lines += ["", f"3. Walls: cells over 1% of N in some {alt} draw ({int(over.sum()):,} of "
                  f"{over.size:,} {alt} draws; {len(walls)} most frequent cells shown)"]
    if len(walls):
        cells = list(zip(walls["zone"], walls["unit"]))
        for tag, lam in (("hmc", hmc), (alt.lower(), vb)):
            shares = np.exp(compare.cell_log_shares(inputs, lam, cells))
            walls[f"{tag}_p50"], walls[f"{tag}_p99"], walls[f"{tag}_max"] = (
                np.quantile(shares, [0.5, 0.99, 1.0], axis=0))
        walls.insert(2, "record", inputs.units.iloc[:, 0].to_numpy()[walls["unit"]])
        shown = walls.drop(columns=["unit"]).copy()
        for column in shown.columns[3:]:
            shown[column] = shown[column].map(lambda v: f"{100 * v:.4f}%")
        lines += [f"   share of N on each cell, HMC and {alt}:"]
        lines += ["     " + row for row in shown.to_string(index=False).splitlines()]

    # 4. Outcomes
    crosstabs = [tuple(pair) for pair in args.crosstab] if args.crosstab else [compare.RACE_BY_INCOME]
    matrix, meta = compare.outcome_matrix(inputs, crosstabs)
    zones = inputs.zones.iloc[:, 0].to_numpy()
    summaries = [compare.summarise(compare.outcome_draws(inputs, lam, matrix), meta, zones)
                 for lam in (hmc, vb)]
    outcomes = compare.compare_outcomes(*summaries)
    lines += ["", f"4. Outcomes per block group, {alt} against HMC in HMC sds (sd floored at "
                  f"{compare.SD_FLOOR}): {len(meta)} outcomes x {inputs.n_zones} zones"]
    for kind, group in outcomes.groupby("kind"):
        lines += [
            f"   {kind} ({len(group):,} cells):",
            f"     |mean diff|    {quantile_row(group['mean_diff'].abs())}",
            f"     |median diff|  {quantile_row(group['median_diff'].abs())}",
            f"     sd ratio       {quantile_row(group['sd_ratio'])}",
        ]
        wide = group[(group["q95_ref"] - group["q05_ref"]) >= args.min_width]
        lines.append(
            f"     90% width ratio {quantile_row(wide['width_ratio'])}  "
            f"({len(wide):,} cells with HMC width >= {args.min_width:g})"
            if len(wide) else f"     90% width ratio: no cell with HMC width >= {args.min_width:g}"
        )
        worst = group.reindex(group["median_diff"].abs().sort_values(ascending=False).index).head(8)
        shown = worst[["zone", "outcome", "q50_ref", "q50_alt", "sd_ref", "sd_alt",
                       "median_diff", "mean_diff", "width_ratio"]]
        lines += ["     worst by |median diff|:"]
        lines += ["       " + row for row in shown.to_string(index=False, float_format="%.2f").splitlines()]

    # 5. Tract and block group pairs
    pairs = compare.pair_comparison(inputs, hmc, vb)
    lines += ["", f"5. Tract and block group multipliers of the same category "
                  f"({len(pairs):,} pairs; rare = block group published count <= {args.rare_count:g})"]
    stats = ["corr", "sd_sum", "sd_diff", "skew_sum", "skew_diff"]
    for label, group in (("all pairs", pairs), ("rare pairs", pairs[pairs["published_bg"] <= args.rare_count])):
        if not len(group):
            continue
        medians = pd.DataFrame({
            "HMC": [group[f"{s}_ref"].median() for s in stats],
            alt: [group[f"{s}_alt"].median() for s in stats],
        }, index=stats)
        lines += [f"   {label} ({len(group):,}), medians:"]
        lines += ["     " + row for row in medians.to_string(float_format="%.3f").splitlines()]
        lines += [f"     sd of sum, {alt} / HMC   {quantile_row(group['sd_sum_ratio'])}",
                  f"     sd of diff, {alt} / HMC  {quantile_row(group['sd_diff_ratio'])}"]
    if not len(pairs):
        lines.append("   no category is constrained at both levels under the same name")
    skewed = pairs.assign(gap=(pairs["skew_sum_alt"] - pairs["skew_sum_ref"]).abs())
    worst = skewed.sort_values("gap", ascending=False).head(10)
    shown = worst[["constraint", "zone", "published_bg", "carriers", "corr_ref", "corr_alt",
                   "sd_sum_ratio", "skew_sum_ref", "skew_sum_alt"]]
    if len(pairs):
        lines += [f"   ten pairs whose sum is skewed most differently under {alt} and HMC:"]
        lines += ["     " + row for row in shown.to_string(index=False, float_format="%.3f").splitlines()]

    text = "\n".join(lines) + "\n"
    (out / f"{name}_compare.txt").write_text(text)
    outcomes.to_csv(out / f"{name}_outcomes.csv", index=False)
    coords.to_csv(out / f"{name}_coordinates.csv", index=False)
    walls.to_csv(out / f"{name}_walls.csv", index=False)
    pairs.to_csv(out / f"{name}_pairs.csv", index=False)
    with contextlib.suppress(BrokenPipeError):
        print(text)


if __name__ == "__main__":
    main()
