"""Diagnose the structure of the ACS replicate covariance.

Four questions, all bearing on how ``Sigma`` should be represented, and all
answerable from one or two downloaded tables:

1. **Does variance scale with the count?** If ``v ~ w * y``, deviations divided
   by ``sqrt(y)`` are roughly homoskedastic and the count is the natural
   scaling. If not, dividing by the standard error -- which forces the
   diagonal to one and discards the relationship -- is the honest fallback.
2. **Do the modelled zero cells sit on that curve?** Census gives a zero count
   ``Var = w * k``. If that is the small-count limit of the same relationship,
   zero cells stop being a special case; if it sits off the curve, the model
   is doing something else and the two regimes stay separate.
3. **Is the within-zone correlation structure homogeneous?** If a zone's
   cell-to-cell correlations are a property of the construct rather than of
   the zone, they can be pooled across zones, and the rank-80 ceiling on a
   per-zone estimate stops binding.
4. **Are cross-zone correlations negligible?** ``Sigma`` is only block-diagonal
   by zone if they are. ACS rakes to county controls, which would induce
   negative dependence between zones in a county, so this is not obviously
   true and is worth measuring rather than assuming.

Answers 3 and 4 together decide the representation: a pooled ``c x c``
correlation rescaled per zone by the published variances, which is block
diagonal and well conditioned, or ``D + L L'`` with shrinkage, which is not
block diagonal but assumes less.

Checked against synthetic data with known structure before being pointed at
real files. On homogeneous, independent zones with ``Var = 18*y`` it returns
b = 0.995, exp(a) = 18.2, a homogeneity ratio of 0.99 and a cross-zone ratio
of 1.00; with cross-zone dependence of 0.25 injected it recovers a signed mean
of +0.2505 without disturbing the homogeneity ratio; with the correlation made
to vary by zone it returns a homogeneity ratio of 2.21 without disturbing the
cross-zone statistics. The two axes are separable, which is what makes the
answers actionable one at a time.

One caution the synthetic runs made plain: the raw off-block share of the
Frobenius norm is not interpretable. With Z zones there are Z^2 - Z off-block
entries against Z diagonal blocks, so noise dominates it by multiplicity --
on independent zones it exceeds 90 percent while every correlation is
essentially zero. It is printed for continuity with the calibrated statistics
beneath it, which are the ones to read.

Run it from somewhere census.gov is reachable, with pmedm_vb importable:

    $CONDA_PREFIX/bin/python experiments/covariance_structure.py
    $CONDA_PREFIX/bin/python experiments/covariance_structure.py \\
        --table B01001 --county 093 --county 155
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pmedm_vb.config import StudyArea
from pmedm_vb.data import variance as v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default="47", help="state FIPS")
    parser.add_argument(
        "--county",
        action="append",
        default=None,
        help="county FIPS; repeatable, omit for the whole state",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--span", type=int, default=5)
    parser.add_argument("--geography", default="block group")
    parser.add_argument(
        "--table",
        default="B01001",
        help="table to diagnose; wants several cells (B01001 has 49)",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=2000,
        help="zone pairs sampled for the between-zone comparison",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load(area: StudyArea, table: str, geography: str):
    """Replicate frame, its deviations, and the per-cell replicate variance."""
    frame = v.fetch_replicates(area, [table], geography=geography)
    estimates = frame["estimate"].to_numpy(float)
    deviations = v.replicate_matrix(frame) - estimates[:, None]
    variances = v.SDR_FACTOR * np.square(deviations).sum(axis=1)
    return frame, estimates, deviations, variances


def q1_variance_versus_count(estimates, variances, weight) -> None:
    """Is v proportional to y, and with what constant?"""
    print("\n1. Variance against count")
    usable = (estimates > 0) & (variances > 0)
    y, var = estimates[usable], variances[usable]
    print(f"   cells with y > 0 and v > 0: {usable.sum()} of {len(estimates)}")
    if usable.sum() < 50:
        print("   too few to say anything")
        return

    ratio = var / y
    print(f"   v/y quartiles: {np.percentile(ratio, [25, 50, 75]).round(2)}")
    print(f"   published average weight w = {weight}")
    print("   -- if v ~ w*y, the median sits near w and the bins below are flat")

    edges = np.unique(np.quantile(y, np.linspace(0, 1, 9)))
    print(f"\n   {'count range':>22}  {'n':>6}  {'median v/y':>11}  {'median v':>10}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (y >= lo) & (y < hi)
        if inside.sum() < 10:
            continue
        print(f"   {f'{lo:.0f} - {hi:.0f}':>22}  {inside.sum():>6}  "
              f"{np.median(ratio[inside]):>11.2f}  {np.median(var[inside]):>10.1f}")

    # log v = a + b log y. b near 1 means variance proportional to the count;
    # b near 2 means a constant coefficient of variation instead.
    big = y >= 5
    if big.sum() > 50:
        slope, intercept = np.polyfit(np.log(y[big]), np.log(var[big]), 1)
        resid = np.log(var[big]) - (slope * np.log(y[big]) + intercept)
        stderr = np.sqrt(
            np.sum(resid**2)
            / (big.sum() - 2)
            / np.sum((np.log(y[big]) - np.log(y[big]).mean()) ** 2)
        )
        print(f"\n   log v = a + b log y  (y >= 5, n = {big.sum()})")
        print(f"     b = {slope:.3f} +/- {1.96 * stderr:.3f}   "
              f"exp(a) = {np.exp(intercept):.2f}")
        print(f"     b = 1 means v proportional to y, and exp(a) should be near w = {weight}")
        print("     b = 2 means constant coefficient of variation instead")
        return slope, intercept
    return None


def q2_zero_cells(estimates, variances, fit, area, geography) -> None:
    """Does w*k continue the curve, or sit off it?"""
    print("\n2. Zero cells against that curve")
    modelled = v.zero_cell_variance(area, geography=geography)
    values = np.unique(modelled.to_numpy().round(2))
    print(f"   modelled Var_0 = w*k: {values[:5]}"
          f"{' ...' if len(values) > 5 else ''}")

    small = (estimates > 0) & (estimates <= 3) & (variances > 0)
    if small.sum() >= 10:
        print(f"   observed median v for y in 1..3 (n = {small.sum()}): "
              f"{np.median(variances[small]):.1f}")
    if fit is not None:
        slope, intercept = fit
        for count in (1, 2, 5):
            print(f"   fitted v at y = {count}: "
                  f"{np.exp(intercept + slope * np.log(count)):.1f}")
    print("   -- if the modelled value sits among these, the zero-count model is")
    print("      the small-count limit of the same relationship, not a separate rule")


def zone_blocks(frame, deviations, variances):
    """Per-zone standardised deviation blocks, cells aligned across zones."""
    geoids = frame.index.get_level_values("geoid").to_numpy()
    cells = frame.index.get_level_values("cell").to_numpy()
    cell_order = sorted(set(cells))
    index = {cell: i for i, cell in enumerate(cell_order)}

    scale = np.where(variances > 0, np.sqrt(np.where(variances > 0, variances, 1)), np.inf)
    standardised = deviations / scale[:, None]      # zero-variance rows -> 0

    blocks, kept = {}, []
    for geoid in pd.unique(geoids):
        rows = geoids == geoid
        block = np.zeros((len(cell_order), deviations.shape[1]))
        block[[index[c] for c in cells[rows]]] = standardised[rows]
        blocks[geoid] = block
        kept.append(geoid)
    return np.array([blocks[g] for g in kept]), kept, cell_order


def correlation(block: np.ndarray, columns: slice) -> np.ndarray:
    """Correlation among cells from a slice of the replicate columns."""
    part = block[:, columns]
    scale = np.sqrt(np.square(part).sum(axis=1))
    scale[scale == 0] = np.inf
    unit = part / scale[:, None]
    return unit @ unit.T


def q3_homogeneity(blocks, rng, pairs) -> None:
    """Between-zone correlation differences against a within-zone null.

    The null splits each zone's 80 replicates in half and compares the two
    halves: that difference is estimation noise alone. Between-zone
    differences use the same half, so both carry the same noise, and the
    comparison is between like and like.

    The split is a heuristic. The 80 SDR replicates are generated by a
    specific scheme rather than drawn independently, so treating two halves
    as exchangeable is an assumption this script does not verify.
    """
    print("\n3. Homogeneity of within-zone correlation")
    n_zones, n_cells, n_reps = blocks.shape
    print(f"   zones: {n_zones}   cells: {n_cells}   replicates: {n_reps}")
    if n_zones < 10 or n_cells < 3:
        print("   too small to compare")
        return

    half = n_reps // 2
    first = np.array([correlation(b, slice(0, half)) for b in blocks])
    second = np.array([correlation(b, slice(half, n_reps)) for b in blocks])

    off = ~np.eye(n_cells, dtype=bool)
    within = np.array([np.abs(a - b)[off].mean() for a, b in zip(first, second)])

    left = rng.integers(0, n_zones, pairs)
    right = rng.integers(0, n_zones, pairs)
    keep = left != right
    between = np.abs(first[left[keep]] - first[right[keep]])[:, off].mean(axis=1)

    print(f"   mean |dR| within a zone, half against half: {within.mean():.4f}")
    print(f"   mean |dR| between zones, same half:         {between.mean():.4f}")
    print(f"   ratio between/within: {between.mean() / within.mean():.2f}")
    print("   -- near 1 means correlations look homogeneous and can be pooled;")
    print("      well above 1 means they genuinely differ by zone")

    full = np.array([correlation(b, slice(0, n_reps)) for b in blocks])
    pooled = full.mean(axis=0)
    print(f"   pooled off-diagonal correlation: mean {pooled[off].mean():+.3f}, "
          f"max |r| {np.abs(pooled[off]).max():.3f}")


def q4_cross_zone(blocks) -> None:
    """How much covariance lives outside the per-zone blocks, against the null.

    Computed from the factor rather than by forming the matrix: for an n x 80
    factor F, ``||F F'||_F = ||F' F||_F``, and the second is 80 x 80.

    The raw share is not interpretable on its own. With Z zones there are
    Z^2 - Z off-diagonal blocks against Z diagonal ones, so for any realistic
    Z the off-block mass is dominated by estimation noise through sheer
    multiplicity -- on independent zones it runs well above 90 percent while
    every individual correlation is essentially zero. What matters is the size
    of a typical cross-zone correlation against what independence predicts:
    two independent unit vectors of length R have ``E[r^2] = 1/R``, so the
    null RMS is ``1/sqrt(R)``.
    """
    print("\n4. Cross-zone correlation")
    n_zones, n_cells, n_reps = blocks.shape
    flat = blocks.reshape(n_zones * n_cells, n_reps)

    norms = np.square(flat).sum(axis=1)
    usable = norms > 0
    per_zone = (np.square(blocks).sum(axis=2) > 0).sum(axis=1)
    n_total, n_off = usable.sum(), int(usable.sum()) ** 2 - int((per_zone ** 2).sum())
    if n_off <= 0 or n_total == 0:
        print("   nothing to compare")
        return

    gram = flat.T @ flat
    total = np.square(gram).sum()
    block_total = sum(np.square(b.T @ b).sum() for b in blocks)
    outside = max(total - block_total, 0.0)

    # Rows are standardised, so a diagonal entry is a correlation of 1. That
    # calibrates the gram units into correlation units without needing the
    # 4/80 factor here.
    diagonal = np.square(norms[usable]).sum()
    mean_r2_off = (outside * n_total / diagonal) / n_off
    rms_off, null = np.sqrt(mean_r2_off), 1 / np.sqrt(n_reps)

    print(f"   off-block share of ||Sigma||_F^2: {outside / total:.4f} "
          f"(uninformative on its own -- see below)")
    print(f"   RMS cross-zone correlation: {rms_off:.4f}")
    print(f"   expected under independence (1/sqrt({n_reps})): {null:.4f}")
    print(f"   ratio: {rms_off / null:.2f}")
    print("   -- near 1 means cross-zone correlation is indistinguishable from")
    print("      noise, and Sigma may be treated as block diagonal by zone")

    # The signed mean is the sharper test for a systematic effect: noise
    # averages away, a raking-induced negative dependence does not. Computed
    # exactly without forming the Z x Z matrix, since the sum of all its
    # entries is ||sum_z m_z||^2.
    means = []
    for cell in range(n_cells):
        m = blocks[:, cell, :]
        keep = np.square(m).sum(axis=1) > 0
        if keep.sum() < 2:
            continue
        m = m[keep]
        z = len(m)
        means.append(
            (np.square(m.sum(axis=0)).sum() - np.square(m).sum())
            / (z * z - z) * (z / np.square(m).sum())
        )
    means = np.array(means)
    if len(means):
        print(f"   signed mean cross-zone correlation per cell: "
              f"median {np.median(means):+.4f}, "
              f"range [{means.min():+.4f}, {means.max():+.4f}]")
        print(f"   null standard error of that mean is roughly "
              f"{1 / np.sqrt(n_reps * n_zones):.4f}")
        print("   -- systematically negative beyond that would be the raking to")
        print("      county controls; near zero supports block diagonal")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    area = StudyArea(
        name="covariance diagnostic",
        state=args.state,
        year=args.year,
        counties=tuple(args.county or ()),
        span=args.span,
    )
    print(f"{args.table}, {args.geography}, state {args.state}, "
          f"counties {args.county or 'all'}, {args.year} {args.span}-year")

    frame, estimates, deviations, variances = load(area, args.table, args.geography)
    weight = v.average_weight(area)

    fit = q1_variance_versus_count(estimates, variances, weight)
    q2_zero_cells(estimates, variances, fit, area, args.geography)

    blocks, _, cells = zone_blocks(frame, deviations, variances)
    print(f"\n   aligned {len(cells)} cells across {len(blocks)} zones")
    q3_homogeneity(blocks, rng, args.pairs)
    q4_cross_zone(blocks)


if __name__ == "__main__":
    main()
