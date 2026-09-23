"""Where does the Laplace approximation over lambda go wrong, and is VB's ELBO sound?

The VB sweep on Knox found VB beating its own Laplace start by tens of
thousands of nats, with Laplace ELBOs whose Monte Carlo standard errors run to
thousands. That says some Laplace draws land where the posterior
``pi(lambda) ~ exp(-n f(lambda))`` is vanishingly small. This script, for one
fit, measures three things:

1. **Per-draw excess.** For a draw ``lambda = lambda* + G'^{-1} eps`` from the
   Laplace approximation, a Gaussian posterior would give
   ``n (f(lambda) - f*) = |eps|^2 / 2`` exactly. The excess over that is how far
   the posterior falls below the Gaussian at that draw. Quantiles are printed
   for Laplace draws and, when a VB result is given, for VB draws.
2. **Coordinate scan.** Each multiplier is moved alone to
   ``lambda* +- 2 sd_k``, with ``sd_k`` its Laplace marginal standard deviation,
   and the rise in ``n f`` is compared with the quadratic prediction
   ``2 sd_k^2 (n H)_kk``. The largest excesses are listed with the constraint
   they belong to -- level, area, ``table.category``, published value and SE,
   and the MAP fitted value -- and which side (``+`` or ``-``) is steep. If the
   hypothesis is right they are small- or zero-count cells with one steep side.
3. **ELBO bound.** ``log integral exp(-n f)`` is estimated by importance sampling
   with the VB Gaussian as proposal; the VB ELBO must not exceed it (beyond
   noise). The effective sample size says how far to trust the estimate, and
   in thousands of dimensions it will usually say not at all: even on a
   36-dimensional synthetic problem it was 25 of 20,000, and importance
   sampling degrades rapidly with dimension. Expect this part to be
   inconclusive on real PUMAs; parts 1 and 2 are the evidence. The Laplace
   formula ``-n f* + (m/2) log 2 pi - (1/2) log det(n H)`` is printed beside it
   for scale; it is an approximation, not a bound.

Needs torch. Takes a few minutes for an 8,000-constraint PUMA; on ISAAC run it
on a compute node (``srun``) rather than the shared login node::

    $CONDA_PREFIX/bin/python experiments/laplace_diagnostic.py \\
        --puma 4701501 --taper tract --alpha 1.0 \\
        --run /lustre/isaac24/proj/UTK0496/pmedm_vb_runs/<jobid>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

import pmedm_vb
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.solvers.base import ConstraintOperator, dual_state
from pmedm_vb.solvers.map_dual import DualHessian, solve_map
from pmedm_vb.solvers.vb import StructuredGaussian, _DualTarget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--puma", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--taper", choices=["tract", "none"], default="tract")
    parser.add_argument("--area", default="knox-2024-5yr", help="assembled-inputs directory name")
    parser.add_argument("--run", type=Path, default=None, help="VB results directory (optional)")
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--is-draws", type=int, default=4000)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def constraint_table(inputs: PMEDMInputs) -> pd.DataFrame:
    """One row per stacked constraint: level, area, name, published value and SE."""
    n_t, n_b = len(inputs.tracts), len(inputs.block_groups)
    return pd.DataFrame({
        "level": ["tract"] * inputs.Y_T.size + ["bg"] * inputs.Y_B.size,
        "geoid": list(inputs.tracts.iloc[:, 0]) * len(inputs.tract_constraints)
                 + list(inputs.block_groups.iloc[:, 0]) * len(inputs.bg_constraints),
        "constraint": [c for c in inputs.tract_constraints for _ in range(n_t)]
                      + [c for c in inputs.bg_constraints for _ in range(n_b)],
        "published": inputs.targets(),
        "se": np.sqrt(inputs.sigma_v),
    })


def coordinate_rise(target: _DualTarget, centre: np.ndarray, step: np.ndarray, batch: int) -> np.ndarray:
    """``f(centre + step_k e_k)`` for every k, built a batch at a time."""
    m, out = centre.size, []
    with torch.no_grad():
        for start in range(0, m, batch):
            index = np.arange(start, min(start + batch, m))
            chunk = np.tile(centre, (index.size, 1))
            chunk[np.arange(index.size), index] += step[index]
            out.append(target(torch.as_tensor(chunk, dtype=torch.float64)).numpy())
    return np.concatenate(out)


def batched_f(target: _DualTarget, lam: np.ndarray, batch: int) -> np.ndarray:
    """``f`` at each column of ``lam`` (m x k)."""
    out = []
    with torch.no_grad():
        for start in range(0, lam.shape[1], batch):
            chunk = torch.as_tensor(lam[:, start:start + batch].T.copy(), dtype=torch.float64)
            out.append(target(chunk).numpy())
    return np.concatenate(out)


def load_vb(path: Path) -> tuple[StructuredGaussian, float, float]:
    """The fitted VB Gaussian and its ELBO, from a run_map.py vb result."""
    with np.load(path) as saved:
        count = sum(1 for key in saved.files if key.startswith("block_"))
        return StructuredGaussian(
            mean=saved["mean"],
            rows=[saved[f"rows_{i}"] for i in range(count)],
            blocks=[saved[f"block_{i}"] for i in range(count)],
            W=saved["W"],
            V=saved["V"],
        ), float(saved["elbo"]), float(saved["elbo_se"])


def quantiles(x: np.ndarray) -> str:
    qs = np.percentile(x, [50, 90, 99, 100])
    return "  ".join(f"{name} {value:>12,.1f}" for name, value in zip(["median", "p90", "p99", "max"], qs))


def main() -> None:
    args = parse_args()
    pmedm_vb.set_verbosity("WARNING")
    rng = np.random.default_rng(args.seed)
    taper = None if args.taper == "none" else args.taper
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / args.puma)
    n, m = inputs.n, inputs.n_constraints
    sigma = inputs.sigma(args.alpha, taper)

    result = solve_map(inputs, alpha=args.alpha, taper=taper)
    f_star = result.objective
    laplace = StructuredGaussian.laplace(inputs, result)
    target = _DualTarget(inputs, sigma, "cpu")
    print(f"PUMA {args.puma}, taper {args.taper}, alpha {args.alpha}: m = {m:,}, n = {n:,}, "
          f"f* = {f_star:.6f}")

    # -- 1. per-draw excess -------------------------------------------------
    lam = laplace.sample(rng, args.draws)
    eps_sq = np.einsum("id,id->d", lam - result.lam[:, None],
                       laplace.precision_matvec(lam - result.lam[:, None]))
    rise = n * (batched_f(target, lam, args.batch) - f_star)
    extra = rise - 0.5 * eps_sq
    print(f"\n1. Laplace draws ({args.draws}): n(f - f*) - |eps|^2/2  (0 if the posterior were Gaussian)")
    print(f"   {quantiles(extra)}")
    print(f"   |eps|^2/2 for reference: median {np.median(0.5 * eps_sq):,.1f} (m/2 = {m / 2:,.1f})")

    vb = None
    if args.run is not None:
        vb_path = args.run / f"{args.puma}_{args.taper}_a{args.alpha:g}.npz"
        if vb_path.exists():
            vb, vb_elbo, vb_elbo_se = load_vb(vb_path)
            lam_vb = vb.sample(rng, args.draws)
            d = lam_vb - result.lam[:, None]
            rise_vb = n * (batched_f(target, lam_vb, args.batch) - f_star)
            extra_vb = rise_vb - 0.5 * np.einsum("id,id->d", d, laplace.precision_matvec(d))
            print(f"   VB draws, same quantity: {quantiles(extra_vb)}")
            print(f"   VB mean shift from lambda*, in Laplace metric: "
                  f"{math.sqrt(float((vb.mean - result.lam) @ laplace.precision_matvec(vb.mean - result.lam))):.2f}")
        else:
            print(f"   (no VB result at {vb_path})")

    # -- 2. coordinate scan ---------------------------------------------------
    sd = lam.std(axis=1, ddof=1)
    sigma_state = dual_state(inputs, result.lam, sigma)
    hessian = DualHessian(inputs, sigma_state, sigma)
    diag_nh = np.empty(m)
    for rows, block in zip(hessian.rows, hessian.blocks):
        diag_nh[rows] = n * np.diag(block)
    diag_nh += n * (hessian.w**2 @ hessian.signs)
    rows_out = []
    for sign in (+1, -1):
        rise_k = n * (coordinate_rise(target, result.lam, sign * 2 * sd, args.batch) - f_star)
        rows_out.append(rise_k - 0.5 * (2 * sd) ** 2 * diag_nh)
    plus, minus = rows_out
    cells = constraint_table(inputs)
    cells["fitted"] = ConstraintOperator(inputs).forward(result.W)
    cells["lambda*"] = result.lam
    cells["sd"] = sd
    cells["excess +2sd"] = plus
    cells["excess -2sd"] = minus
    cells["worst"] = np.maximum(plus, minus)
    cells["steep side"] = np.where(plus >= minus, "+", "-")
    top = cells.sort_values("worst", ascending=False).head(args.top)
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 40)
    print("\n2. Coordinate scan: rise in n f at lambda* +- 2 sd beyond the quadratic prediction")
    print(f"   excess over all {m:,} constraints: {quantiles(cells['worst'].to_numpy())}")
    print(f"   {args.top} largest:")
    print(top[["level", "geoid", "constraint", "published", "se", "fitted", "lambda*", "sd",
               "excess +2sd", "excess -2sd", "steep side"]].round(3).to_string(index=False))
    small = cells["published"] <= 2 * cells["se"]
    print(f"\n   share of cells with published <= 2 SE: all {small.mean():.1%}, "
          f"top {args.top}: {small[top.index].mean():.1%}, "
          f"published == 0: all {(cells['published'] == 0).mean():.1%}, "
          f"top {args.top}: {(cells.loc[top.index, 'published'] == 0).mean():.1%}")

    # -- 3. ELBO against an importance-sampled normaliser ---------------------
    logdet_nh = laplace.logdet_precision()
    laplace_normaliser = -n * f_star + 0.5 * m * math.log(2 * math.pi) - 0.5 * logdet_nh
    print("\n3. log integral exp(-n f)")
    print(f"   Laplace formula:                  {laplace_normaliser:,.1f}")
    if vb is not None:
        lam_is = vb.sample(rng, args.is_draws)
        d = lam_is - vb.mean[:, None]
        log_q = (-0.5 * m * math.log(2 * math.pi) + 0.5 * vb.logdet_precision()
                 - 0.5 * np.einsum("id,id->d", d, vb.precision_matvec(d)))
        log_w = -n * batched_f(target, lam_is, args.batch) - log_q
        estimate = logsumexp(log_w) - math.log(args.is_draws)
        ess = math.exp(2 * logsumexp(log_w) - logsumexp(2 * log_w))
        print(f"   importance sampling (VB proposal): {estimate:,.1f}   ESS {ess:,.0f} of {args.is_draws:,}")
        print(f"   VB ELBO (from the run):            {vb_elbo:,.1f} +- {vb_elbo_se:,.1f}")
        verdict = "OK: ELBO below the normaliser" if vb_elbo <= estimate + 3 * vb_elbo_se \
            else "PROBLEM: ELBO above the normaliser estimate"
        print(f"   {verdict}. The IS estimate is biased low when ESS is small, so a "
              f"violation is only conclusive with a healthy ESS.")


if __name__ == "__main__":
    main()
