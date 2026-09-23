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
   for Laplace draws and, when a VB result is given, for VB draws. VB draws
   are measured against the same Laplace quadratic, so a skewed fit that
   moves mass to the flat side can read negative; the upper tail (p99, max) is
   what says whether its draws still reach a wall.
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
4. **Worst VB draws, decomposed.** With ``delta = lambda - lambda*`` and
   ``z = -X delta`` the change in every (zone, unit) logit, the excess of
   part 1 splits exactly into ``n grad f*' delta`` (near zero at the optimum)
   plus ``n [log E exp(z) - E z - Var(z)/2]``, expectations under ``p*``: the
   log-sum-exp term's departure from its quadratic. It is large when
   ``p(lambda)`` piles onto cells where ``z`` is large, so for each of the
   worst draws the report lists the cells holding most of ``p(lambda)``, then,
   for the top cell, the constraints its logit shift comes from
   (``-loading x delta_k`` for each constraint the cell loads on). If one
   constraint carries nearly all of the shift, the wall is axis-aligned and a
   per-coordinate skew can in principle follow it; if the shift is spread
   over several constraints, it cannot.

Needs torch. Takes a few minutes per fit for an 8,000-constraint PUMA. Several
PUMAs, alphas and tapers may be given; every combination runs in its own
process and writes its report to ``<out>/<puma>_<taper>_a<alpha>.txt``. On
ISAAC use ``laplace_diagnostic.sbatch``, or a compute node from ``srun``, not
the shared login node::

    $CONDA_PREFIX/bin/python experiments/laplace_diagnostic.py \\
        --alpha 1.0 0.1 --taper tract \\
        --run /lustre/isaac24/proj/UTK0496/pmedm_vb_runs/<jobid>
"""

from __future__ import annotations

import argparse
import contextlib
import math
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.special import logsumexp

import pmedm_vb
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.progress import record_run
from pmedm_vb.solvers.base import ConstraintOperator, dual_state
from pmedm_vb.solvers.map_dual import DualHessian, solve_map
from pmedm_vb.solvers.vb import StructuredGaussian, _DualTarget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--puma", nargs="*", default=None, help="default: every assembled PUMA")
    parser.add_argument("--alpha", type=float, nargs="+", required=True)
    parser.add_argument("--taper", nargs="+", choices=["tract", "none"], default=["tract"])
    parser.add_argument("--area", default="knox-2024-5yr", help="assembled-inputs directory name")
    parser.add_argument("--run", type=Path, default=None, help="VB results directory (optional)")
    parser.add_argument(
        "--variance-floor", default="none",
        help="'none', 'zero' or a number, as in run_map.py; must match the VB run",
    )
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--is-draws", type=int, default=4000)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--worst", type=int, default=5, help="VB draws decomposed in part 4")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=None,
        help="report directory (default: <run>/diagnostics, else ./diagnostics)",
    )
    parser.add_argument(
        "--cores", type=int, default=None,
        help="cores to use (default: $SLURM_CPUS_PER_TASK, else all)",
    )
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
    """The fitted VB family and its ELBO, from a run_map.py vb result."""
    with np.load(path) as saved:
        count = sum(1 for key in saved.files if key.startswith("block_"))
        skewed = {key: saved[key] for key in ("skew", "log_tail", "scale") if key in saved.files}
        return StructuredGaussian(
            mean=saved["mean"],
            rows=[saved[f"rows_{i}"] for i in range(count)],
            blocks=[saved[f"block_{i}"] for i in range(count)],
            W=saved["W"],
            V=saved["V"],
            **skewed,
        ), float(saved["elbo"]), float(saved["elbo_se"])


def cell_loadings(inputs: PMEDMInputs, zone: int, unit: int) -> tuple[np.ndarray, np.ndarray]:
    """Stacked-constraint indices a (zone, unit) cell loads on, and the loadings.

    Constraint ``(area a, attribute k)`` sits at ``a + k n_areas`` within its
    level (column-major), and the cell's loading on it is ``A[a, zone] X[unit, k]``.
    """
    index, value, offset = [], [], 0
    for A, X, Y in ((inputs.A_T, inputs.X_T, inputs.Y_T), (inputs.A_B, inputs.X_B, inputs.Y_B)):
        col = sp.csc_matrix(A)[:, [zone]]
        row = sp.csr_matrix(X)[[unit], :]
        areas, attributes = col.indices, row.indices
        index.append(offset + (areas[:, None] + attributes[None, :] * Y.shape[0]).ravel())
        value.append((col.data[:, None] * row.data[None, :]).ravel())
        offset += Y.size
    return np.concatenate(index), np.concatenate(value)


def worst_draws(inputs: PMEDMInputs, op: ConstraintOperator, lam_star: np.ndarray,
                gradient: np.ndarray, lam: np.ndarray, extra: np.ndarray,
                cells: pd.DataFrame, vb: StructuredGaussian, count: int, top: int) -> None:
    """Part 4: split each of the ``count`` worst draws' excess over cells and constraints."""
    n, N = inputs.n, inputs.N
    with np.errstate(divide="ignore"):
        base = np.log(inputs.q) - op.adjoint(lam_star)
    base -= logsumexp(base)
    p_star = np.exp(base)
    zone_ids = inputs.zones.iloc[:, 0].to_numpy()
    unit_ids = inputs.units.iloc[:, 0].to_numpy()
    shares, top_units = [], []
    print(f"\n4. The {count} worst VB draws of part 1, decomposed"
          "\n   excess = n grad' delta + n [log E exp(z) - E z - Var z / 2],  z = -X delta, E under p*")
    for rank, d in enumerate(np.argsort(extra)[::-1][:count], start=1):
        delta = lam[:, d] - lam_star
        z = -op.adjoint(delta)
        shifted = base + z
        log_mgf = logsumexp(shifted)
        p_new = np.exp(shifted - log_mgf)
        mean = float((p_star * z).sum())
        cumulant = n * (log_mgf - mean - 0.5 * (float((p_star * z * z).sum()) - mean**2))
        linear = n * float(gradient @ delta)
        print(f"\n   worst {rank}: excess {extra[d]:,.1f} = cumulant part {cumulant:,.1f} "
              f"+ gradient part {linear:,.1f}  (residual {extra[d] - cumulant - linear:,.2g})")

        order = np.argsort(p_new, axis=None)[::-1][:top]
        zones, units = np.unravel_index(order, p_new.shape)
        print(f"   top {top} cells hold {p_new.ravel()[order].sum():.1%} of p(lambda), "
              f"{p_star.ravel()[order].sum():.2%} of p*")
        print(pd.DataFrame({
            "zone": zone_ids[zones], "unit": unit_ids[units],
            "N p*": N * p_star[zones, units], "N p(lambda)": N * p_new[zones, units],
            "z": z[zones, units],
        }).round(3).to_string(index=False))

        index, loading = cell_loadings(inputs, zones[0], units[0])
        contribution = -loading * delta[index]
        positive = contribution[contribution > 0].sum()
        share = contribution.max() / positive if positive > 0 else float("nan")
        shares.append(share)
        top_units.append(unit_ids[units[0]])
        table = cells.iloc[index][["level", "geoid", "constraint", "published", "se", "fitted"]].copy()
        table["loading"] = loading
        table["delta/sd"] = delta[index] / cells["sd"].to_numpy()[index]
        if vb.is_skewed:
            table["skew"] = vb.skew[index]
            table["log_tail"] = vb.log_tail[index]
        table["contribution"] = contribution
        print(f"   top cell's logit shift z = {contribution.sum():.3f} over {index.size} constraints; "
              f"the largest carries {share:.0%} of the positive part")
        print(table.sort_values("contribution", ascending=False).head(6)
              .round(3).to_string(index=False))

    print(f"\n   across the {count} draws: largest single constraint's share of the top cell's "
          f"positive shift: median {np.nanmedian(shares):.0%}, range "
          f"{np.nanmin(shares):.0%} to {np.nanmax(shares):.0%}; "
          f"{len(set(top_units))} distinct top units")


def quantiles(x: np.ndarray) -> str:
    qs = np.percentile(x, [50, 90, 99, 100])
    return "  ".join(f"{name} {value:>12,.1f}" for name, value in zip(["median", "p90", "p99", "max"], qs))


def diagnose(args: argparse.Namespace) -> None:
    """Print the report for one (PUMA, taper, alpha); ``args`` holds scalars here."""
    pmedm_vb.set_verbosity("WARNING")
    rng = np.random.default_rng(args.seed)
    taper = None if args.taper == "none" else args.taper
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / args.puma)
    n, m = inputs.n, inputs.n_constraints
    floor = {"none": None, "zero": "zero"}.get(args.variance_floor)
    if floor is None and args.variance_floor not in ("none", None):
        floor = float(args.variance_floor)
    sigma = inputs.sigma(args.alpha, taper, floor)

    result = solve_map(inputs, alpha=args.alpha, taper=taper, variance_floor=floor)
    f_star = result.objective
    laplace = StructuredGaussian.laplace(inputs, result)
    target = _DualTarget(inputs, sigma, "cpu")
    print(f"PUMA {args.puma}, taper {args.taper}, alpha {args.alpha}, "
          f"variance floor {args.variance_floor}: m = {m:,}, n = {n:,}, f* = {f_star:.6f}")
    if floor is not None:
        raised = sigma.v > inputs.sigma_v
        print(f"   floor raised {raised.sum():,} of {m:,} variances; published values of those "
              f"cells: median {np.median(inputs.targets()[raised]):.0f}, "
              f"max {inputs.targets()[raised].max():.0f}")

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
            print(f"   VB draws ({'skewed' if vb.is_skewed else 'Gaussian'} family), same quantity: {quantiles(extra_vb)}")
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
        log_q = vb.log_density(lam_is)
        log_w = -n * batched_f(target, lam_is, args.batch) - log_q
        estimate = logsumexp(log_w) - math.log(args.is_draws)
        ess = math.exp(2 * logsumexp(log_w) - logsumexp(2 * log_w))
        print(f"   importance sampling (VB proposal): {estimate:,.1f}   ESS {ess:,.0f} of {args.is_draws:,}")
        print(f"   VB ELBO (from the run):            {vb_elbo:,.1f} +- {vb_elbo_se:,.1f}")
        verdict = "OK: ELBO below the normaliser" if vb_elbo <= estimate + 3 * vb_elbo_se \
            else "PROBLEM: ELBO above the normaliser estimate"
        print(f"   {verdict}. The IS estimate is biased low when ESS is small, so a "
              f"violation is only conclusive with a healthy ESS.")

    # -- 4. worst VB draws, decomposed ---------------------------------------
    if vb is not None and args.worst > 0:
        worst_draws(inputs, ConstraintOperator(inputs), result.lam, sigma_state.gradient,
                    lam_vb, extra_vb, cells, vb, args.worst, 5)


def diagnose_to_file(args: argparse.Namespace, path: Path) -> tuple[Path, str]:
    """Worker: one report to ``path``. Never raises; a failure is written into it."""
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    with open(path, "w") as handle, contextlib.redirect_stdout(handle):
        try:
            diagnose(args)
            return path, ""
        except Exception as error:
            print(f"\nFAILED\n{traceback.format_exc()}")
            return path, repr(error)


def main() -> None:
    args = parse_args()
    root = processed_dir() / "inputs" / args.area
    pumas = args.puma or sorted(p.name for p in root.glob("*") if (p / "manifest.json").exists())
    out = args.out or ((args.run / "diagnostics") if args.run else Path("diagnostics"))
    out.mkdir(parents=True, exist_ok=True)
    record_run(out, args)
    jobs = [
        (argparse.Namespace(**{**vars(args), "puma": puma, "taper": taper, "alpha": alpha}),
         out / f"{puma}_{taper}_a{alpha:g}.txt")
        for puma in pumas for taper in args.taper for alpha in args.alpha
    ]
    cores = args.cores or int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    workers = min(cores, len(jobs))
    threads = max(1, cores // workers)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(threads)
    print(f"{len(jobs)} diagnostics, {workers} workers x {threads} thread(s), reports in {out}",
          flush=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [executor.submit(diagnose_to_file, job, path) for job, path in jobs]
        for count, future in enumerate(as_completed(futures), start=1):
            path, error = future.result()
            print(f"{count} of {len(jobs)}: {path.name}" + (f"  FAILED {error}" if error else ""),
                  flush=True)


if __name__ == "__main__":
    main()
