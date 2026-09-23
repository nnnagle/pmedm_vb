"""How often do VB posterior draws hand one household an implausible weight?

Part 4 of ``laplace_diagnostic.py`` found the worst VB draws piling tens of
thousands of people onto a single (block group, household) cell. This script
measures how often that happens, who is responsible, and whether the
posterior itself or only the VB family puts mass there. For each fit it draws
``lambda`` from the VB family and forms ``N p(lambda)`` for every cell. It
reports:

- **Concentration.** Quantiles of each draw's largest cell as a share of
  ``N``, and the share of draws whose largest cell exceeds each of three
  thresholds: the largest cell weight at the MAP, ``N max p*`` (for
  reference; ordinary posterior spread exceeds it), 1% of ``N`` and 10% of
  ``N``.
- **Artifact or posterior?** For each draw the log importance ratio
  ``log pi(lambda) - log q(lambda)`` with ``pi ~ exp(-n f)``. Its quantiles
  are compared, relative to the median over all draws, between draws whose
  largest cell is under 1%, 1-10% and over 10% of ``N``. Draws that sit far
  below the rest are ones q overweights -- an artifact of the family; if they
  sit with the rest, the posterior itself allows them. The self-normalised
  importance estimate of each group's posterior probability is printed beside
  q's share, with its ESS; in thousands of dimensions the ESS will be small,
  so the quantile comparison is the robust part.
- **Households.** Every household whose weight ever exceeds 1% of ``N``,
  sorted by its largest weight: size, whether a group-quarters person, draws
  over each threshold, and the categories it carries more than once.

Household size is the sum of the unit's ``B01001`` (sex by age) counts, which
cover every member; it is blank if ``B01001`` is not among the constraints.

``p*`` comes from the MAP multipliers saved with the VB result, so nothing is
re-solved; ``--variance-floor`` must match the VB run, since it sets
``Sigma`` in ``f``. Every (PUMA, taper, alpha) runs in its own process; each
writes ``<out>/<puma>_<taper>_a<alpha>.txt`` and ``..._households.csv``. On
ISAAC use ``weight_draws.sbatch``::

    $CONDA_PREFIX/bin/python experiments/weight_draws.py --variance-floor zero \\
        --alpha 1.0 0.1 --run /lustre/isaac24/proj/UTK0496/pmedm_vb_runs/<jobid>
"""

from __future__ import annotations

import argparse
import contextlib
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
from pmedm_vb.solvers.base import ConstraintOperator

from laplace_diagnostic import batched_f, load_vb, quantiles
from pmedm_vb.solvers.vb import _DualTarget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="VB results directory")
    parser.add_argument("--puma", nargs="*", default=None, help="default: every assembled PUMA")
    parser.add_argument("--alpha", type=float, nargs="+", required=True)
    parser.add_argument("--taper", nargs="+", choices=["tract", "none"], default=["tract"])
    parser.add_argument("--area", default="knox-2024-5yr", help="assembled-inputs directory name")
    parser.add_argument(
        "--variance-floor", default="none",
        help="'none', 'zero' or a number, as in run_map.py; must match the VB run",
    )
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--top", type=int, default=20, help="households listed in the report")
    parser.add_argument("--f-batch", type=int, default=32, help="draws per evaluation of f")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="default: <run>/weight_draws")
    parser.add_argument(
        "--cores", type=int, default=None,
        help="cores to use (default: $SLURM_CPUS_PER_TASK, else all)",
    )
    return parser.parse_args()


def household_profile(inputs: PMEDMInputs) -> tuple[np.ndarray, list[str]]:
    """Each unit's size and the categories it carries more than once.

    Categories are named ``table.category x count``, taken from whichever
    level carries them (tract first), so a table constrained at both levels
    is listed once.
    """
    X = sp.hstack([inputs.X_T, inputs.X_B]).tocsr()
    names = list(inputs.tract_constraints) + list(inputs.bg_constraints)
    seen, keep = set(), []
    for column, name in enumerate(names):
        if name not in seen:
            seen.add(name)
            keep.append(column)
    X, names = X[:, keep], [names[c] for c in keep]

    age_sex = [c for c, name in enumerate(names) if name.startswith("B01001.")]
    size = np.asarray(X[:, age_sex].sum(axis=1)).ravel() if age_sex else np.full(X.shape[0], np.nan)

    bulk = []
    for unit in range(X.shape[0]):
        row = X.getrow(unit)
        pairs = sorted(
            ((names[c], v) for c, v in zip(row.indices, row.data) if v > 1),
            key=lambda pair: -pair[1],
        )
        bulk.append(", ".join(f"{name} x{v:g}" for name, v in pairs))
    return size, bulk


SHARES = (0.01, 0.10)


def parse_floor(value: str):
    """``--variance-floor`` as :meth:`PMEDMInputs.sigma` takes it."""
    if value in (None, "none"):
        return None
    return "zero" if value == "zero" else float(value)


def check(args: argparse.Namespace) -> None:
    """Print the report for one (PUMA, taper, alpha); ``args`` holds scalars here."""
    pmedm_vb.set_verbosity("WARNING")
    rng = np.random.default_rng(args.seed)
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / args.puma)
    path = args.run / f"{args.puma}_{args.taper}_a{args.alpha:g}.npz"
    vb, _, _ = load_vb(path)
    with np.load(path) as saved:
        lam_star = saved["map_lam"]
    taper = None if args.taper == "none" else args.taper
    target = _DualTarget(inputs, inputs.sigma(args.alpha, taper, parse_floor(args.variance_floor)), "cpu")

    op = ConstraintOperator(inputs)
    N, n = inputs.N, inputs.n
    with np.errstate(divide="ignore"):
        log_q = np.log(inputs.q)
    logits = log_q - op.adjoint(lam_star)
    weight_star = N * np.exp(logits - logsumexp(logits))
    thresholds = {"MAP max": weight_star.max(), **{f"{s:.0%} of N": s * N for s in SHARES}}
    n_zones, n_units = weight_star.shape

    largest = np.empty(args.draws)
    log_ratio = np.empty(args.draws)
    over = {name: np.zeros(n_units, dtype=int) for name in thresholds}
    unit_max = np.zeros(n_units)
    done = 0
    while done < args.draws:
        lam = vb.sample(rng, min(args.batch, args.draws - done))
        span = slice(done, done + lam.shape[1])
        log_ratio[span] = -n * batched_f(target, lam, args.f_batch) - vb.log_density(lam)
        for k in range(lam.shape[1]):
            logits = log_q - op.adjoint(lam[:, k])
            unit_weight = (N * np.exp(logits - logsumexp(logits))).max(axis=0)
            largest[done] = unit_weight.max()
            for name, level in thresholds.items():
                over[name] += unit_weight > level
            np.maximum(unit_max, unit_weight, out=unit_max)
            done += 1

    share = largest / N
    print(f"PUMA {args.puma}, taper {args.taper}, alpha {args.alpha}, variance floor "
          f"{args.variance_floor}: {n_zones} zones x {n_units:,} units, N = {N:,.0f}, "
          f"{args.draws:,} VB draws")
    print(f"\n1. Each draw's largest cell (one household record in one block group)")
    print("   share of N: " + "  ".join(
        f"{name} {value:.3%}" for name, value in
        zip(["median", "p90", "p99", "max"], np.percentile(share, [50, 90, 99, 100]))))
    for name, level in thresholds.items():
        above = largest > level
        print(f"   draws with a cell over {name} ({level:,.1f}): {above.sum():,} of {args.draws:,} "
              f"({above.mean():.2%}); households ever over it: {(over[name] > 0).sum():,}")

    print("\n2. log pi(lambda) - log q(lambda), relative to its median over all draws")
    print("   (far below the rest: q overweights those draws; with the rest: pi allows them)")
    centred = log_ratio - np.median(log_ratio)
    weights = np.exp(log_ratio - logsumexp(log_ratio))
    ess = 1.0 / np.sum(weights**2)
    groups = {
        "largest < 1% of N": share < SHARES[0],
        "1% to 10% of N": (share >= SHARES[0]) & (share < SHARES[1]),
        "largest >= 10% of N": share >= SHARES[1],
    }
    for name, member in groups.items():
        if not member.any():
            print(f"   {name:<20} no draws")
            continue
        q10, q50, q90 = np.percentile(centred[member], [10, 50, 90])
        print(f"   {name:<20} {member.sum():>5,} draws  p10 {q10:>10,.1f}  median {q50:>10,.1f}  "
              f"p90 {q90:>10,.1f}   share under q {member.mean():.2%}, "
              f"IS estimate under pi {weights[member].sum():.2%}")
    print(f"   IS effective sample size {ess:,.1f} of {args.draws:,}; the IS shares mean little "
          f"when it is small")

    listed = np.flatnonzero(over[f"{SHARES[0]:.0%} of N"])
    print(f"\n3. Households ever over {SHARES[0]:.0%} of N: {listed.size:,} of {n_units:,}")
    if listed.size == 0:
        return
    size, bulk = household_profile(inputs)
    table = pd.DataFrame({
        "unit": inputs.units["SERIALNO"].to_numpy()[listed],
        "gq": inputs.units["is_group_quarters"].to_numpy()[listed],
        "size": size[listed],
        **{f"draws > {name}": over[name][listed] for name in thresholds},
        "MAP max": weight_star.max(axis=0)[listed],
        "draw max / N": unit_max[listed] / N,
        "carried more than once": [bulk[u] for u in listed],
    }).sort_values("draw max / N", ascending=False)
    csv = args.report.with_name(args.report.stem + "_households.csv")
    table.to_csv(csv, index=False)
    print(f"   all of them in {csv.name}; the {min(args.top, len(table))} with the largest weight:")
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 120)
    print(table.head(args.top).round(4).to_string(index=False))
    if not np.isnan(size).all():
        print(f"\n   size: these households median {np.median(table['size']):g}, "
              f"mean {table['size'].mean():.2f}; all units median {np.median(size):g}, "
              f"mean {size.mean():.2f}")


def check_to_file(args: argparse.Namespace) -> tuple[Path, str]:
    """Worker: one report to ``args.report``. Never raises; a failure is written into it."""
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    with open(args.report, "w") as handle, contextlib.redirect_stdout(handle):
        try:
            check(args)
            return args.report, ""
        except Exception as error:
            print(f"\nFAILED\n{traceback.format_exc()}")
            return args.report, repr(error)


def main() -> None:
    args = parse_args()
    root = processed_dir() / "inputs" / args.area
    pumas = args.puma or sorted(p.name for p in root.glob("*") if (p / "manifest.json").exists())
    out = args.out or (args.run / "weight_draws")
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        argparse.Namespace(**{**vars(args), "puma": puma, "taper": taper, "alpha": alpha,
                              "report": out / f"{puma}_{taper}_a{alpha:g}.txt"})
        for puma in pumas for taper in args.taper for alpha in args.alpha
    ]
    cores = args.cores or int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    workers = min(cores, len(jobs))
    threads = max(1, cores // workers)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = str(threads)
    print(f"{len(jobs)} fits, {workers} workers x {threads} thread(s), reports in {out}", flush=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [executor.submit(check_to_file, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), start=1):
            path, error = future.result()
            print(f"{count} of {len(jobs)}: {path.name}" + (f"  FAILED {error}" if error else ""),
                  flush=True)


if __name__ == "__main__":
    main()
