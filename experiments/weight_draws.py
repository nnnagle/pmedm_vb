"""How often do VB posterior draws hand one household an implausible weight?

Part 4 of ``laplace_diagnostic.py`` found the worst VB draws piling tens of
thousands of people onto a single (block group, household) cell. This script
measures how often that happens and who is responsible. For each fit it draws
``lambda`` from the VB family, forms ``N p(lambda)`` for every cell, and flags
a cell whose weight exceeds the **largest cell weight at the MAP**,
``N max p*`` -- bigger than anything the point estimate gives any record. It
reports:

- the share of draws with at least one flagged cell, and quantiles of each
  draw's largest cell weight relative to that threshold;
- the households behind the flags: how many draws flag them, their largest
  weight, size, whether a group-quarters person, and the categories they carry
  more than once (the loadings above 1 that make a small move in ``lambda`` a
  large move in their weight).

Household size is the sum of the unit's ``B01001`` (sex by age) counts, which
cover every member; it is blank if ``B01001`` is not among the constraints.

``p*`` comes from the MAP multipliers saved with the VB result, so nothing is
re-solved. Every (PUMA, taper, alpha) runs in its own process; each writes
``<out>/<puma>_<taper>_a<alpha>.txt`` and, for every flagged household,
``..._households.csv``. On ISAAC use ``weight_draws.sbatch``::

    $CONDA_PREFIX/bin/python experiments/weight_draws.py \\
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
from scipy.special import logsumexp

import pmedm_vb
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.solvers.base import ConstraintOperator

from laplace_diagnostic import load_vb, quantiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True, help="VB results directory")
    parser.add_argument("--puma", nargs="*", default=None, help="default: every assembled PUMA")
    parser.add_argument("--alpha", type=float, nargs="+", required=True)
    parser.add_argument("--taper", nargs="+", choices=["tract", "none"], default=["tract"])
    parser.add_argument("--area", default="knox-2024-5yr", help="assembled-inputs directory name")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--top", type=int, default=20, help="households listed in the report")
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


def check(args: argparse.Namespace) -> None:
    """Print the report for one (PUMA, taper, alpha); ``args`` holds scalars here."""
    pmedm_vb.set_verbosity("WARNING")
    rng = np.random.default_rng(args.seed)
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / args.puma)
    path = args.run / f"{args.puma}_{args.taper}_a{args.alpha:g}.npz"
    vb, _, _ = load_vb(path)
    with np.load(path) as saved:
        lam_star = saved["map_lam"]

    op = ConstraintOperator(inputs)
    N = inputs.N
    with np.errstate(divide="ignore"):
        log_q = np.log(inputs.q)
    logits = log_q - op.adjoint(lam_star)
    weight_star = N * np.exp(logits - logsumexp(logits))
    threshold = weight_star.max()
    n_zones, n_units = weight_star.shape

    largest = np.empty(args.draws)
    flagged_cells = np.zeros(args.draws, dtype=int)
    draws_flagged = np.zeros(n_units, dtype=int)
    zones_flagged = np.zeros((n_zones, n_units), dtype=bool)
    unit_max = np.zeros(n_units)
    done = 0
    while done < args.draws:
        lam = vb.sample(rng, min(args.batch, args.draws - done))
        for k in range(lam.shape[1]):
            logits = log_q - op.adjoint(lam[:, k])
            weight = N * np.exp(logits - logsumexp(logits))
            flag = weight > threshold
            largest[done] = weight.max()
            flagged_cells[done] = flag.sum()
            draws_flagged += flag.any(axis=0)
            zones_flagged |= flag
            np.maximum(unit_max, weight.max(axis=0), out=unit_max)
            done += 1

    any_flag = flagged_cells > 0
    print(f"PUMA {args.puma}, taper {args.taper}, alpha {args.alpha}: "
          f"{n_zones} zones x {n_units:,} units, N = {N:,.0f}, {args.draws:,} VB draws")
    print(f"   threshold = largest cell weight at the MAP: {threshold:,.1f} "
          f"({threshold / N:.3%} of N)")
    print(f"\n   draws with at least one cell above it: {any_flag.sum():,} of {args.draws:,} "
          f"({any_flag.mean():.2%})")
    print(f"   cells above it per draw, among those draws: "
          + (f"median {np.median(flagged_cells[any_flag]):.0f}, max {flagged_cells.max()}"
             if any_flag.any() else "none"))
    print(f"   each draw's largest cell / threshold: {quantiles(largest / threshold)}")
    print(f"   each draw's largest cell / N:         "
          + "  ".join(f"{name} {value:.3%}" for name, value in
                      zip(["median", "p90", "p99", "max"],
                          np.percentile(largest / N, [50, 90, 99, 100]))))

    responsible = np.flatnonzero(draws_flagged)
    print(f"\n   households ever above the threshold: {responsible.size:,} of {n_units:,}")
    if responsible.size == 0:
        return
    size, bulk = household_profile(inputs)
    table = pd.DataFrame({
        "unit": inputs.units["SERIALNO"].to_numpy()[responsible],
        "gq": inputs.units["is_group_quarters"].to_numpy()[responsible],
        "size": size[responsible],
        "draws": draws_flagged[responsible],
        "share of draws": draws_flagged[responsible] / args.draws,
        "zones": zones_flagged[:, responsible].sum(axis=0),
        "MAP max": weight_star.max(axis=0)[responsible],
        "draw max": unit_max[responsible],
        "draw max / N": unit_max[responsible] / N,
        "carried more than once": [bulk[u] for u in responsible],
    }).sort_values("draws", ascending=False)
    csv = args.report.with_name(args.report.stem + "_households.csv")
    table.to_csv(csv, index=False)
    print(f"   all of them in {csv.name}; the {min(args.top, len(table))} flagged most often:")
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 120)
    print(table.head(args.top).round(4).to_string(index=False))
    top_share = table["draws"].head(args.top).sum() / draws_flagged.sum()
    print(f"\n   these {min(args.top, len(table))} account for {top_share:.0%} of household-draw flags")
    if not np.isnan(size).all():
        print(f"   size: flagged households median {np.median(table['size']):g}, "
              f"mean {table['size'].mean():.2f}; all units median {np.median(size):g}, "
              f"mean {size.mean():.2f}")


def check_to_file(args: argparse.Namespace) -> tuple[Path, str]:
    """Worker: one report to ``args.report``. Never raises; a failure is written into it."""
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
