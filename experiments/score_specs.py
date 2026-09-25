"""Spec files for scoring the whole comparison grid with compare_methods.sbatch.

One spec file per (PUMA, alpha) cell, holding one ``compare_methods.py`` call
per method: the HMC reference first (it caches its own summaries, which the
others reuse), then short HMC, MAP + Laplace, the three VB families and the two
raking baselines (raking has no alpha, so each PUMA's raking fit is scored in
every alpha cell, against that cell's reference). Every call names the cell's
HMC reference (``--reference``); the HMC calls also name the skewed VB fit that
whitened them (``--whiten``). Each cell appends to its own CSV,
``<scores>/<puma>_a<alpha>.csv``, so concurrent jobs never write one file.

CPU fits come from ``--jobs`` CSVs (``job_id, step, puma, alpha, method,
family``; ``grid_cpu_jobs.csv`` and the repository's ``runs.csv``), with the
latest job id winning when a fit appears twice; the results directory is
``<runs>/<job_id>``. HMC runs are ``<hmc>/<puma>_a<alpha>_{ref,short}``, or
``--extra-hmc`` ``PUMA:ALPHA:KIND:DIR``. A method whose files are missing is
left out of the spec and listed on stdout::

    $CONDA_PREFIX/bin/python experiments/score_specs.py --runs $RUNS \\
        --jobs $RUNS/grid_cpu_jobs.csv runs.csv --hmc $RUNS/hmc \\
        --extra-hmc 4701501:0.01:short:$RUNS/6276611/mcmc \\
        --specs $RUNS/scores/specs --scores $RUNS/scores
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ALPHAS = (1.0, 0.1, 0.01)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, nargs="+", required=True)
    parser.add_argument("--hmc", type=Path, required=True)
    parser.add_argument("--extra-hmc", nargs="*", default=[])
    parser.add_argument("--specs", type=Path, required=True, help="directory for the spec files")
    parser.add_argument("--scores", type=Path, required=True, help="directory for the CSVs")
    parser.add_argument("--by-table", action="store_true",
                        help="also write per-table CSVs, <scores>/<puma>_a<alpha>_tables.csv")
    parser.add_argument("--pumas", nargs="+",
                        default=["4701501", "4701502", "4701503", "4701504"])
    return parser.parse_args()


def fit_name(puma: str, alpha: float) -> str:
    return f"{puma}_tract_a{alpha:g}"


def cpu_fits(args) -> dict:
    """``{(method, puma, alpha or None): run directory}``, latest job id winning."""
    frames = [pd.read_csv(path, dtype=str) for path in args.jobs]
    jobs = pd.concat(frames, ignore_index=True)
    jobs = jobs[jobs.step.isin(["solve", "vb", "rake"])].copy()
    jobs["job"] = jobs.job_id.astype(int)
    fits = {}
    for _, job in jobs.sort_values("job").iterrows():
        if job.step == "rake":
            key = (job.method, job.puma, None)
        else:
            method = "map_laplace" if job.step == "solve" else f"vb_{job.family}"
            key = (method, job.puma, float(job.alpha))
        fits[key] = args.runs / job.job_id
    return fits


def hmc_runs(args) -> dict:
    runs = {}
    for directory in args.hmc.glob("*_a*_*"):  # alpha written as 1, 1.0, 0.1, ...
        match = re.match(r"(\d+)_a([\d.]+)_(short|ref)$", directory.name)
        if match and directory.is_dir():
            runs[(match.group(3), match.group(1), float(match.group(2)))] = directory
    for spec in args.extra_hmc:
        puma, alpha, kind, directory = spec.split(":", 3)
        runs[(kind, puma, float(alpha))] = Path(directory)
    return runs


def main() -> None:
    args = parse_args()
    args.specs.mkdir(parents=True, exist_ok=True)
    args.scores.mkdir(parents=True, exist_ok=True)
    fits, hmc = cpu_fits(args), hmc_runs(args)
    missing = []
    for puma in args.pumas:
        for alpha in ALPHAS:
            name = fit_name(puma, alpha)
            out = args.scores / f"{puma}_a{alpha:g}.csv"
            reference = hmc.get(("ref", puma, alpha))
            whiten = fits.get(("vb_skewed", puma, alpha))
            if reference is None or not (reference / f"{name}_trace.npz").exists():
                missing.append(f"{puma} a{alpha:g}: HMC reference (cell skipped)")
                continue
            common = f"--puma {puma} --alpha {alpha:g} --reference {reference} --out {out}"
            if args.by_table:
                common += f" --by-table {args.scores / f'{puma}_a{alpha:g}_tables.csv'}"
            calls = []
            for kind in ("ref", "short"):
                run = hmc.get((kind, puma, alpha))
                if run is None or not (run / f"{name}_trace.npz").exists():
                    missing.append(f"{puma} a{alpha:g}: hmc_{kind}")
                    continue
                extra = f" --whiten {whiten}" if whiten is not None else ""
                calls.append(f"--method hmc_{kind} --run {run} {common}{extra}")
            for method in ("map_laplace", "vb_gaussian", "vb_skewed", "vb_sumdiff"):
                run = fits.get((method, puma, alpha))
                if run is None or not (run / f"{name}.npz").exists():
                    missing.append(f"{puma} a{alpha:g}: {method}")
                    continue
                calls.append(f"--method {method} --run {run} {common}")
            for method in ("ipf", "sinkhorn"):
                run = fits.get((method, puma, None))
                if run is None or not (run / f"{puma}_{method}.npz").exists():
                    missing.append(f"{puma} a{alpha:g}: {method}")
                    continue
                calls.append(f"--method {method} --run {run} {common}")
            spec = args.specs / f"{puma}_a{alpha:g}.txt"
            spec.write_text("\n".join(calls) + "\n")
            print(f"{spec}: {len(calls)} calls")
    print("missing:" if missing else "nothing missing")
    for line in missing:
        print("  " + line)


if __name__ == "__main__":
    main()
