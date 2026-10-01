"""Wall time of every fit in the comparison grid, in one table.

Reads, under ``--runs``:

- the CPU fits listed in ``--jobs`` CSVs (columns ``job_id, step, puma, alpha,
  method, family``; the grid's ``grid_cpu_jobs.csv`` and the repository's
  ``runs.csv`` both qualify): each job's ``summary-<step>-<jobid>.csv``
  (``seconds``, and for VB ``map_seconds``), and the node from the first line
  of its ``slurm-<jobid>.log``;
- every HMC run found under ``--hmc`` (``<puma>_a<alpha>_{short,ref}``) and in
  ``--extra-hmc`` directories: the sampler's own ``seconds`` and
  ``warmup_seconds`` from its trace (summed over resubmissions), and the
  lambda bulk ESS (min, median) from its report, when there is one.

Writes ``<out>/timing.csv``, one row per fit, and ``<out>/timing.txt``: per
method and alpha the median over PUMAs with the range, then the full table.
"VB skewed + short HMC" is added as the sum of its two parts, the proposed
method's cost::

    $CONDA_PREFIX/bin/python experiments/timing_summary.py --runs $RUNS \\
        --jobs $RUNS/grid_cpu_jobs.csv runs.csv --hmc $RUNS/hmc \\
        --extra-hmc 4701501:0.01:short:$RUNS/6276611/mcmc --out $RUNS/timing
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, nargs="+", required=True)
    parser.add_argument("--hmc", type=Path, default=None, help="directory of bundled HMC runs")
    parser.add_argument("--extra-hmc", nargs="*", default=[],
                        help="PUMA:ALPHA:KIND:DIR for HMC runs outside --hmc")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def node_of(runs: Path, job: str) -> tuple[str, float]:
    """``(node, cores)`` from the first line of a job's Slurm log."""
    path = runs / f"slurm-{job}.log"
    if not path.exists():
        return "", np.nan
    first = path.read_text(errors="replace").splitlines()[:1]
    match = re.match(r"node (\S+?),(?: (\d+) cores,)?", first[0]) if first else None
    if not match:
        return "", np.nan
    return match.group(1), float(match.group(2)) if match.group(2) else np.nan


def cpu_rows(args) -> list[dict]:
    rows = []
    for jobs_file in args.jobs:
        jobs = pd.read_csv(jobs_file, dtype=str)
        for _, job in jobs.iterrows():
            step = job.get("step", "")
            if step not in ("solve", "vb", "rake"):
                continue
            summaries = list(args.runs.glob(f"summary-*-{job['job_id']}.csv"))
            if not summaries:
                continue
            summary = pd.read_csv(summaries[0]).iloc[0]
            if step == "solve":
                method = "map"
            elif step == "rake":
                method = str(summary.get("method", job.get("method", "")))
            else:
                method = f"vb_{summary.get('family', job.get('family', ''))}"
            node, cores = node_of(args.runs, job["job_id"])
            alpha = summary.get("alpha", np.nan)
            row = {
                "method": method, "puma": str(summary["puma"]),
                "alpha": float(alpha) if pd.notna(alpha) else np.nan,
                "seconds": float(summary["seconds"]), "job_id": job["job_id"],
                "hardware": f"CPU {cores:.0f} cores" if cores == cores else "CPU",
                "node": node,
                "converged": summary.get("converged", np.nan),
                "iterations": summary.get("n_iter", summary.get("n_sweeps", np.nan)),
            }
            if step == "vb" and "map_seconds" in summary and pd.notna(summary["map_seconds"]):
                row["map_seconds"] = float(summary["map_seconds"])
                row["vb_seconds"] = row["seconds"] - row["map_seconds"]
            rows.append(row)
    return rows


def ess_from_report(directory: Path) -> dict:
    """Bulk ESS from a run_mcmc.py report, if present.

    Lambda ESS (min, median) comes from the thinned kept draws, so thinning
    caps it; the log pi and largest-cell share ESS use every iteration.
    """
    ess = {"ess_bulk_min": np.nan, "ess_bulk_median": np.nan,
           "ess_log_pi": np.nan, "ess_max_share": np.nan}
    for report in directory.glob("*_report.txt"):
        text = report.read_text()
        match = re.search(r"ESS bulk min ([\d,]+), median ([\d,]+)", text)
        if match:
            ess["ess_bulk_min"], ess["ess_bulk_median"] = (
                float(g.replace(",", "")) for g in match.groups())
        for key, label in (("ess_log_pi", "log pi"), ("ess_max_share", "largest-cell share")):
            match = re.search(rf"^\s*{label}\s+R-hat \S+\s+ESS bulk ([\d,]+)", text, re.M)
            if match:
                ess[key] = float(match.group(1).replace(",", ""))
        if ess["ess_log_pi"] == ess["ess_log_pi"]:
            break
    return ess


def hmc_row(puma: str, alpha: float, kind: str, directory: Path) -> dict | None:
    traces = list(directory.glob("*_trace.npz"))
    if not traces:
        return None
    with np.load(traces[0]) as saved:  # lazy: the kept lambda draws are not read
        seconds = float(saved["seconds"]) if "seconds" in saved.files else np.nan
        warmup_seconds = float(saved["warmup_seconds"]) if "warmup_seconds" in saved.files else np.nan
        iterations = int(saved["log_pi"].shape[0])
        warmup = int(saved["warmup"])
        chains = int(saved["log_pi"].shape[1])
        divergent = int(saved["divergent"][warmup:].sum())
    ess = ess_from_report(directory)
    return {
        "method": f"hmc_{kind}", "puma": puma, "alpha": alpha, "seconds": seconds,
        "warmup_seconds": warmup_seconds, "iterations": iterations, "chains": chains,
        "divergent": divergent, **ess,
        "seconds_per_1000_ess_min": 1000 * seconds / ess["ess_bulk_min"],
        "seconds_per_1000_ess_log_pi": 1000 * seconds / ess["ess_log_pi"],
        "hardware": "GPU V100", "directory": str(directory),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = cpu_rows(args)
    if args.hmc is not None:
        for directory in sorted(args.hmc.glob("*_a*_*")):
            match = re.match(r"(\d+)_a([\d.]+)_(short|ref)$", directory.name)
            if match:
                row = hmc_row(match.group(1), float(match.group(2)), match.group(3), directory)
                if row:
                    rows.append(row)
    for spec in args.extra_hmc:
        puma, alpha, kind, directory = spec.split(":", 3)
        row = hmc_row(puma, float(alpha), kind, Path(directory))
        if row:
            rows.append(row)
    frame = pd.DataFrame(rows)

    # The proposed method: VB skewed, then a short HMC run whitened by it.
    skew = frame[frame.method == "vb_skewed"].set_index(["puma", "alpha"])["seconds"]
    short = frame[frame.method == "hmc_short"].set_index(["puma", "alpha"])["seconds"]
    both = pd.concat([skew.rename("vb"), short.rename("hmc")], axis=1).dropna()
    frame = pd.concat([frame, pd.DataFrame({
        "method": "vb_skewed+hmc_short", "puma": [p for p, _ in both.index],
        "alpha": [a for _, a in both.index], "seconds": (both.vb + both.hmc).to_numpy(),
        "hardware": "CPU + GPU V100"})], ignore_index=True)
    frame.to_csv(args.out / "timing.csv", index=False)

    order = ["ipf", "sinkhorn", "map", "vb_gaussian", "vb_skewed", "vb_sumdiff",
             "hmc_short", "vb_skewed+hmc_short", "hmc_ref"]
    frame["alpha_label"] = frame.alpha.map(lambda a: "none" if a != a else f"{a:g}")

    def cell(x):
        x = x.dropna()
        if not len(x):
            return ""
        return f"{np.median(x):,.0f} ({x.min():,.0f}-{x.max():,.0f}) n={len(x)}"

    table = (frame.groupby(["method", "alpha_label"])["seconds"].apply(cell).unstack()
             .reindex([m for m in order if m in set(frame.method)]))
    lines = ["Wall seconds per fit: median over PUMAs (min-max), n fits", "",
             table.to_string(), ""]
    hmc = frame[frame.method.str.startswith("hmc")]
    if len(hmc):
        lines += ["HMC detail:",
                  hmc[["method", "puma", "alpha_label", "seconds", "warmup_seconds", "iterations",
                       "divergent", "ess_bulk_min", "ess_bulk_median", "ess_log_pi",
                       "ess_max_share", "seconds_per_1000_ess_min",
                       "seconds_per_1000_ess_log_pi"]]
                  .sort_values(["method", "alpha_label", "puma"])
                  .to_string(index=False, float_format=lambda v: f"{v:,.1f}"), ""]
    lines += ["All fits:", frame.drop(columns=["alpha_label"]).sort_values(["method", "alpha", "puma"])
              .to_string(index=False, float_format=lambda v: f"{v:,.1f}")]
    text = "\n".join(lines) + "\n"
    (args.out / "timing.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
