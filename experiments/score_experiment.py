"""Score one experiment's fits into ``<exp>/scores/<tag>``, a new folder per scoring.

Scoring is kept apart from the fits so it can be redone as the scoring code
changes, without touching them: each run of this script writes a new tag
(``s01``, ``s02``, ...) and refuses one that exists::

    <exp>/scores/<tag>/
      scoring.json             commit, date, options, the jobs
      specs/<puma>_a<alpha>.txt   one compare_methods.py call per method
      per_cell/<puma>_a<alpha>.csv, <puma>_a<alpha>_tables.csv
      logs/
      all_scores.csv, all_tables.csv     (the combine step)
      score_summary.txt, table_summary.txt, distribution_summary.txt,
      calibration_summary.txt, missing.txt

Usage, from the repository root on a login node::

    $CONDA_PREFIX/bin/python experiments/score_experiment.py <root>/exp01_baseline --tag s01

One ``compare_methods.sbatch`` job per (PUMA, alpha) cell, each calling
``compare_methods.py`` for the cell's HMC reference first (it caches its
summaries for the rest), then short HMC, MAP + Laplace, the VB families and
the raking fits (raking has no alpha: each PUMA's raking is scored in every
alpha cell, against that cell's reference). Every call names the cell's
reference; the HMC calls also name the skewed VB fit that whitened them. Then
one ``run_python.sbatch`` job, after all of them (``afterany``), runs this
script with ``--combine``: it concatenates the cells' CSVs and writes the
``score_summary.py`` views.

A method whose results are missing is listed and, with ``--allow-missing``,
left out (a cell without its HMC reference is left out whole); without it
nothing is submitted. ``--dry-run`` prints the jobs and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import (  # noqa: E402
    EXPERIMENTS, area_slug, data_dir, fit_name, git_commit, model_args, rake_output,
)

SCORE_SLURM: list[str] = []
COMBINE_SLURM = ["--cpus-per-task=4", "--mem=32G", "--time=01:00:00"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", type=Path, help="the experiment's folder (<root>/<name>)")
    parser.add_argument("--tag", required=True, help="the scoring's folder name, e.g. s01")
    parser.add_argument("--draws", type=int, default=4000, help="compare_methods.py --draws")
    parser.add_argument("--allow-missing", action="store_true",
                        help="score what exists, leaving out missing methods")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--combine", action="store_true",
                        help="(the final job) combine the cells' CSVs and write the summaries")
    return parser.parse_args()


def cell_calls(spec: dict, exp: Path, puma: str, alpha: float, out: Path, draws: int
               ) -> tuple[list[str], list[str]]:
    """The cell's compare_methods.py argument lines and what is missing."""
    name = fit_name(spec, puma, alpha)
    cell = f"{puma}_a{alpha:g}"
    reference = exp / puma / "hmc_ref"
    whiten = exp / puma / "vb_skewed"
    if not (reference / f"{name}_trace.npz").exists():
        return [], [f"{cell}: hmc_ref (cell left out)"]
    common = ["--puma", puma, "--alpha", f"{alpha:g}", *model_args(spec),
              "--area", area_slug(spec["area"]), "--draws", str(draws),
              "--reference", str(reference),
              "--out", str(out / "per_cell" / f"{cell}.csv"),
              "--by-table", str(out / "per_cell" / f"{cell}_tables.csv")]
    calls, missing = [], []
    for kind in ("ref", "short"):
        run = exp / puma / f"hmc_{kind}"
        if (run / f"{name}_trace.npz").exists():
            calls.append(["--method", f"hmc_{kind}", "--run", str(run), *common,
                          "--whiten", str(whiten)])
        else:
            missing.append(f"{cell}: hmc_{kind}")
    fits = [("map_laplace", "map")] + [(f"vb_{f}", f"vb_{f}") for f in spec["vb_families"]]
    for method, folder in fits:
        run = exp / puma / folder
        if (run / f"{name}.npz").exists():
            calls.append(["--method", method, "--run", str(run), *common])
        else:
            missing.append(f"{cell}: {method}")
    for method in spec["rake"]:
        output = rake_output(spec, exp, puma, method)
        if output.exists():
            calls.append(["--method", method, "--run", str(output.parent), *common])
        else:
            missing.append(f"{cell}: {method}")
    return [" ".join(call) for call in calls], missing


def sbatch(command: list[str], dry_run: bool) -> str:
    if dry_run:
        print("  " + " ".join(command))
        return "DRYRUN"
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"sbatch failed: {result.stderr.strip()}\n{' '.join(command)}")
    return result.stdout.strip().split(";")[0]


def submit(args: argparse.Namespace) -> None:
    exp = args.experiment.resolve()
    spec = json.loads((exp / "experiment.json").read_text())
    out = exp / "scores" / args.tag
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new tag")

    cells, missing = {}, []
    for puma in spec["pumas"]:
        for alpha in spec["alphas"]:
            calls, lacking = cell_calls(spec, exp, puma, alpha, out, args.draws)
            missing += lacking
            if calls:
                cells[f"{puma}_a{alpha:g}"] = calls
    if missing:
        print("missing:\n  " + "\n  ".join(missing))
        if not args.allow_missing:
            raise SystemExit("nothing submitted; finish the fits, or pass --allow-missing")
    if not cells:
        raise SystemExit("no cell has its HMC reference; nothing to score")

    export = f"--export=ALL,PMEDM_VB_DATA={data_dir(spec)}"
    log = f"--output={out / 'logs' / 'slurm-%j.log'}"
    if not args.dry_run:
        (out / "specs").mkdir(parents=True)
        (out / "per_cell").mkdir()
        (out / "logs").mkdir()
    jobs = {}
    for cell, calls in cells.items():
        spec_file = out / "specs" / f"{cell}.txt"
        if args.dry_run:
            print(f"  would write {spec_file}: {len(calls)} calls")
        else:
            spec_file.write_text("\n".join(calls) + "\n")
        jobs[cell] = sbatch(["sbatch", "--parsable", export, log,
                             f"--job-name={spec['name']}-{args.tag}-{cell}", *SCORE_SLURM,
                             str(EXPERIMENTS / "compare_methods.sbatch"), str(spec_file)],
                            args.dry_run)
    combine = sbatch(["sbatch", "--parsable", export, log,
                      f"--job-name={spec['name']}-{args.tag}-combine", *COMBINE_SLURM,
                      f"--dependency=afterany:{':'.join(jobs.values())}",
                      str(EXPERIMENTS / "run_python.sbatch"), "score_experiment.py", str(exp),
                      "--tag", args.tag, "--combine"], args.dry_run)
    record = {"tag": args.tag, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "commit": git_commit(), "draws": args.draws, "experiment": str(exp),
              "level": spec.get("_level"), "missing": missing, "jobs": jobs,
              "combine_job": combine}
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    (out / "scoring.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{len(jobs)} scoring job(s) and the combine job {combine} submitted; results in {out}")


def combine(args: argparse.Namespace) -> None:
    import pandas as pd

    exp = args.experiment.resolve()
    spec = json.loads((exp / "experiment.json").read_text())
    out = exp / "scores" / args.tag
    record = json.loads((out / "scoring.json").read_text())
    files = sorted((out / "per_cell").glob("*.csv"))
    tables = [path for path in files if path.name.endswith("_tables.csv")]
    scores = [path for path in files if not path.name.endswith("_tables.csv")]
    if not scores:
        raise SystemExit(f"no scores in {out / 'per_cell'}")
    frame = pd.concat([pd.read_csv(path, dtype={"puma": str}) for path in scores],
                      ignore_index=True)
    frame.to_csv(out / "all_scores.csv", index=False)
    if tables:
        pd.concat([pd.read_csv(path, dtype={"puma": str}) for path in tables],
                  ignore_index=True).to_csv(out / "all_tables.csv", index=False)

    # Every (method, PUMA, alpha) the scoring was meant to have, against what it has.
    methods = (["hmc_ref", "hmc_short", "map_laplace"]
               + [f"vb_{f}" for f in spec["vb_families"]] + list(spec["rake"]))
    have = set(zip(frame["method"], frame["puma"], frame["alpha"].round(12)))
    lines = list(record.get("missing", []))
    for puma in spec["pumas"]:
        for alpha in spec["alphas"]:
            for method in methods:
                if (method, puma, round(float(alpha), 12)) not in have:
                    lines.append(f"{puma}_a{alpha:g}: {method} (no scores)")
    (out / "missing.txt").write_text("\n".join(lines) + "\n" if lines else "nothing missing\n")

    summary = EXPERIMENTS / "score_summary.py"
    subprocess.run([sys.executable, str(summary), "--scores", str(out / "all_scores.csv"),
                    "--out", str(out)], check=True)
    if tables:
        subprocess.run([sys.executable, str(summary), "--by-table", str(out / "all_tables.csv"),
                        "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
    record["combined"] = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": git_commit(),
                          "cells": len(scores), "missing": len(lines)}
    (out / "scoring.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"combined {len(scores)} cell(s) into {out}; {len(lines)} missing (missing.txt)")


def main() -> None:
    args = parse_args()
    combine(args) if args.combine else submit(args)


if __name__ == "__main__":
    main()
