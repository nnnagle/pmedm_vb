"""Score one experiment's fits into ``<exp>/scores/<tag>``, a new folder per scoring.

Scoring is kept apart from the fits so it can be redone as the scoring code
changes, without touching them: each run of this script writes a new tag
(``s01``, ``s02``, ...) and refuses one that exists::

    <exp>/scores/<tag>/
      scoring.json             commit, date, options, the jobs
      specs/<puma>_a<alpha>.txt   one compare_methods.py call per method
      per_cell/<puma>_a<alpha>.csv, <puma>_a<alpha>_tables.csv
      detail/<puma>_a<alpha>/<method>_cells.parquet, <method>_draws.parquet
                               every cell's and every draw's values, for exact pooling
      logs/
      all_scores.csv, all_tables.csv     (the combine step)
      score_summary.txt, table_summary.txt, distribution_summary.txt,
      calibration_summary.txt, missing.txt

Usage, from the repository root on a login node::

    $CONDA_PREFIX/bin/python experiments/score_experiment.py <root>/exp01_baseline --tag s01

One ``compare_methods.sbatch`` job per (PUMA, alpha) cell, each calling
``compare_methods.py`` for the cell's HMC reference first (it caches its
summaries for the rest), then short HMC, MAP + Laplace, the VB families and
the VB families again truncated (``vb_<family>_trunc``), and
the raking fits (raking has no alpha: each PUMA's raking is scored in every
alpha cell, against that cell's reference). Every call names the cell's
reference; the HMC calls also name the skewed VB fit that whitened them. Then
one ``run_python.sbatch`` job, after all of them (``afterany``), runs this
script with ``--combine``: it concatenates the cells' CSVs and writes the
``score_summary.py`` views.

A method whose results are missing is listed and, with ``--allow-missing``,
left out (a cell without its HMC reference is left out whole); without it
nothing is submitted. ``--dry-run`` prints the jobs and writes nothing.

**Resuming.** ``--resume`` takes a tag that exists and submits only the calls
that have no scores yet -- a method counts as scored once its rows are in the
cell's ``per_cell`` CSV -- skipping cells whose job is still queued or running.
So a submission cut short (a queue limit, a failed job) is finished by calling
again with ``--resume``, without scoring anything twice. ``scoring.json`` is
written before the first job is submitted and after each one. ``--max-jobs``
caps the jobs one call submits; ``--no-combine`` leaves the combine step to the
caller (``run_paper.py``).
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

#: The short partition, capped at 1 hour, allocates quickly.
SCORE_SLURM = ["--partition=short", "--qos=short", "--time=01:00:00"]
COMBINE_SLURM = ["--partition=short", "--qos=short", "--cpus-per-task=4", "--mem=32G",
                 "--time=01:00:00"]


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
    parser.add_argument("--resume", action="store_true",
                        help="the tag exists: submit only the calls without scores")
    parser.add_argument("--max-jobs", type=int, default=None,
                        help="submit at most this many scoring jobs in this call")
    parser.add_argument("--no-combine", action="store_true",
                        help="do not submit the combine job")
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
              "--by-table", str(out / "per_cell" / f"{cell}_tables.csv"),
              "--detail", str(out / "detail" / cell)]
    calls, missing = [], []
    for kind in ("ref", "short"):
        run = exp / puma / f"hmc_{kind}"
        if (run / f"{name}_trace.npz").exists():
            calls.append(["--method", f"hmc_{kind}", "--run", str(run), *common,
                          "--whiten", str(whiten)])
        else:
            missing.append(f"{cell}: hmc_{kind}")
    # Each VB family is scored as fitted and truncated (vb_<family>_trunc: the same
    # fit, with the draws the HMC reference never makes turned away).
    fits = ([("map_laplace", "map")] + [(f"vb_{f}", f"vb_{f}") for f in spec["vb_families"]]
            + [(f"vb_{f}_trunc", f"vb_{f}") for f in spec["vb_families"]])
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


def method_of(call: str) -> str:
    """The ``--method`` of one compare_methods.py argument line."""
    tokens = call.split()
    return tokens[tokens.index("--method") + 1]


def scored_methods(out: Path, cell: str) -> set[str]:
    """The methods whose scores are in the cell's CSV."""
    import pandas as pd

    path = out / "per_cell" / f"{cell}.csv"
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["method"]).method.unique())


def live_jobs(job_ids) -> set[str]:
    """Those of ``job_ids`` still queued or running."""
    ids = [j for j in job_ids if j and not str(j).startswith("DRYRUN")]
    if not ids:
        return set()
    try:
        result = subprocess.run(["squeue", "-h", "-o", "%i", "-j", ",".join(ids)],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return set(ids)  # unknown: assume live, so nothing is submitted twice
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def plan(spec: dict, exp: Path, out: Path, draws: int) -> tuple[dict[str, list[str]], list[str]]:
    """Every cell's compare_methods.py calls, and what is missing to score."""
    cells, missing = {}, []
    for puma in spec["pumas"]:
        for alpha in spec["alphas"]:
            calls, lacking = cell_calls(spec, exp, puma, alpha, out, draws)
            missing += lacking
            if calls:
                cells[f"{puma}_a{alpha:g}"] = calls
    return cells, missing


def scoring_state(exp: Path, tag: str) -> dict:
    """Where a scoring stands: ``calls`` (the calls without scores, by cell),
    ``live`` (scoring or combine jobs queued or running), ``combined`` (whether
    ``all_scores.csv`` is newer than every cell's CSV) and ``exists``."""
    out = exp / "scores" / tag
    record_path = out / "scoring.json"
    if not record_path.exists():
        return dict(exists=False, calls={}, live=set(), combined=False)
    record = json.loads(record_path.read_text())
    spec = json.loads((exp / "experiment.json").read_text())
    cells, _ = plan(spec, exp, out, record["draws"])
    calls = {}
    for cell, lines in cells.items():
        done = scored_methods(out, cell)
        left = [line for line in lines if method_of(line) not in done]
        if left:
            calls[cell] = left
    jobs = [*record.get("jobs", {}).values(), record.get("combine_job")]
    live = live_jobs(jobs)
    csvs = list((out / "per_cell").glob("*.csv"))
    summary = out / "all_scores.csv"
    combined = summary.exists() and all(p.stat().st_mtime <= summary.stat().st_mtime for p in csvs)
    return dict(exists=True, calls=calls, live=live, combined=combined)


def write_record(out: Path, record: dict) -> None:
    tmp = out / "scoring.json.tmp"
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(out / "scoring.json")


def submit(args: argparse.Namespace) -> None:
    exp = args.experiment.resolve()
    spec = json.loads((exp / "experiment.json").read_text())
    out = exp / "scores" / args.tag
    if out.exists() and not args.resume:
        raise SystemExit(f"{out} exists; pick a new tag, or pass --resume to submit what has "
                         "no scores yet")

    cells, missing = plan(spec, exp, out, args.draws)
    if missing:
        print("missing:\n  " + "\n  ".join(missing))
        if not args.allow_missing:
            raise SystemExit("nothing submitted; finish the fits, or pass --allow-missing")
    if not cells:
        raise SystemExit("no cell has its HMC reference; nothing to score")

    record_path = out / "scoring.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["draws"] != args.draws:
            raise SystemExit(f"{out} was scored with --draws {record['draws']}, not {args.draws}")
        record.setdefault("history", []).append(
            {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": git_commit()})
    else:
        record = {"tag": args.tag, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "commit": git_commit(), "draws": args.draws, "experiment": str(exp),
                  "level": spec.get("_level"), "missing": missing, "jobs": {},
                  "combine_job": None}
    live = live_jobs(record["jobs"].values())
    todo = {}
    for cell, calls in cells.items():
        if record["jobs"].get(cell) in live:
            continue
        done = scored_methods(out, cell)
        left = [call for call in calls if method_of(call) not in done]
        if left:
            todo[cell] = left
    if not todo:
        print(f"nothing to submit: every call has scores or a live job ({len(live)} live)")
        return

    export = f"--export=ALL,PMEDM_VB_DATA={data_dir(spec)}"
    log = f"--output={out / 'logs' / 'slurm-%j.log'}"
    if not args.dry_run:
        for sub in ("specs", "per_cell", "logs", "detail"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        write_record(out, record)
    submitted = []
    for cell, calls in todo.items():
        if args.max_jobs is not None and len(submitted) >= args.max_jobs:
            print(f"--max-jobs {args.max_jobs} reached; the rest on a later call")
            break
        whole = len(calls) == len(cells[cell])
        spec_file = out / "specs" / (f"{cell}.txt" if whole else
                                     f"{cell}_part{time.strftime('%Y%m%d%H%M%S')}.txt")
        if args.dry_run:
            print(f"  would write {spec_file}: {len(calls)} calls")
        else:
            spec_file.write_text("\n".join(calls) + "\n")
        try:
            job = sbatch(["sbatch", "--parsable", export, log,
                          f"--job-name={spec['name']}-{args.tag}-{cell}", *SCORE_SLURM,
                          str(EXPERIMENTS / "compare_methods.sbatch"), str(spec_file)],
                         args.dry_run)
        except SystemExit as error:
            print(f"{error}\nstopped after {len(submitted)} job(s); call again with --resume")
            break
        record["jobs"][cell] = job
        submitted.append(job)
        if not args.dry_run:
            write_record(out, record)
    print(f"{len(submitted)} scoring job(s) submitted for {args.tag}")
    if args.no_combine or not submitted:
        return
    waiting = sorted(set(submitted) | live_jobs(record["jobs"].values()))
    try:
        record["combine_job"] = sbatch(
            ["sbatch", "--parsable", export, log, f"--job-name={spec['name']}-{args.tag}-combine",
             *COMBINE_SLURM, f"--dependency=afterany:{':'.join(waiting)}",
             str(EXPERIMENTS / "run_python.sbatch"), "score_experiment.py", str(exp),
             "--tag", args.tag, "--combine"], args.dry_run)
    except SystemExit as error:
        record["combine_error"] = str(error)
        print(f"the combine job was not submitted:\n{error}\nWhen the scoring jobs finish, "
              f"combine with\n  {sys.executable} {EXPERIMENTS / 'score_experiment.py'} {exp} "
              f"--tag {args.tag} --combine")
    if not args.dry_run:
        write_record(out, record)
    else:
        print(json.dumps(record, indent=2))


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
               + [f"vb_{f}" for f in spec["vb_families"]]
               + [f"vb_{f}_trunc" for f in spec["vb_families"]] + list(spec["rake"]))
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
    write_record(out, record)
    print(f"combined {len(scores)} cell(s) into {out}; {len(lines)} missing (missing.txt)")


def main() -> None:
    args = parse_args()
    combine(args) if args.combine else submit(args)


if __name__ == "__main__":
    main()
