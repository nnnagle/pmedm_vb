"""Run one of the paper's experiments: submit its fits as Slurm jobs into one folder.

An experiment is a JSON file (``experiments/paper/expNN_<name>.json``): the
study area, PUMAs and alphas, the model (taper, variance floor, hierarchy,
roll-up, share cap), the raking methods, the VB families and the HMC settings.
Everything it produces lives under ``<root>/<name>``::

    <root>/<name>/
      experiment.json          the JSON, plus the commit and data directory it was started with
      jobs.jsonl               one line per submitted job
      logs/                    the jobs' Slurm logs
      <puma>/
        ipf/ sinkhorn/         raking   (run_map.py rake)
        map/ vb_<family>/      MAP and VB, every alpha   (run_map_bundle.sbatch)
        hmc_short/ hmc_ref/    HMC, every alpha   (run_mcmc_bundle.sbatch; hmc_spec.txt)
      scores/<tag>/            score_experiment.py

Usage, from the repository root on a login node::

    $CONDA_PREFIX/bin/python experiments/run_experiment.py experiments/paper/exp01_baseline.json
    $CONDA_PREFIX/bin/python experiments/run_experiment.py experiments/paper/exp01_baseline.json --status

Each call submits, for every PUMA, whatever is neither finished nor already
queued or running:

- **rake**: one ``run_map.sbatch rake`` job per (PUMA, method), on ``short``;
- **fits**: one ``run_map_bundle.sbatch`` job per PUMA -- MAP, then the VB
  families, every alpha;
- **hmc**: one ``run_mcmc_bundle.sbatch`` job per PUMA -- short runs at every
  alpha, then the references -- after that PUMA's fits (``afterok`` when the
  fits job is still queued or running; not at all while the skewed VB fits are
  missing and no fits job is). The campus-gpu QoS takes 6 submitted jobs per
  user, so only as many are submitted as ``--max-gpu-jobs`` less the user's
  jobs already in campus-gpu; the rest wait for a later call.

So the experiment is run by calling this again until ``--status`` shows every
output: a job that ran out of time resumes where it stopped (``run_map.py``
skips finished fits; the HMC sampler checkpoints). ``--stage`` limits a call to
some stages, ``--dry-run`` prints the ``sbatch`` commands and spec files
without submitting or writing anything.

The JSON is copied into the folder on the first call; a later call with a
JSON that differs (other than in ``description``) is refused, so one folder
always holds one experiment. Every job is given ``PMEDM_VB_DATA`` from the
JSON's ``data_dir`` (``$USER`` expanded) and refuses to start unless that
directory holds the assembled inputs (and, under a roll-up, the table trees).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
DEFAULT_ROOT = Path("/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/paper")

#: Slurm options per stage, on top of each script's header.
RAKE_SLURM = ["--partition=short", "--qos=short", "--time=01:00:00", "--cpus-per-task=4"]
FITS_SLURM = ["--cpus-per-task=48"]
HMC_SLURM: list[str] = []
GPU_PARTITION = "campus-gpu"

#: Keys whose difference does not make a different experiment.
FREE_KEYS = {"description"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", type=Path, help="the experiment's JSON file")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"folder holding the experiment folders (default {DEFAULT_ROOT})")
    parser.add_argument("--stage", nargs="+", choices=["rake", "fits", "hmc"],
                        default=["rake", "fits", "hmc"])
    parser.add_argument("--status", action="store_true", help="report what is done; submit nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be submitted; write nothing")
    parser.add_argument("--max-gpu-jobs", type=int, default=6,
                        help="the user's campus-gpu jobs (queued or running) allowed at once")
    return parser.parse_args()


# -- the experiment -----------------------------------------------------------


def load_experiment(path: Path) -> dict:
    spec = json.loads(path.read_text())
    for key in ("name", "area", "pumas", "alphas", "model", "data_dir", "rake", "vb_families",
                "hmc"):
        if key not in spec:
            raise SystemExit(f"{path}: missing '{key}'")
    if "skewed" not in spec["vb_families"] and spec["hmc"]:
        raise SystemExit(f"{path}: HMC is whitened by the skewed VB fit; add 'skewed' to vb_families")
    return spec


def comparable(spec: dict) -> dict:
    """The spec without its free keys, and without unset (``null``) model options,
    so an option added or renamed later does not tell apart folders that never
    set it."""
    out = {key: value for key, value in spec.items()
           if key not in FREE_KEYS and not key.startswith("_")}
    if isinstance(out.get("model"), dict):
        out["model"] = {k: v for k, v in out["model"].items() if v is not None}
    return out


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    return out.stdout.strip() + ("-dirty" if dirty else "")


def area_slug(area: dict) -> str:
    from pmedm_vb.config import StudyArea

    return StudyArea(name=area["name"], state=area.get("state", "47"),
                     counties=tuple(area.get("counties", ["093"])), year=area["year"],
                     span=area["span"]).slug


def data_dir(spec: dict) -> Path:
    path = os.path.expandvars(spec["data_dir"])
    if "$" in path:
        raise SystemExit(f"data_dir {spec['data_dir']!r}: a variable in it is not set")
    return Path(path).expanduser()


def level(spec: dict) -> str:
    """The hierarchy level string that names the MAP, VB and HMC results."""
    from pmedm_vb.assemble.hierarchy import level_name
    from pmedm_vb.assemble.sharecap import cap_level

    model = spec["model"]
    return cap_level(level_name(model["hierarchy"], None, model.get("rollup")),
                     model.get("share_cap"), model.get("cap_strength"))


def fit_name(spec: dict, puma: str, alpha: float) -> str:
    """``<puma>_<taper>_a<alpha>[_h<level>]``, as run_map.py and run_mcmc.py name them."""
    lev = level(spec)
    return f"{puma}_{spec['model']['taper']}_a{alpha:g}" + ("" if lev == "none" else f"_h{lev}")


def model_args(spec: dict) -> list[str]:
    """The model's options, shared by run_map.py, run_mcmc.py and compare_methods.py."""
    model = spec["model"]
    args = ["--taper", model["taper"], "--variance-floor", str(model["variance_floor"]),
            "--hierarchy", model["hierarchy"]]
    if model.get("rollup"):
        args += ["--rollup", str(model["rollup"])]
    if model.get("share_cap") is not None:
        args += ["--share-cap", f"{model['share_cap']:g}",
                 "--cap-strength", f"{model['cap_strength']:g}"]
    return args


def area_args(spec: dict) -> list[str]:
    """run_map.py's study-area options."""
    area = spec["area"]
    args = ["--name", area["name"], "--state", area.get("state", "47"),
            "--year", str(area["year"]), "--span", str(area["span"])]
    for county in area.get("counties", ["093"]):
        args += ["--county", county]
    return args


# -- what is done -------------------------------------------------------------


def rake_output(spec: dict, exp: Path, puma: str, method: str) -> Path:
    from pmedm_vb.rake import rake_name

    return exp / puma / method / f"{rake_name(puma, method, spec['model'].get('rollup'))}.npz"


def fit_outputs(spec: dict, exp: Path, puma: str) -> list[Path]:
    folders = ["map"] + [f"vb_{family}" for family in spec["vb_families"]]
    return [exp / puma / folder / f"{fit_name(spec, puma, alpha)}.npz"
            for folder in folders for alpha in spec["alphas"]]


def hmc_runs(spec: dict, exp: Path, puma: str) -> list[tuple[str, float, Path]]:
    """``(kind, alpha, out)`` in the order the bundle runs them: short runs first."""
    return [(kind, alpha, exp / puma / f"hmc_{kind}")
            for kind in ("short", "ref") if kind in spec["hmc"].get("kinds", ["short", "ref"])
            for alpha in spec["alphas"]]


def hmc_outputs(spec: dict, exp: Path, puma: str) -> list[Path]:
    # run_mcmc_bundle.sbatch writes the report only after the sampler has finished.
    return [out / f"{fit_name(spec, puma, alpha)}_report.txt"
            for _, alpha, out in hmc_runs(spec, exp, puma)]


# -- Slurm --------------------------------------------------------------------


def active_jobs(job_ids: list[str]) -> dict[str, str]:
    """``{job id: state}`` for those of ``job_ids`` still queued or running."""
    if not job_ids:
        return {}
    try:
        out = subprocess.run(["squeue", "-h", "-o", "%i %T", "-j", ",".join(job_ids)],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(line.split() for line in out.splitlines() if line.strip())


def gpu_jobs_in_queue() -> int:
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-p",
                              GPU_PARTITION], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.splitlines() if line.strip())


def read_jobs(exp: Path) -> list[dict]:
    path = exp / "jobs.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_jobs(jobs: list[dict]) -> dict[tuple, dict]:
    """The last job submitted for each (stage, puma, method)."""
    return {(job["stage"], job["puma"], job.get("method")): job for job in jobs}


class Submitter:
    def __init__(self, exp: Path, spec: dict, dry_run: bool):
        self.exp, self.spec, self.dry_run = exp, spec, dry_run
        self.commit = git_commit()
        self.count = 0

    def submit(self, stage: str, puma: str, script: str, args: list[str], *,
               slurm: list[str], env: dict[str, str] | None = None,
               after: str | None = None, method: str | None = None) -> str:
        exports = {"PMEDM_VB_DATA": str(data_dir(self.spec)), **(env or {})}
        export = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
        command = ["sbatch", "--parsable", f"--export={export}",
                   f"--job-name={self.spec['name']}-{stage}-{puma}" + (f"-{method}" if method else ""),
                   f"--output={self.exp / 'logs' / 'slurm-%j.log'}", *slurm]
        if after:
            command.append(f"--dependency=afterok:{after}")
        command += [str(EXPERIMENTS / script), *args]
        self.count += 1
        if self.dry_run:
            print("  " + " ".join(f"'{part}'" if " " in part else part for part in command))
            return f"DRYRUN{self.count}"
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"sbatch failed: {result.stderr.strip()}\n{' '.join(command)}")
        job_id = result.stdout.strip().split(";")[0]
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job_id": job_id,
                  "stage": stage, "puma": puma, "method": method, "after": after,
                  "commit": self.commit, "command": command}
        with open(self.exp / "jobs.jsonl", "a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"  submitted {job_id}: {stage} {puma}" + (f" {method}" if method else "")
              + (f" (after {after})" if after else ""))
        return job_id


# -- setup --------------------------------------------------------------------


def check_inputs(spec: dict) -> list[str]:
    """What the jobs need from the data directory and do not find."""
    from pmedm_vb.assemble.inputs import TREES_FILE

    root = data_dir(spec) / "processed" / "inputs" / area_slug(spec["area"])
    problems = []
    for puma in spec["pumas"]:
        folder = root / puma
        if not (folder / "manifest.json").exists():
            problems.append(f"{folder}: not assembled (run_map.sbatch assemble)")
            continue
        if not (folder / "heldout").exists():
            problems.append(f"{folder}/heldout: missing (run_map.sbatch heldout)")
        if spec["model"].get("rollup") and not (folder / TREES_FILE).exists():
            problems.append(f"{folder}/{TREES_FILE}: missing (table_trees.py --write-inputs)")
    return problems


def prepare(exp: Path, spec: dict, dry_run: bool) -> None:
    """Create the folder and record the experiment, or check it is the same one."""
    saved = exp / "experiment.json"
    if saved.exists():
        before = json.loads(saved.read_text())
        if comparable(before) != comparable(spec):
            raise SystemExit(f"{saved} records a different experiment; "
                             "give the changed one a new name")
        return
    record = {**spec, "_started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "_commit": git_commit(), "_data_dir": str(data_dir(spec)),
              "_level": level(spec)}
    if dry_run:
        print(f"would create {exp} and write {saved}")
        return
    (exp / "logs").mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(record, indent=2) + "\n")
    print(f"created {exp}")


# -- stages -------------------------------------------------------------------


def stage_rake(sub: Submitter, spec: dict, exp: Path, puma: str, active: dict) -> None:
    for method in spec["rake"]:
        output = rake_output(spec, exp, puma, method)
        job = active.get(("rake", puma, method))
        if output.exists() or job:
            continue
        args = ["rake", *area_args(spec), "--puma", puma, "--method", method,
                "--variance-floor", str(spec["model"]["variance_floor"]),
                "--out", str(output.parent)]
        if spec["model"].get("rollup"):
            args += ["--rollup", str(spec["model"]["rollup"])]
        sub.submit("rake", puma, "run_map.sbatch", args, slurm=RAKE_SLURM, method=method)


def stage_fits(sub: Submitter, spec: dict, exp: Path, puma: str, active: dict) -> str | None:
    """Submit the PUMA's fits unless done or queued; the fits job id when one is live."""
    job = active.get(("fits", puma, None))
    if job:
        return job["job_id"]
    if all(path.exists() for path in fit_outputs(spec, exp, puma)):
        return None
    args = [spec["area"]["name"], puma, "--state", spec["area"].get("state", "47"),
            "--year", str(spec["area"]["year"]), "--span", str(spec["area"]["span"]),
            "--alpha", *[f"{alpha:g}" for alpha in spec["alphas"]], *model_args(spec)]
    for county in spec["area"].get("counties", ["093"]):
        args += ["--county", county]
    env = {"OUT_DIR": str(exp / puma), "FAMILIES": " ".join(spec["vb_families"])}
    return sub.submit("fits", puma, "run_map_bundle.sbatch", args, slurm=FITS_SLURM, env=env)


def hmc_spec_lines(spec: dict, exp: Path, puma: str) -> list[str]:
    hmc = spec["hmc"]
    lines = []
    for kind, alpha, out in hmc_runs(spec, exp, puma):
        settings = hmc[kind]
        seed = (settings["seed"] + 10 * spec["pumas"].index(puma)
                + spec["alphas"].index(alpha))
        args = ["--puma", puma, "--alpha", f"{alpha:g}", *model_args(spec),
                "--area", area_slug(spec["area"]), "--vb-run", str(exp / puma / "vb_skewed"),
                "--trajectory", f"{hmc['trajectory']:g}", "--chains", str(hmc["chains"]),
                "--warmup", str(settings["warmup"]), "--samples", str(settings["samples"]),
                "--thin", str(settings["thin"]), "--seed", str(seed)]
        if "checkpoint" in settings:
            args += ["--checkpoint", str(settings["checkpoint"])]
        lines.append(" ".join([str(out), *args]))
    return lines


def stage_hmc(sub: Submitter, spec: dict, exp: Path, puma: str, active: dict,
              fits_job: str | None, slots: list[int]) -> None:
    if active.get(("hmc", puma, None)):
        return
    if all(path.exists() for path in hmc_outputs(spec, exp, puma)):
        return
    whiteners = [exp / puma / "vb_skewed" / f"{fit_name(spec, puma, alpha)}.npz"
                 for alpha in spec["alphas"]]
    if fits_job is None and not all(path.exists() for path in whiteners):
        print(f"  hmc {puma}: waiting -- the skewed VB fits are missing and no fits job is live")
        return
    if slots[0] <= 0:
        print(f"  hmc {puma}: not submitted -- {GPU_PARTITION} is full; call again later")
        return
    spec_file = exp / puma / "hmc_spec.txt"
    lines = hmc_spec_lines(spec, exp, puma)
    if sub.dry_run:
        print(f"  would write {spec_file}:")
        print("\n".join("    " + line for line in lines))
    else:
        spec_file.parent.mkdir(parents=True, exist_ok=True)
        spec_file.write_text("\n".join(lines) + "\n")
    sub.submit("hmc", puma, "run_mcmc_bundle.sbatch", [str(spec_file)], slurm=HMC_SLURM,
               after=fits_job)
    slots[0] -= 1


# -- status -------------------------------------------------------------------


def status(spec: dict, exp: Path, active: dict) -> None:
    print(f"{exp}  (level {level(spec)})")
    for puma in spec["pumas"]:
        print(f"  {puma}")
        for method in spec["rake"]:
            done = rake_output(spec, exp, puma, method).exists()
            print(f"    {method:<12} {'done' if done else live(active, ('rake', puma, method))}")
        groups = [("map", "map")] + [(f"vb_{f}", f"vb_{f}") for f in spec["vb_families"]]
        for label, folder in groups:
            have = [f"a{alpha:g}" for alpha in spec["alphas"]
                    if (exp / puma / folder / f"{fit_name(spec, puma, alpha)}.npz").exists()]
            print(f"    {label:<12} {len(have)}/{len(spec['alphas'])} {' '.join(have)}")
        print(f"    {'fits job':<12} {live(active, ('fits', puma, None))}")
        for kind in ("short", "ref"):
            runs = [(alpha, out) for k, alpha, out in hmc_runs(spec, exp, puma) if k == kind]
            if not runs:
                continue
            have = [f"a{alpha:g}" for alpha, out in runs
                    if (out / f"{fit_name(spec, puma, alpha)}_report.txt").exists()]
            print(f"    hmc_{kind:<8} {len(have)}/{len(runs)} {' '.join(have)}")
        print(f"    {'hmc job':<12} {live(active, ('hmc', puma, None))}")


def live(active: dict, key: tuple) -> str:
    job = active.get(key)
    return f"job {job['job_id']} {job['state'].lower()}" if job else "-"


def main() -> None:
    args = parse_args()
    spec = load_experiment(args.experiment)
    exp = args.root / spec["name"]
    jobs = latest_jobs(read_jobs(exp))
    states = active_jobs([job["job_id"] for job in jobs.values()])
    active = {key: {**job, "state": states[job["job_id"]]}
              for key, job in jobs.items() if job["job_id"] in states}
    if args.status:
        status(spec, exp, active)
        return

    problems = check_inputs(spec)
    if problems:
        print("the data directory is not ready:\n  " + "\n  ".join(problems))
        if not args.dry_run:
            raise SystemExit(1)
    prepare(exp, spec, args.dry_run)
    sub = Submitter(exp, spec, args.dry_run)
    slots = [args.max_gpu_jobs - gpu_jobs_in_queue()]
    for puma in spec["pumas"]:
        print(f"{puma}:")
        if "rake" in args.stage:
            stage_rake(sub, spec, exp, puma, active)
        fits_job = (stage_fits(sub, spec, exp, puma, active) if "fits" in args.stage
                    else (active.get(("fits", puma, None)) or {}).get("job_id"))
        if "hmc" in args.stage and spec["hmc"]:
            stage_hmc(sub, spec, exp, puma, active, fits_job, slots)
    print(f"{sub.count} job(s) {'would be ' if args.dry_run else ''}submitted; "
          f"see --status")


if __name__ == "__main__":
    sys.exit(main())
