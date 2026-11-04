"""Run one of the paper's experiments: submit its fits as Slurm jobs into one folder.

An experiment is a JSON file (``experiments/paper/expNN_<name>.json``): the
study area, PUMAs and alphas, the model (taper, variance floor, hierarchy,
roll-up, share cap, multiplier prior), the raking methods, the VB families and the HMC settings.
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
  fits job is still queued or running; not at all while the whitening VB fits
  -- ``hmc.whiten``, skewed by default -- are missing and no fits job is). The campus-gpu QoS takes 6 submitted jobs per
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
Every job is also given ``PMEDM_VB_REPO`` and ``PMEDM_VB_ENV``: the clone this
script runs from and the Python environment running it, so that a job runs
the code and environment that submitted it, whichever project folder they are in.

Optional keys of the JSON:

- ``area.counties: []`` is the whole state; ``area.tables`` is a table set
  (``"default"``, ``"all"``) or a list of table IDs
  (:func:`pmedm_vb.assemble.allcells.table_subset`).
- ``pumas: "all"`` is every PUMA assembled for the area, so assemble them all
  before the first call: the list is fixed in the folder's record then.
- ``heldout: false``: the PUMAs need no held-out tables (nothing is scored).
- ``model.epsilon``: the statewide support (``run_map.py --epsilon``): the fits
  and HMC runs use the inputs assembled with it, ``<area>-state-e<epsilon>``.
- ``vb_device``: the torch device for the VB fits (``"cuda"`` on a GPU node).
- ``slurm``: ``account`` for every job; ``fits`` and ``hmc``, lists of
  ``sbatch`` options replacing the stage's defaults; ``gpu_partition``, where
  GPU jobs are counted against ``--max-gpu-jobs`` (default ``campus-gpu``);
  ``max_gpu_jobs``, that limit's default for this experiment. With
  ``vb_device`` other than ``cpu`` the fits jobs count as GPU jobs too.

- ``bundle``: ``{"jobs": k}`` runs the unfinished PUMAs in ``k`` jobs of
  ``run_puma_bundle.sbatch`` in place of a fits job and an HMC job per PUMA:
  each job fits and samples its PUMAs one per GPU at a time
  (``experiments/run_puma_bundle.py``), with ``slurm.bundle``'s ``sbatch``
  options (``--gpus=4`` for a whole node). A PUMA in a live bundle job is
  not submitted again; one whose bundle ended unfinished is, by the next call.

``--slurm-override`` replaces options of every job a call submits, e.g. to
send a smoke test to a debug partition without editing the JSON.
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
#: Next to the clone: ``/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/paper`` for
#: the UTK0496 clone, and the same place in another project's folder for its clone.
DEFAULT_ROOT = REPO.parent / "pmedm_vb_runs" / "paper"

#: Slurm options per stage, on top of each script's header.
RAKE_SLURM = ["--partition=short", "--qos=short", "--time=01:00:00", "--cpus-per-task=4"]
#: MAP and VB on campus (12 h): the all-cell problems outgrow short's hour.
FITS_SLURM = ["--partition=campus", "--qos=campus", "--time=12:00:00", "--cpus-per-task=48"]
#: The fits job of an experiment with no VB families: the MAP alone, minutes on
#: a few cores, so not worth waiting for a whole campus node.
MAP_SLURM = ["--partition=short", "--qos=short", "--time=01:00:00", "--cpus-per-task=8"]
#: Each PUMA's HMC bundle (short runs, then the long references) on campus-gpu.
HMC_SLURM = ["--partition=campus-gpu", "--qos=campus-gpu", "--time=24:00:00"]
GPU_PARTITION = "campus-gpu"

#: Keys whose difference does not make a different experiment: the description,
#: the scoring-only choices (which VB families are also scored truncated, and
#: the truncation's rule), which leave every fit as it is, and the fits job's
#: Slurm request (fits_slurm).
FREE_KEYS = {"description", "vb_truncated", "vb_trunc_share", "vb_trunc_distinct", "fits_slurm",
             "slurm", "vb_device", "heldout", "bundle"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", type=Path, help="the experiment's JSON file")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"folder holding the experiment folders (default {DEFAULT_ROOT})")
    parser.add_argument("--stage", nargs="+", choices=["rake", "fits", "hmc"],
                        default=["rake", "fits", "hmc"],
                        help="with a bundle block, fits and hmc are run together as bundles")
    parser.add_argument("--status", action="store_true", help="report what is done; submit nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be submitted; write nothing")
    parser.add_argument("--max-gpu-jobs", type=int, default=None,
                        help="the user's jobs (queued or running) in the GPU partition allowed "
                             "at once (default: the JSON's slurm.max_gpu_jobs, else 6)")
    parser.add_argument("--puma", nargs="+", default=None,
                        help="only these PUMAs in this call (e.g. one PUMA as a smoke test); "
                             "the experiment and its seeds are unchanged")
    parser.add_argument("--slurm-override", default="",
                        help="sbatch options replacing those of every job this call submits, "
                             "e.g. '--partition=ai-tenn-debug --qos=ai-tenn-debug --time=01:00:00'")
    return parser.parse_args()


# -- the experiment -----------------------------------------------------------


def load_experiment(path: Path) -> dict:
    """The experiment, with ``pumas`` and ``alphas`` narrowed to its ``subset``.

    ``pumas`` and ``alphas`` are the experiment's grid; an optional ``subset``
    (``{"pumas": [...], "alphas": [...]}``, each a subset of the grid) is what is
    run, scored and tabled. The grid is kept as ``_grid``: it is what tells one
    experiment from another (so a subset of an experiment run on the whole grid
    reuses its fits), and it fixes the HMC seeds, which therefore do not depend
    on the subset.
    """
    spec = json.loads(path.read_text())
    for key in ("name", "area", "pumas", "alphas", "model", "data_dir", "rake", "vb_families",
                "hmc"):
        if key not in spec:
            raise SystemExit(f"{path}: missing '{key}'")
    whiten = (spec["hmc"] or {}).get("whiten", "skewed")
    if spec["hmc"] and whiten != "laplace" and whiten not in spec["vb_families"]:
        raise SystemExit(f"{path}: HMC is whitened by the {whiten} VB fit (hmc.whiten); "
                         f"add '{whiten}' to vb_families")
    if spec["pumas"] == "all":
        spec["pumas"] = assembled_pumas(spec)
        if not spec["pumas"]:
            raise SystemExit(f"{path}: pumas is 'all', but no PUMA is assembled under "
                             f"{inputs_dir(spec)}; run the assemble step first")
    spec["_grid"] = {"pumas": list(spec["pumas"]), "alphas": list(spec["alphas"])}
    subset = spec.get("subset") or {}
    for key in ("pumas", "alphas"):
        if key in subset:
            extra = [v for v in subset[key] if v not in spec[key]]
            if extra:
                raise SystemExit(f"{path}: subset {key} {extra} not in the grid's {key}")
            spec[key] = [v for v in spec[key] if v in subset[key]]
    return spec


def grid(spec: dict) -> dict:
    """The experiment's whole grid of PUMAs and alphas (a folder's record may predate ``_grid``)."""
    return spec.get("_grid") or {"pumas": spec["pumas"], "alphas": spec["alphas"]}


def comparable(spec: dict) -> dict:
    """The spec without its free keys, and without unset (``null``) model options,
    so an option added or renamed later does not tell apart folders that never
    set it."""
    out = {key: value for key, value in spec.items()
           if key not in FREE_KEYS and key != "subset" and not key.startswith("_")}
    out.update(grid(spec))
    if isinstance(out.get("model"), dict):
        out["model"] = {k: v for k, v in out["model"].items() if v is not None}
    return out


def git_commit() -> str:
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    return out.stdout.strip() + ("-dirty" if dirty else "")


def study_area(area: dict):
    from pmedm_vb.config import StudyArea

    return StudyArea(name=area["name"], state=area.get("state", "47"),
                     counties=tuple(area.get("counties", ["093"])), year=area["year"],
                     span=area["span"])


def county_args(area: dict) -> list[str]:
    """run_map.py's ``--county`` options: ``--county all`` for the whole state."""
    counties = area.get("counties", ["093"])
    if not counties:
        return ["--county", "all"]
    return [arg for county in counties for arg in ("--county", county)]


def inputs_name(spec: dict) -> str:
    """The assembled-inputs folder name: the area's, or its statewide support's
    (``<area>-state-e<epsilon>``, as run_map.py names it)."""
    epsilon = spec["model"].get("epsilon")
    slug = area_slug(spec["area"])
    return slug if epsilon is None else f"{slug}-state-e{epsilon:g}"


def inputs_dir(spec: dict) -> Path:
    """Where the area's assembled PUMAs are."""
    return data_dir(spec) / "processed" / "inputs" / inputs_name(spec)


def assembled_pumas(spec: dict) -> list[str]:
    """Every PUMA assembled for the experiment's area, sorted."""
    root = inputs_dir(spec)
    return sorted(d.name for d in root.glob("*") if (d / "manifest.json").exists())


def area_slug(area: dict) -> str:
    return study_area(area).slug


def data_dir(spec: dict) -> Path:
    path = os.path.expandvars(spec["data_dir"])
    if "$" in path:
        raise SystemExit(f"data_dir {spec['data_dir']!r}: a variable in it is not set")
    return Path(path).expanduser()


def level(spec: dict) -> str:
    """The hierarchy level string that names the MAP, VB and HMC results."""
    from pmedm_vb.assemble.hierarchy import level_name
    from pmedm_vb.assemble.prior import prior_level
    from pmedm_vb.assemble.sharecap import cap_level
    from pmedm_vb.assemble.k0 import k0_level
    from pmedm_vb.assemble.tweedie import tweedie_level

    model = spec["model"]
    return k0_level(tweedie_level(
        prior_level(cap_level(level_name(model["hierarchy"], None, model.get("rollup")),
                              model.get("share_cap"), model.get("cap_strength")),
                    model.get("multiplier_prior_sd")),
        model.get("tweedie_weight")), model.get("k0_weight"), model.get("k0_df", 0.0))


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
    if model.get("multiplier_prior_sd") is not None:
        args += ["--prior-sd", f"{model['multiplier_prior_sd']:g}"]
    if model.get("tweedie_weight") is not None:
        args += ["--tweedie-weight", f"{model['tweedie_weight']:g}"]
    if model.get("k0_weight") is not None:
        args += ["--k0-weight", f"{model['k0_weight']:g}", "--k0-df", f"{model.get('k0_df', 0):g}"]
    return args


def area_args(spec: dict) -> list[str]:
    """run_map.py's study-area options."""
    area = spec["area"]
    from pmedm_vb.assemble.allcells import tables_arg

    args = ["--name", area["name"], "--state", area.get("state", "47"),
            "--year", str(area["year"]), "--span", str(area["span"]), *county_args(area)]
    tables = tables_arg(area.get("tables", "default"))
    if tables != "default":
        args += ["--tables", tables]
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


def gpu_partition(spec: dict) -> str:
    """Where the experiment's GPU jobs run, for counting them (``slurm.gpu_partition``)."""
    return (spec.get("slurm") or {}).get("gpu_partition", GPU_PARTITION)


def gpu_jobs_in_queue(partition: str = GPU_PARTITION) -> int:
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-p",
                              partition], capture_output=True, text=True, timeout=60).stdout
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


def slurm_options(spec: dict, stage: str, default: list[str]) -> list[str]:
    """The ``sbatch`` options for a stage: the JSON's ``slurm.<stage>`` (or, for the
    fits, its older ``fits_slurm``) in place of ``default``."""
    slurm = spec.get("slurm") or {}
    if stage == "fits" and spec.get("fits_slurm"):
        default = spec["fits_slurm"]
    return list(slurm.get(stage) or default)


def override(options: list[str], replacements: list[str]) -> list[str]:
    """``options`` with each ``--key=value`` of ``replacements`` in place of the same key."""
    keys = {r.split("=", 1)[0] for r in replacements}
    return [o for o in options if o.split("=", 1)[0] not in keys] + list(replacements)


class Submitter:
    def __init__(self, exp: Path, spec: dict, dry_run: bool, overrides: list[str] | None = None):
        self.exp, self.spec, self.dry_run = exp, spec, dry_run
        self.overrides = list(overrides or [])
        self.commit = git_commit()
        self.count = 0

    def submit(self, stage: str, puma: str, script: str, args: list[str], *,
               slurm: list[str], env: dict[str, str] | None = None,
               after: str | None = None, method: str | None = None,
               extra: dict | None = None) -> str:
        # The jobs run the clone and environment that submit them.
        exports = {"PMEDM_VB_DATA": str(data_dir(self.spec)), "PMEDM_VB_REPO": str(REPO),
                   "PMEDM_VB_ENV": sys.prefix, **(env or {})}
        export = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
        account = (self.spec.get("slurm") or {}).get("account")
        slurm = override([*([f"--account={account}"] if account else []), *slurm], self.overrides)
        command = ["sbatch", "--parsable", f"--export={export}",
                   f"--job-name={self.spec['name']}-{stage}-{puma}" + (f"-{method}" if method else ""),
                   f"--output={self.exp / 'logs' / 'slurm-%j.log'}", *slurm]
        if after:
            # A failed prerequisite cancels the job rather than leaving it
            # pending for good, so a later call resubmits it.
            command += [f"--dependency=afterok:{after}", "--kill-on-invalid-dep=yes"]
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
                  "commit": self.commit, "command": command, **(extra or {})}
        with open(self.exp / "jobs.jsonl", "a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"  submitted {job_id}: {stage} {puma}" + (f" {method}" if method else "")
              + (f" (after {after})" if after else ""))
        return job_id


# -- setup --------------------------------------------------------------------


def check_inputs(spec: dict) -> list[str]:
    """What the jobs need from the data directory and do not find."""
    from pmedm_vb.assemble.allcells import table_set
    from pmedm_vb.assemble.inputs import TREES_FILE

    from pmedm_vb.assemble.allcells import built_tables, table_subset

    root = inputs_dir(spec)
    wanted = spec["area"].get("tables", "default")
    subset = table_subset(wanted)
    problems = []
    for puma in spec["pumas"]:
        folder = root / puma
        if not (folder / "manifest.json").exists():
            problems.append(f"{folder}: not assembled (run_map.sbatch assemble)")
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        names = manifest["tract_constraints"] + manifest["bg_constraints"]
        built = built_tables(names) if subset is not None else table_set(names)
        if built != (subset if subset is not None else wanted):
            problems.append(f"{folder}: assembled with the {built!r} tables, the experiment "
                            f"asks for {wanted!r} -- give that set its own area name")
            continue
        if spec.get("heldout", True) and not (folder / "heldout").exists():
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
        # The free keys follow the JSON: scoring reads them from the record.
        changed = {k: spec.get(k) for k in FREE_KEYS if before.get(k) != spec.get(k)}
        if changed:
            print(f"{saved}: {'would update' if dry_run else 'updating'} "
                  + ", ".join(f"{k} = {v!r}" for k, v in changed.items()))
            if not dry_run:
                saved.write_text(json.dumps({**before, **changed}, indent=2) + "\n")
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


def fits_args(spec: dict) -> list[str]:
    """run_map.py's options for a PUMA's fits, after its name and PUMA."""
    return ["--state", spec["area"].get("state", "47"), "--year", str(spec["area"]["year"]),
            "--span", str(spec["area"]["span"]),
            "--alpha", *[f"{alpha:g}" for alpha in spec["alphas"]], *model_args(spec),
            *county_args(spec["area"]),
            *(["--epsilon", f"{spec['model']['epsilon']:g}"]
              if spec["model"].get("epsilon") is not None else [])]


def stage_fits(sub: Submitter, spec: dict, exp: Path, puma: str, active: dict,
               slots: list[int]) -> str | None:
    """Submit the PUMA's fits unless done or queued; the fits job id when one is live."""
    job = active.get(("fits", puma, None))
    if job:
        return job["job_id"]
    if all(path.exists() for path in fit_outputs(spec, exp, puma)):
        return None
    on_gpu = bool(spec["vb_families"]) and spec.get("vb_device", "cpu") != "cpu"
    if on_gpu and slots[0] <= 0:
        print(f"  fits {puma}: not submitted -- {gpu_partition(spec)} is full; call again later")
        return None
    args = [spec["area"]["name"], puma, *fits_args(spec)]
    env = {"OUT_DIR": str(exp / puma), "FAMILIES": " ".join(spec["vb_families"]),
           "VB_DEVICE": spec.get("vb_device", "cpu")}
    # A spec may set its own request (slurm.fits, or fits_slurm): a small problem
    # need not wait for a whole campus node.
    slurm = slurm_options(spec, "fits", FITS_SLURM if spec["vb_families"] else MAP_SLURM)
    job = sub.submit("fits", puma, "run_map_bundle.sbatch", args, slurm=slurm, env=env)
    if on_gpu:
        slots[0] -= 1
    return job


#: Optional per-kind HMC settings and the run_mcmc.py options they become.
HMC_OPTIONS = {"samples": "--samples", "thin": "--thin", "checkpoint": "--checkpoint",
               "superchains": "--superchains", "draws": "--draws", "ess_target": "--ess-target",
               "max_extend": "--max-extend", "nested_alpha": "--nested-alpha",
               "init_inflate": "--init-inflate", "init_floor": "--init-floor",
               "warmup_init": "--warmup-init",
               "warmup_base": "--warmup-base", "warmup_term": "--warmup-term"}


def whitening_folder(spec: dict) -> str:
    """The fit that whitens HMC: ``vb_<hmc.whiten>``, skewed unless the spec says;
    ``map`` for ``hmc.whiten: "laplace"``, the MAP's Laplace approximation (for a
    model VB cannot fit, such as the K0 term)."""
    whiten = spec["hmc"].get("whiten", "skewed")
    return "map" if whiten == "laplace" else f"vb_{whiten}"


def hmc_spec_lines(spec: dict, exp: Path, puma: str) -> list[str]:
    """One run_mcmc_bundle.sbatch line per HMC run."""
    return [" ".join([str(out), *args]) for out, args in hmc_run_args(spec, exp, puma)]


def hmc_run_args(spec: dict, exp: Path, puma: str) -> list[tuple[Path, list[str]]]:
    """``(out, run_mcmc.py options)`` per HMC run. Each kind's settings may set
    its own ``chains`` and any of :data:`HMC_OPTIONS`; ``adapt_diag: true``
    learns the diagonal in warmup."""
    hmc = spec["hmc"]
    runs = []
    for kind, alpha, out in hmc_runs(spec, exp, puma):
        settings = hmc[kind]
        full = grid(spec)
        seed = (settings["seed"] + 10 * full["pumas"].index(puma)
                + full["alphas"].index(alpha))
        args = ["--puma", puma, "--alpha", f"{alpha:g}", *model_args(spec),
                "--area", inputs_name(spec),
                "--map-run" if whitening_folder(spec) == "map" else "--vb-run",
                str(exp / puma / whitening_folder(spec)),
                "--trajectory", f"{hmc['trajectory']:g}",
                "--chains", str(settings.get("chains", hmc["chains"])),
                "--warmup", str(settings["warmup"]), "--seed", str(seed)]
        for key, option in HMC_OPTIONS.items():
            if key in settings:
                args += [option, f"{settings[key]:g}" if isinstance(settings[key], float)
                         else str(settings[key])]
        if settings.get("adapt_diag"):
            args.append("--adapt-diag")
        runs.append((out, args))
    return runs


def stage_hmc(sub: Submitter, spec: dict, exp: Path, puma: str, active: dict,
              fits_job: str | None, slots: list[int]) -> None:
    if active.get(("hmc", puma, None)):
        return
    if all(path.exists() for path in hmc_outputs(spec, exp, puma)):
        return
    whiteners = [exp / puma / whitening_folder(spec) / f"{fit_name(spec, puma, alpha)}.npz"
                 for alpha in spec["alphas"]]
    if fits_job is None and not all(path.exists() for path in whiteners):
        print(f"  hmc {puma}: waiting -- the {whitening_folder(spec)} fits are missing and no "
              "fits job is live")
        return
    if slots[0] <= 0:
        print(f"  hmc {puma}: not submitted -- {gpu_partition(spec)} is full; call again later")
        return
    spec_file = exp / puma / "hmc_spec.txt"
    lines = hmc_spec_lines(spec, exp, puma)
    if sub.dry_run:
        print(f"  would write {spec_file}:")
        print("\n".join("    " + line for line in lines))
    else:
        spec_file.parent.mkdir(parents=True, exist_ok=True)
        spec_file.write_text("\n".join(lines) + "\n")
    sub.submit("hmc", puma, "run_mcmc_bundle.sbatch", [str(spec_file)],
               slurm=slurm_options(spec, "hmc", HMC_SLURM), after=fits_job)
    slots[0] -= 1


def puma_complete(spec: dict, exp: Path, puma: str) -> bool:
    """Whether the PUMA's fits and HMC reports all exist."""
    paths = fit_outputs(spec, exp, puma) + (hmc_outputs(spec, exp, puma) if spec["hmc"] else [])
    return all(path.exists() for path in paths)


def bundled(active: dict) -> dict[str, dict]:
    """``{puma: job}`` for every PUMA in a live bundle job."""
    return {puma: job for key, job in active.items() if key[0] == "bundle"
            for puma in job.get("pumas", [])}


def stage_bundle(sub: Submitter, spec: dict, exp: Path, active: dict, slots: list[int]) -> None:
    """Submit the unfinished PUMAs that no live bundle job holds, split over
    ``bundle.jobs`` jobs of run_puma_bundle.sbatch."""
    live = bundled(active)
    todo = [p for p in spec["pumas"] if p not in live and not puma_complete(spec, exp, p)]
    if live:
        print(f"bundles: {len(live)} PUMA(s) in live job(s) "
              f"{', '.join(sorted({job['job_id'] for job in live.values()}))}")
    if not todo:
        print("bundles: nothing to submit")
        return
    jobs = max(1, min(int(spec["bundle"].get("jobs", 1)), len(todo)))
    groups = [todo[k::jobs] for k in range(jobs)]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for k, group in enumerate(groups):
        if slots[0] <= 0:
            print(f"bundles: {gpu_partition(spec)} is full; {len(groups) - k} job(s) wait "
                  "for a later call")
            return
        items = []
        for puma in group:
            items.append({
                "puma": puma, "out": str(exp / puma),
                "fit_outputs": [str(path) for path in fit_outputs(spec, exp, puma)],
                "hmc": [{"out": str(out), "args": args, "report": str(report)}
                        for (out, args), report in zip(hmc_run_args(spec, exp, puma),
                                                       hmc_outputs(spec, exp, puma))]
                       if spec["hmc"] else [],
            })
        bundle = {"name": spec["area"]["name"], "fits_args": fits_args(spec),
                  "families": spec["vb_families"], "draws": 64,
                  "device": spec.get("vb_device", "cpu"), "pumas": items}
        path = exp / "bundles" / f"bundle-{stamp}-{k + 1}.json"
        print(f"bundle {k + 1}/{jobs}: {len(group)} PUMA(s) {' '.join(group)}")
        if sub.dry_run:
            print(f"  would write {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(bundle, indent=2) + "\n")
        sub.submit("bundle", f"{stamp}-{k + 1}", "run_puma_bundle.sbatch", [str(path)],
                   slurm=slurm_options(spec, "bundle", FITS_SLURM), extra={"pumas": group})
        slots[0] -= 1


# -- status -------------------------------------------------------------------


def status(spec: dict, exp: Path, active: dict) -> None:
    print(f"{exp}  (level {level(spec)})")
    live = bundled(active)
    if spec.get("bundle"):
        done = sum(puma_complete(spec, exp, p) for p in spec["pumas"])
        print(f"  {done} of {len(spec['pumas'])} PUMA(s) complete; {len(live)} in live bundle job(s)")
    for puma in spec["pumas"]:
        print(f"  {puma}" + (f"  (bundle job {live[puma]['job_id']} {live[puma]['state'].lower()})"
                             if puma in live else ""))
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


def active_for(exp: Path) -> dict:
    """The experiment's last job per (stage, PUMA, method), where still queued or running."""
    jobs = latest_jobs(read_jobs(exp))
    states = active_jobs([job["job_id"] for job in jobs.values()])
    return {key: {**job, "state": states[job["job_id"]]}
            for key, job in jobs.items() if job["job_id"] in states}


def is_complete(spec: dict, exp: Path) -> bool:
    """Whether every raking, MAP, VB and HMC output exists."""
    for puma in spec["pumas"]:
        paths = [rake_output(spec, exp, puma, m) for m in spec["rake"]] + fit_outputs(spec, exp, puma)
        if spec["hmc"]:
            paths += hmc_outputs(spec, exp, puma)
        if not all(path.exists() for path in paths):
            return False
    return True


def advance(spec: dict, root: Path, stages=("rake", "fits", "hmc"),
            max_gpu_jobs: int | None = None, dry_run: bool = False,
            overrides: list[str] | None = None) -> int:
    """Submit, for every PUMA, whatever is neither finished nor queued or running; the
    number of jobs submitted. A refused ``sbatch`` (a queue limit) ends the call
    early, keeping what was submitted: the next call goes on from there."""
    exp = root / spec["name"]
    active = active_for(exp)
    problems = check_inputs(spec)
    if problems:
        print("the data directory is not ready:\n  " + "\n  ".join(problems))
        if not dry_run:
            return 0
    prepare(exp, spec, dry_run)
    sub = Submitter(exp, spec, dry_run, overrides)
    if max_gpu_jobs is None:
        max_gpu_jobs = (spec.get("slurm") or {}).get("max_gpu_jobs", 6)
    slots = [max_gpu_jobs - gpu_jobs_in_queue(gpu_partition(spec))]
    try:
        if spec.get("bundle"):
            if "rake" in stages and spec["rake"]:
                for puma in spec["pumas"]:
                    stage_rake(sub, spec, exp, puma, active)
            if "fits" in stages or "hmc" in stages:
                stage_bundle(sub, spec, exp, active, slots)
            return sub.count
        for puma in spec["pumas"]:
            print(f"{puma}:")
            if "rake" in stages:
                stage_rake(sub, spec, exp, puma, active)
            fits_job = (stage_fits(sub, spec, exp, puma, active, slots) if "fits" in stages
                        else (active.get(("fits", puma, None)) or {}).get("job_id"))
            if "hmc" in stages and spec["hmc"]:
                stage_hmc(sub, spec, exp, puma, active, fits_job, slots)
    except SystemExit as error:
        print(f"stopped early: {error}")
    return sub.count


def main() -> None:
    args = parse_args()
    spec = load_experiment(args.experiment)
    if args.puma:
        unknown = sorted(set(args.puma) - set(spec["pumas"]))
        if unknown:
            raise SystemExit(f"--puma {unknown}: not among the experiment's PUMAs")
        spec["pumas"] = [p for p in spec["pumas"] if p in set(args.puma)]
    exp = args.root / spec["name"]
    if args.status:
        status(spec, exp, active_for(exp))
        return
    if check_inputs(spec) and not args.dry_run:
        print("the data directory is not ready:\n  " + "\n  ".join(check_inputs(spec)))
        raise SystemExit(1)
    count = advance(spec, args.root, args.stage, args.max_gpu_jobs, args.dry_run,
                    args.slurm_override.split())
    print(f"{count} job(s) {'would be ' if args.dry_run else ''}submitted; see --status")


if __name__ == "__main__":
    sys.exit(main())
