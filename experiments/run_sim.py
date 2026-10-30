"""Run a design-based simulation (``experiments/sim/<name>.json``) from its truths
to ``sim.zip``, and resume it.

Each stage is started once what it needs exists and nothing for it is queued
or running:

1. **build** -- ``sim_build.py``: the truths, their surveys, the calibrated
   ``c`` and the synthetic inputs (``<root>/<fits>/simulation``, the data
   directory); done when ``calibration.json`` is there, which it writes last;
2. **fits** -- :func:`run_experiment.advance` on the fits JSON
   (``experiments/sim/<name>_fits.json``): raking, MAP + VB, then HMC short,
   topping the GPU queue up as slots free;
3. **collect** -- ``sim_collect.py``: the parquet tabulation, redone when its
   code changes or a fit or truth is newer than it (its stamp,
   ``provenance.py``);
4. **score** -- ``sim_score.py``: the scores, the supplement tables and the
   summaries, redone when its code changes or the tabulation is newer;
5. **zip** -- ``<root>/<fits>/sim.zip`` for the Quarto project
   (``figures/sim_manifest.txt``: ``fig_sim.R`` and the scores' tables), built
   in the driver itself, and rebuilt when any file it packs is newer.

As ``run_paper.py``: ``start`` (on a login node) runs one pass and submits
``run_sim.sbatch``, a small job on ``short`` that runs ``tick`` -- one
:func:`advance` -- and, while work remains, submits itself again ``--every``
minutes later; when everything is done it submits a one-line job for the
"done" e-mail, and a stage that needs attention exits non-zero, which mails the
failure. Its jobs and logs are under ``<root>/<fits>/run_sim``; a stage
submitted ``--max-attempts`` times at the current commit without finishing
stops the run. ``--dry-run`` prints what one pass would submit::

    S=experiments/sim/sim01.json
    $CONDA_PREFIX/bin/python experiments/run_sim.py start --sim $S --root $R
    $CONDA_PREFIX/bin/python experiments/run_sim.py status --sim $S --root $R

Unzipped in the Quarto project's folder, ``sim.zip`` gives ``sim/``:
``source("sim/fig_sim.R"); plot_sim(sim_data("sim"))``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))

import make_results  # noqa: E402
import provenance as pv  # noqa: E402
import run_experiment as rx  # noqa: E402
from run_paper import (ATTENTION, DONE, NOTIFY, WAITING, Attention, Jobs, Lock,  # noqa: E402
                       exp_attempts, guard, key_done)
from sim_common import Simulation  # noqa: E402

FOLDER = "run_sim"
MANIFEST = Path(__file__).resolve().parent / "figures" / "sim_manifest.txt"

#: Slurm options of the driver's jobs (on top of run_python.sbatch's header).
SHORT = ["--partition=short", "--qos=short", "--time=01:00:00", "--cpus-per-task=4",
         "--mem=64G"]
#: sim_score.py's settings, as the driver runs it (its stamp records them).
SCORE_SETTINGS = dict(subtotals=False, by_table=False, dh_floor=0.04, seed=0)
HH_MIN = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["start", "tick", "advance", "status"])
    parser.add_argument("--sim", type=Path, required=True, help="experiments/sim/<name>.json")
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--every", type=int, default=30, help="minutes between driver runs")
    parser.add_argument("--max-gpu-jobs", type=int, default=6)
    parser.add_argument("--max-attempts", type=int, default=4,
                        help="submissions of one stage's job at the current commit before "
                             "asking for attention")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.sim, args.root = args.sim.resolve(), args.root.resolve()
    return args


# -- the stages' state -------------------------------------------------------------


def outputs(sim: Simulation) -> dict[str, Path]:
    out = sim.folder / "output"
    return {"calibration": sim.folder / "calibration.json", "collect": out / "draws.parquet",
            "score": out / "scores" / "summary.md", "zip": sim.fits_folder / "sim.zip"}


def built(sim: Simulation) -> bool:
    """Whether the build is complete; a partial one (calibration.json written,
    something missing) needs a person, since fits may use what is there."""
    if not outputs(sim)["calibration"].exists():
        return False
    missing = [p for p in sim.sim_pumas() if not sim.truth_file(p).exists()
               or not (sim.sim_inputs(p) / "manifest.json").exists()]
    if missing:
        raise Attention(f"build: calibration.json is written but {missing[:3]}... are missing; "
                        f"rerun sim_build.py --overwrite by hand, then the fits")
    return True


def fit_files(sim: Simulation) -> list[Path]:
    spec, exp = sim.fits, sim.fits_folder
    paths = []
    for puma in spec["pumas"]:
        paths += [rx.rake_output(spec, exp, puma, m) for m in spec["rake"]]
        paths += rx.fit_outputs(spec, exp, puma) + rx.hmc_outputs(spec, exp, puma)
        paths.append(sim.truth_file(puma))
    return paths


def newer_than(stamped: Path, inputs: list[Path]) -> Path | None:
    """The newest of ``inputs`` if it is newer than ``stamped``'s stamp."""
    newest = max(inputs, key=lambda path: path.stat().st_mtime)
    stamp = pv.stamp_path(stamped)
    return newest if newest.stat().st_mtime > stamp.stat().st_mtime else None


def collect_stale(sim: Simulation) -> str | None:
    out = outputs(sim)["collect"]
    why = pv.stale(out, pv.SIM_COLLECT_SCRIPTS, sim=sim.settings["name"], hh_min=HH_MIN)
    if why:
        return why
    newer = newer_than(out, fit_files(sim))
    return f"{newer} is newer than {out}" if newer else None


def score_stale(sim: Simulation) -> str | None:
    out = outputs(sim)["score"]
    why = pv.stale(out, pv.SIM_SCORE_SCRIPTS, **SCORE_SETTINGS)
    if why:
        return why
    newer = newer_than(out, [pv.stamp_path(outputs(sim)["collect"])])
    return f"the tabulation is newer than {out}" if newer else None


def zip_roots(sim: Simulation) -> dict[str, Path]:
    return {"scores": outputs(sim)["score"].parent, "sim": sim.folder,
            "output": outputs(sim)["collect"].parent}


def zip_stale(sim: Simulation) -> str | None:
    out = outputs(sim)["zip"]
    if not out.exists():
        return f"{out} is missing"
    files = make_results.entries(sim.folder, MANIFEST, zip_roots(sim))
    newest = max([MANIFEST, *(path for path, _, _ in files)], key=lambda p: p.stat().st_mtime)
    if newest.stat().st_mtime > out.stat().st_mtime:
        return f"{newest} is newer than {out}"
    return None


# -- one pass ----------------------------------------------------------------------


def submit(jobs: Jobs, key: str, sim: Simulation, script: str, args) -> None:
    guard(jobs, key, args)
    jobs.sbatch(key, sim.source, "run_python.sbatch",
                [script, "--sim", str(args.sim), "--root", str(args.root)]
                + (["--overwrite"] if key == "build" else []), SHORT)


def advance(args) -> int:
    """One pass over every stage; DONE, WAITING or ATTENTION."""
    sim = Simulation(args.sim, args.root)
    jobs = Jobs(sim.fits_folder, args.dry_run, folder=FOLDER, prefix=sim.settings["name"])
    try:
        if not built(sim):
            if jobs.is_live("build"):
                print("build: job live")
            else:
                print("build: truths, surveys and inputs not written")
                submit(jobs, "build", sim, "sim_build.py", args)
            return WAITING

        spec, exp = sim.fits, sim.fits_folder
        if not rx.is_complete(spec, exp):
            over = {k: n for k, n in exp_attempts(exp, jobs.commit).items()
                    if n >= args.max_attempts}
            live = rx.active_for(exp)
            stuck = [k for k in over if k not in live and not key_done(spec, exp, k)]
            if stuck:
                raise Attention(f"fits: {stuck} submitted {args.max_attempts}+ times at "
                                f"commit {jobs.commit[:12]} without finishing; see {exp / 'logs'}")
            print("fits:")
            rx.advance(spec, args.root, max_gpu_jobs=args.max_gpu_jobs, dry_run=args.dry_run)
            return WAITING

        for key, stale, script in (("collect", collect_stale, "sim_collect.py"),
                                   ("score", score_stale, "sim_score.py")):
            why = stale(sim)
            if not why:
                continue
            if jobs.is_live(key):
                print(f"{key}: job live")
            else:
                print(f"{key}: {why}")
                submit(jobs, key, sim, script, args)
            return WAITING

        why = zip_stale(sim)
        if why:
            print(f"zip: {why}; building")
            if not args.dry_run:
                try:
                    make_results.build(sim.folder, outputs(sim)["zip"], MANIFEST, zip_roots(sim))
                except SystemExit as error:
                    raise Attention(f"sim.zip: {error}") from None
        print(f"done: every stage is complete; {outputs(sim)['zip']}")
        return DONE
    except Attention as error:
        print(f"NEEDS ATTENTION: {error}")
        return ATTENTION


# -- the driver job ------------------------------------------------------------------


def driver_args(args) -> list[str]:
    return ["--sim", str(args.sim), "--root", str(args.root), "--every", str(args.every),
            "--max-gpu-jobs", str(args.max_gpu_jobs), "--max-attempts", str(args.max_attempts)]


def tick(args) -> int:
    sim = Simulation(args.sim, args.root)
    with Lock(sim.fits_folder, FOLDER):
        result = advance(args)
        jobs = Jobs(sim.fits_folder, args.dry_run, folder=FOLDER, prefix=sim.settings["name"])
        if result == WAITING:
            jobs.submit("driver", ["sbatch", "--parsable",
                                   f"--output={jobs.dir / 'logs' / 'driver-%j.log'}",
                                   f"--begin=now+{args.every}minutes",
                                   str(rx.EXPERIMENTS / "run_sim.sbatch"), *driver_args(args)])
        elif result == DONE:
            jobs.submit("notify", ["sbatch", "--parsable", *NOTIFY,
                                   f"--output={jobs.dir / 'logs' / 'done-%j.log'}",
                                   f"--job-name={sim.settings['name']}-done",
                                   f"--wrap=echo the simulation {sim.settings['name']} "
                                   "is complete"])
    return result


def start(args) -> int:
    sim = Simulation(args.sim, args.root)
    jobs = Jobs(sim.fits_folder, args.dry_run, folder=FOLDER, prefix=sim.settings["name"])
    if jobs.is_live("driver"):
        raise SystemExit(f"a driver job is already queued or running: "
                         f"{jobs.latest['driver']['job_id']}")
    sim.fits_folder.mkdir(parents=True, exist_ok=True)
    return advance(args) if args.dry_run else tick(args)


def status(args) -> None:
    sim = Simulation(args.sim, args.root)
    jobs = Jobs(sim.fits_folder, dry_run=True, folder=FOLDER, prefix=sim.settings["name"])
    live = {key: " (job live)" if jobs.is_live(key) else "" for key in
            ("build", "collect", "score", "driver")}
    print(f"{sim.settings['name']}: fits in {sim.fits_folder}")
    try:
        state = "done" if built(sim) else "not done"
    except Attention as error:
        state = str(error)
    print(f"  {'build':<8} {state}{live['build']}")
    complete = rx.is_complete(sim.fits, sim.fits_folder)
    running = len(rx.active_for(sim.fits_folder))
    print(f"  {'fits':<8} {'done' if complete else 'not done'}"
          + (f" ({running} job(s) live)" if running else ""))
    if complete:
        for key, stale in (("collect", collect_stale), ("score", score_stale)):
            why = stale(sim)
            print(f"  {key:<8} {'current' if not why else why}{live[key]}")
            if why:
                break
        else:
            why = zip_stale(sim)
            print(f"  {'sim.zip':<8} {'current' if not why else why}")
    driver = jobs.latest.get("driver")
    if driver:
        print(f"  driver   job {driver['job_id']} "
              f"{'queued or running' if jobs.is_live('driver') else 'not live'}")


def main() -> int:
    args = parse_args()
    if args.command == "status":
        status(args)
        return 0
    if args.command == "start":
        return start(args)
    if args.command == "advance":
        return advance(args)
    return tick(args)


if __name__ == "__main__":
    sys.exit(main())
