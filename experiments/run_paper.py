"""Run the paper's whole analysis, from the downloads to the tables, and resume it.

Experiments 1-3 (``experiments/paper/exp01_baseline.json``, ``exp02_nullspace``,
``exp03_rollup``) go through every stage, each started once what it needs exists
and nothing for it is queued or running:

1. **prefetch** (``start`` only, on a login node: compute nodes are offline) --
   every file the later stages read, the national replicate files for the
   table trees included;
2. **assemble**, then **heldout**, then (for the roll-up) **trees** -- the
   inputs in the data directory, one job each;
3. **fits** -- per experiment, :func:`run_experiment.advance`: raking, MAP + VB,
   then HMC, topping the GPU queue up as slots free;
4. **score** -- per experiment, once its fits are complete:
   ``score_experiment.py --resume`` (so only calls without scores are run);
5. **combine** -- per experiment, once every call has scores;
6. **tables** -- ``paper_tables.py`` over the three experiments, into
   ``<root>/tables``.

Every stage looks at what exists, so the same call runs the analysis from an
empty data directory and results root, or resumes one part way, without redoing
what is done. A stage that has failed ``--max-attempts`` times stops the run
and asks for attention rather than resubmitting forever.

Usage, from the repository root::

    # on a login node: download, take the first steps, and start the driver job
    $CONDA_PREFIX/bin/python experiments/run_paper.py start
    # anywhere, any time
    $CONDA_PREFIX/bin/python experiments/run_paper.py status

``start`` ends by submitting ``run_paper.sbatch``, a small job on ``short``
that runs ``tick`` -- one :func:`advance` -- and, while work remains, submits
itself again to start ``--every`` minutes later. When everything is done it
submits a one-line job whose only purpose is the "done" e-mail; if a stage
needs attention it exits non-zero, which mails the failure. ``tick`` takes a
lock under ``<root>/run_paper``, so two drivers never submit the same work.

``--root`` and ``--data-dir`` choose the folders: new ones for a run from
scratch, the existing ones to resume. ``--dry-run`` prints what one ``advance``
would submit and changes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment as rx  # noqa: E402
import score_experiment as sx  # noqa: E402

EXPERIMENT_FILES = ["exp01_baseline.json", "exp02_nullspace.json", "exp03_rollup.json"]

#: Slurm options of the driver's own jobs (on top of the scripts' headers).
DATA_SLURM = ["--partition=campus", "--qos=campus", "--time=04:00:00", "--cpus-per-task=24"]
TREES_SLURM = ["--partition=short", "--qos=short", "--time=01:00:00", "--cpus-per-task=4",
               "--mem=32G"]
TABLES_SLURM = ["--partition=short", "--qos=short", "--time=01:00:00", "--mem=64G"]
NOTIFY = ["--partition=short", "--qos=short", "--time=00:01:00", "--mem=100M",
          "--account=acf-utk0011", "--mail-type=END", "--mail-user=nnagle@utk.edu"]

#: What tick() returns: everything done, waiting on jobs, or needing a person.
DONE, WAITING, ATTENTION = 0, 3, 1

#: A lock older than this (seconds) is taken to be left by a driver that died.
STALE_LOCK = 2 * 3600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["start", "tick", "advance", "status"])
    parser.add_argument("--root", type=Path, default=rx.DEFAULT_ROOT,
                        help=f"results root (default {rx.DEFAULT_ROOT})")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="data directory (default: the experiments' data_dir)")
    parser.add_argument("--tag", default="paper1", help="the scoring tag")
    parser.add_argument("--draws", type=int, default=4000, help="scoring draws per method")
    parser.add_argument("--every", type=int, default=30, help="minutes between driver runs")
    parser.add_argument("--max-gpu-jobs", type=int, default=6)
    parser.add_argument("--max-attempts", type=int, default=4,
                        help="submissions of one stage's job before asking for attention")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    return args


# -- the experiments and the data directory ------------------------------------


def experiments(args) -> list[dict]:
    specs = [rx.load_experiment(rx.EXPERIMENTS / "paper" / name) for name in EXPERIMENT_FILES]
    if args.data_dir is not None:
        for spec in specs:
            # Only a different directory replaces the JSON's: the same one spelled
            # differently would make an existing experiment folder refuse it.
            if args.data_dir.resolve() != rx.data_dir(spec).resolve():
                spec["data_dir"] = str(args.data_dir.resolve())
    if len({rx.data_dir(spec) for spec in specs}) != 1:
        raise SystemExit("the experiments name different data directories; pass --data-dir")
    if len({json.dumps(spec["area"], sort_keys=True) for spec in specs}) != 1:
        raise SystemExit("the experiments name different study areas")
    return specs


def inputs_root(spec: dict) -> Path:
    return rx.data_dir(spec) / "processed" / "inputs" / rx.area_slug(spec["area"])


def pumas(specs: list[dict]) -> list[str]:
    return sorted({puma for spec in specs for puma in spec["pumas"]})


def data_state(specs: list[dict]) -> dict[str, bool]:
    """Which of the data stages are complete for every PUMA."""
    from pmedm_vb.assemble.inputs import TREES_FILE

    root = inputs_root(specs[0])
    folders = [root / puma for puma in pumas(specs)]
    return {"assemble": all((f / "manifest.json").exists() for f in folders),
            "heldout": all((f / "heldout").exists() for f in folders),
            "trees": (not any(spec["model"].get("rollup") for spec in specs)
                      or all((f / TREES_FILE).exists() for f in folders))}


# -- the driver's own jobs ------------------------------------------------------


class Jobs:
    """The driver's jobs (data, combine, tables, itself), in ``<root>/run_paper/jobs.jsonl``."""

    def __init__(self, root: Path, dry_run: bool):
        self.dir = root / "run_paper"
        self.path = self.dir / "jobs.jsonl"
        self.dry_run = dry_run
        self.records = ([json.loads(line) for line in self.path.read_text().splitlines()
                         if line.strip()] if self.path.exists() else [])
        latest = {r["key"]: r for r in self.records}
        self.live = set(rx.active_jobs([r["job_id"] for r in latest.values()
                                        if r["job_id"] and r["job_id"] != "DRYRUN"]))
        self.latest = latest

    def is_live(self, key: str) -> bool:
        record = self.latest.get(key)
        return bool(record) and record["job_id"] in self.live

    def attempts(self, key: str) -> int:
        return sum(1 for r in self.records if r["key"] == key)

    def submit(self, key: str, command: list[str]) -> str:
        if self.dry_run:
            print("  would submit " + key + ": " + " ".join(command))
            return "DRYRUN"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"sbatch failed: {result.stderr.strip()}\n{' '.join(command)}")
        job_id = result.stdout.strip().split(";")[0]
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "key": key, "job_id": job_id,
                  "commit": rx.git_commit(), "command": command}
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        self.records.append(record)
        self.latest[key] = record
        self.live.add(job_id)
        print(f"  submitted {job_id}: {key}")
        return job_id

    def note(self, key: str) -> None:
        """Record a submission made elsewhere (scoring), so that it counts as an attempt."""
        if self.dry_run:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "key": key, "job_id": ""}
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        self.records.append(record)
        self.latest[key] = record

    def sbatch(self, key: str, spec: dict, script: str, args: list[str], slurm: list[str],
               after: str | None = None) -> str:
        command = ["sbatch", "--parsable",
                   f"--export=ALL,PMEDM_VB_DATA={rx.data_dir(spec)}",
                   f"--job-name=paper-{key.replace(':', '-')}",
                   f"--output={self.dir / 'logs' / 'slurm-%j.log'}", *slurm]
        if after:
            command.append(f"--dependency=afterok:{after}")
        return self.submit(key, command + [str(rx.EXPERIMENTS / script), *args])


# -- one pass ---------------------------------------------------------------------


class Attention(Exception):
    """A stage has failed too often, or cannot go on without a person."""


def guard(jobs: Jobs, key: str, args) -> None:
    if jobs.attempts(key) >= args.max_attempts:
        raise Attention(f"{key}: submitted {jobs.attempts(key)} times without finishing; "
                        f"see {jobs.dir / 'logs'}")


def advance_data(specs, jobs: Jobs, args) -> bool:
    """Submit the next data stage; whether the data directory is ready."""
    spec = specs[0]
    state = data_state(specs)
    area = rx.area_args(spec)
    if not state["assemble"]:
        if not jobs.is_live("assemble"):
            guard(jobs, "assemble", args)
            jobs.sbatch("assemble", spec, "run_map.sbatch",
                        ["assemble", *area, "--out", str(jobs.dir / "assemble")], DATA_SLURM)
        return False
    if not state["heldout"]:
        if not jobs.is_live("heldout"):
            guard(jobs, "heldout", args)
            jobs.sbatch("heldout", spec, "run_map.sbatch",
                        ["heldout", *area, "--out", str(jobs.dir / "heldout")], DATA_SLURM)
        return False
    if not state["trees"]:
        if not jobs.is_live("trees"):
            guard(jobs, "trees", args)
            a = spec["area"]
            jobs.sbatch("trees", spec, "run_python.sbatch",
                        ["table_trees.py", "--name", a["name"], "--state", a.get("state", "47"),
                         "--county", *a.get("counties", ["093"]), "--year", str(a["year"]),
                         "--span", str(a["span"]), "--write-inputs", rx.area_slug(a),
                         "--out", str(rx.data_dir(spec) / "table_trees")], TREES_SLURM)
        return False
    return True


def key_done(spec: dict, exp: Path, key: tuple) -> bool:
    """Whether the outputs of one run_experiment job key, (stage, PUMA, method), exist."""
    stage, puma, method = key
    if stage == "rake":
        paths = [rx.rake_output(spec, exp, puma, method)]
    elif stage == "fits":
        paths = rx.fit_outputs(spec, exp, puma)
    else:
        paths = rx.hmc_outputs(spec, exp, puma)
    return all(path.exists() for path in paths)


def exp_attempts(exp: Path) -> dict:
    counts: dict = {}
    for job in rx.read_jobs(exp):
        key = (job["stage"], job["puma"], job.get("method"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def advance_experiment(spec: dict, jobs: Jobs, args) -> bool:
    """Fits, then scoring, then combining, for one experiment; whether it is combined."""
    exp = args.root / spec["name"]
    if not rx.is_complete(spec, exp):
        over = {k: n for k, n in exp_attempts(exp).items() if n >= args.max_attempts}
        live = rx.active_for(exp)
        stuck = [k for k in over if k not in live and not key_done(spec, exp, k)]
        if stuck:
            raise Attention(f"{spec['name']}: {stuck} submitted {args.max_attempts}+ times "
                            f"without finishing; see {exp / 'logs'}")
        print(f"{spec['name']}: fits")
        rx.advance(spec, args.root, max_gpu_jobs=args.max_gpu_jobs, dry_run=args.dry_run)
        return False

    state = sx.scoring_state(exp, args.tag)
    if not state["exists"] or state["calls"]:
        if state["live"]:
            print(f"{spec['name']}: scoring -- {len(state['live'])} job(s) live")
            return False
        key = f"score:{spec['name']}"
        guard(jobs, key, args)
        print(f"{spec['name']}: scoring -- {sum(map(len, state['calls'].values()))} call(s) "
              f"without scores" if state["exists"] else f"{spec['name']}: scoring")
        score = argparse.Namespace(experiment=exp, tag=args.tag, draws=args.draws,
                                   allow_missing=False, dry_run=args.dry_run, combine=False,
                                   resume=True, max_jobs=None, no_combine=True)
        try:
            sx.submit(score)
        except SystemExit as error:
            raise Attention(f"{spec['name']}: scoring not submitted: {error}") from None
        jobs.note(key)
        return False
    if state["combined"]:
        return True
    key = f"combine:{spec['name']}"
    if state["live"] or jobs.is_live(key):
        print(f"{spec['name']}: combining")
        return False
    guard(jobs, key, args)
    jobs.sbatch(key, spec, "run_python.sbatch",
                ["score_experiment.py", str(exp), "--tag", args.tag, "--combine"], sx.COMBINE_SLURM)
    return False


def tables_current(specs, args) -> bool:
    out = args.root / "tables" / "tables.md"
    if not out.exists():
        return False
    summaries = [args.root / s["name"] / "scores" / args.tag / "all_scores.csv" for s in specs]
    if not all(path.exists() for path in summaries):
        return False
    return out.stat().st_mtime > max(path.stat().st_mtime for path in summaries)


def advance(args) -> int:
    """One pass over every stage; DONE, WAITING or ATTENTION."""
    specs = experiments(args)
    jobs = Jobs(args.root, args.dry_run)
    try:
        if not advance_data(specs, jobs, args):
            return WAITING
        combined = [advance_experiment(spec, jobs, args) for spec in specs]
        if not all(combined):
            return WAITING
        if tables_current(specs, args):
            print("done: every stage is complete; tables in", args.root / "tables")
            return DONE
        if not jobs.is_live("tables"):
            guard(jobs, "tables", args)
            jobs.sbatch("tables", specs[0], "run_python.sbatch",
                        ["paper_tables.py", "--root", str(args.root), "--tag", args.tag,
                         "--experiments", *[s["name"] for s in specs],
                         "--out", str(args.root / "tables")], TABLES_SLURM)
        return WAITING
    except Attention as error:
        print(f"NEEDS ATTENTION: {error}")
        return ATTENTION


# -- the driver job ----------------------------------------------------------------


class Lock:
    def __init__(self, root: Path):
        self.path = root / "run_paper" / "lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and time.time() - self.path.stat().st_mtime > STALE_LOCK:
            self.path.unlink()
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(f"{self.path} is held ({self.path.read_text().strip()}); "
                             "another driver is running") from None
        os.write(fd, f"{socket.gethostname()} pid {os.getpid()} {time.ctime()}\n".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def driver_args(args) -> list[str]:
    out = ["--root", str(args.root), "--tag", args.tag, "--draws", str(args.draws),
           "--every", str(args.every), "--max-gpu-jobs", str(args.max_gpu_jobs),
           "--max-attempts", str(args.max_attempts)]
    if args.data_dir is not None:
        out += ["--data-dir", str(args.data_dir.resolve())]
    return out


def schedule(args, jobs: Jobs, now: bool = False) -> None:
    """Submit the next driver job, ``--every`` minutes from now (or at once)."""
    command = ["sbatch", "--parsable", f"--output={jobs.dir / 'logs' / 'driver-%j.log'}"]
    if not now:
        command.append(f"--begin=now+{args.every}minutes")
    jobs.submit("driver", command + [str(rx.EXPERIMENTS / "run_paper.sbatch"), *driver_args(args)])


def tick(args) -> int:
    with Lock(args.root):
        result = advance(args)
        jobs = Jobs(args.root, args.dry_run)
        if result == WAITING:
            schedule(args, jobs)
        elif result == DONE:
            jobs.submit("notify", ["sbatch", "--parsable", *NOTIFY,
                                   f"--output={jobs.dir / 'logs' / 'done-%j.log'}",
                                   "--job-name=paper-done",
                                   "--wrap=echo the paper analysis is complete"])
    return result


def start(args) -> int:
    jobs = Jobs(args.root, args.dry_run)
    if jobs.is_live("driver"):
        raise SystemExit(f"a driver job is already queued or running: "
                         f"{jobs.latest['driver']['job_id']}")
    specs = experiments(args)
    os.environ["PMEDM_VB_DATA"] = str(rx.data_dir(specs[0]))
    from pmedm_vb.config import StudyArea
    from pmedm_vb.data.cache import OFFLINE_ENV

    import run_map

    os.environ.pop(OFFLINE_ENV, None)
    a = specs[0]["area"]
    area = StudyArea(name=a["name"], state=a.get("state", "47"),
                     counties=tuple(a.get("counties", ["093"])), year=a["year"], span=a["span"])
    if args.dry_run:
        print(f"would prefetch {area.slug} into {rx.data_dir(specs[0])}")
    else:
        rx.data_dir(specs[0]).mkdir(parents=True, exist_ok=True)
        args.root.mkdir(parents=True, exist_ok=True)
        run_map.run_prefetch(area)
    return tick(args) if not args.dry_run else advance(args)


def status(args) -> None:
    specs = experiments(args)
    jobs = Jobs(args.root, dry_run=True)
    print(f"root {args.root}\ndata {rx.data_dir(specs[0])}")
    state = data_state(specs)
    for stage in ("assemble", "heldout", "trees"):
        live = " (job live)" if jobs.is_live(stage) else ""
        print(f"  {stage:<9} {'done' if state[stage] else 'not done'}{live}")
    for spec in specs:
        exp = args.root / spec["name"]
        fits = rx.is_complete(spec, exp)
        live = len(rx.active_for(exp))
        line = f"  {spec['name']:<16} fits {'done' if fits else 'not done'}" + (
            f" ({live} job(s) live)" if live else "")
        if fits:
            s = sx.scoring_state(exp, args.tag)
            if not s["exists"]:
                line += "; scoring not started"
            else:
                left = sum(map(len, s["calls"].values()))
                line += (f"; scoring {left} call(s) left" if left else "; scored") + (
                    f" ({len(s['live'])} job(s) live)" if s["live"] else "")
                line += "; combined" if s["combined"] else ""
        print(line)
    print(f"  tables    {'current' if tables_current(specs, args) else 'not current'}")
    driver = jobs.latest.get("driver")
    if driver:
        print(f"  driver    job {driver['job_id']} {'queued or running' if jobs.is_live('driver') else 'not live'}")


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
