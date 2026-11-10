"""Run several experiments' bundles across GPU partitions: each call submits what is left, safe to repeat.

``run_experiment.py`` submits one experiment's bundle jobs to the one
partition its JSON names. This driver takes several experiment JSONs, in
priority order, and spreads their unfinished PUMAs over the partitions of a
queue configuration, by default:

- **ai-tenn-debug** (an H100, about 1.7x a V100): ``--partition=ai-tenn-debug
  --qos=ai-tenn-debug --gpus=1 --time=03:00:00``; at most 2 of the user's jobs
  submitted there and 1 running (the QoS's per-user limits; MaxWall 3 h);
- **campus-gpu** (a V100S): ``--partition=campus-gpu --qos=campus-gpu
  --gpus=v100s:1 --time=03:00:00``; at most 6 submitted and 3 running (seen,
  not documented: make them what the QoS says, ``sacctmgr show qos campus-gpu``);

and ``--account=isaac-utk0533`` for every job (:data:`DEFAULT_QUEUE`).

Usage, from the repository root on a login node::

    $CONDA_PREFIX/bin/python experiments/run_queue.py experiments/tn/tn08_long_state_prior3.json \\
        experiments/tn/tn09_long_prior3_decor2.json --root $R
    $CONDA_PREFIX/bin/python experiments/run_queue.py <the same JSONs> --root $R --status
    $CONDA_PREFIX/bin/python experiments/run_queue.py <the same JSONs> --root $R --dry-run
    $CONDA_PREFIX/bin/python experiments/run_queue.py <the same JSONs> --root $R --watch 20

**Each call**, for each partition, counts the user's jobs queued or running in
it (``squeue -u $USER -p <partition>``, every job, whatever submitted it), so
that its free submit slots are ``max_submitted`` less those. Then, experiment
by experiment in the order given, it takes the PUMAs a new bundle job should
take (``run_experiment.bundle_todo``: unfinished, not in a live bundle job of
the experiment, not claimed by a live job, not failed twice), splits them into
jobs of ``per_job`` PUMAs (default: the experiment's grid split as its
``bundle.jobs`` splits it, so a bundle holds as many PUMAs as
``run_experiment.py`` would put in it) and submits each job to the first
partition in the configuration's order with a free submit slot: so
ai-tenn-debug while it has one, then campus-gpu. A job waiting in
ai-tenn-debug behind the one running there is still preferred to campus-gpu,
as asked; reorder the partitions or lower ``max_submitted`` to change that.
When no partition has a slot the rest waits for a later call. An ``sbatch``
refused (a QoS limit the configuration does not know, say) closes that
partition for the call, and the job goes to the next.

**Repeating is safe.** A PUMA in a live bundle job of the experiment (its
record in ``jobs.jsonl``, asked of ``squeue`` by job id) or claimed by a live
job is not submitted again, whichever partition or script submitted it; the
claims (``run_puma_bundle.py``) keep two jobs off one PUMA even if a record
were lost. A bundle job is given ``--signal=B:USR1@600``: ten minutes before
its time limit it starts no new PUMA, and at the limit it releases its claims.
So a long HMC run that outlives the 3 h limit stops there, and the next call
submits its PUMA again, to whichever partition is free; the sampler resumes
from its last checkpoint (``hmc.<kind>.checkpoint`` iterations; the work since
it is lost) and the finished fits are skipped.

**A job's options**: the experiment's ``slurm.bundle`` options named in
``keep`` (``--mem``, ``--cpus-per-task``, ...: a statewide support needs its
64G), then the chosen partition's ``sbatch`` options in place of any with the
same key, then the account; the job's record in ``jobs.jsonl`` names its
``partition``.

**The configuration** is :data:`DEFAULT_QUEUE`, changed by ``--queue FILE``
(JSON with any of its keys; ``partitions``, if given, is the list of
partitions in order of preference, each entry merged over the default of the
same name) and then by the flags ``--max-submitted NAME=N``,
``--max-running NAME=N``, ``--sbatch NAME='--time=02:00:00 ...'``,
``--partitions NAME ...`` (use only these, in this order), ``--account`` and
``--per-job EXPERIMENT=N``. ``max_running`` is reported, not enforced: a job
beyond it waits in the partition's queue.

``--status`` says, per experiment and PUMA: done; running or queued, on which
partition, in which job; claimed by another job; failed; or waiting for a
slot; with each partition's jobs against its limits. ``--dry-run`` prints the
``sbatch`` commands and work files of one call without submitting or writing
anything. ``--watch MINUTES`` repeats the call every ``MINUTES`` until every
PUMA is done (or failed twice: ``run_experiment.py --retry-failed`` clears
that); an error in one round is printed and the next round tries again, so it
can be left running in ``tmux`` on a login node.

**The driver job** (``--start``), as ``run_paper.py`` does: a small job on a
CPU partition (``--driver-sbatch``, by default ``short``, 30 minutes, 1 core)
runs one call (``--tick``) and, while work remains, submits itself again to
start ``--every`` minutes later (``sbatch --begin=now+<every>minutes``), so
nothing has to keep running on a login node and the prompt is free at once.
The driver's own jobs are recorded in ``<root>/run_queue/jobs.jsonl`` and their
logs go to ``<root>/run_queue/logs``. A lock under ``<root>/run_queue`` keeps
two calls (a driver job and one by hand) from submitting at the same time.
``--stop`` cancels the driver job; ``--status`` also says whether one is queued
or running. A call that fails (``squeue`` or the file system) still schedules
the next one and exits non-zero, which mails the failure; after
:data:`MAX_TICK_FAILURES` failed calls in a row the driver stops rescheduling.
The driver job passes on the experiments and every queue option it was started
with, so ``--start`` with the same arguments as a hand call does the same work::

    $CONDA_PREFIX/bin/python experiments/run_queue.py <the JSONs> --root $R --start
    $CONDA_PREFIX/bin/python experiments/run_queue.py <the JSONs> --root $R --status
    $CONDA_PREFIX/bin/python experiments/run_queue.py <the JSONs> --root $R --stop

Only experiments with a ``bundle`` block are run here (their fits and HMC go
in bundle jobs); raking and pilots stay with ``run_experiment.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx  # noqa: E402
import run_puma_bundle as rpb  # noqa: E402

#: The queue configuration (see the module docstring). ``keep``: the options of
#: an experiment's ``slurm.bundle`` a job keeps; ``per_job``: PUMAs per job by
#: experiment name, where not the experiment's own split.
DEFAULT_QUEUE = {
    "account": "isaac-utk0533",
    "keep": ["--mem", "--mem-per-cpu", "--mem-per-gpu", "--cpus-per-task"],
    "per_job": {},
    "partitions": [
        {"name": "ai-tenn-debug", "max_submitted": 2, "max_running": 1,
         "sbatch": ["--partition=ai-tenn-debug", "--qos=ai-tenn-debug", "--gpus=1",
                    "--time=03:00:00"]},
        {"name": "campus-gpu", "max_submitted": 6, "max_running": 3,
         "sbatch": ["--partition=campus-gpu", "--qos=campus-gpu", "--gpus=v100s:1",
                    "--time=03:00:00"]},
    ],
}


#: The driver job's sbatch options (``--driver-sbatch`` replaces them).
DRIVER_SBATCH = "--partition=short --qos=short --time=00:30:00 --cpus-per-task=1 --mem=4G"
#: Failed driver calls in a row after which it stops rescheduling itself.
MAX_TICK_FAILURES = 3
#: A lock older than this (seconds) is taken to be left by a killed call.
STALE_LOCK = 2 * 3600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", type=Path, nargs="+",
                        help="experiment JSON files, highest priority first")
    parser.add_argument("--root", type=Path, default=rx.DEFAULT_ROOT,
                        help=f"folder holding the experiment folders (default {rx.DEFAULT_ROOT})")
    parser.add_argument("--queue", type=Path, default=None,
                        help="a JSON file of queue settings over the defaults")
    parser.add_argument("--max-submitted", action="append", default=[], metavar="NAME=N",
                        help="the user's jobs allowed queued or running in a partition")
    parser.add_argument("--max-running", action="append", default=[], metavar="NAME=N",
                        help="the user's jobs allowed running in a partition (reported)")
    parser.add_argument("--sbatch", action="append", default=[], metavar="NAME=OPTIONS",
                        help="sbatch options replacing those of the same key in a partition's, "
                             "e.g. \"campus-gpu=--time=06:00:00\"")
    parser.add_argument("--partitions", nargs="+", default=None, metavar="NAME",
                        help="use only these partitions, in this order of preference")
    parser.add_argument("--account", default=None, help="the account of every job")
    parser.add_argument("--per-job", action="append", default=[], metavar="EXPERIMENT=N",
                        help="PUMAs per bundle job for an experiment (by name)")
    parser.add_argument("--status", action="store_true", help="report; submit nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what one call would submit; write nothing")
    parser.add_argument("--watch", type=float, default=None, metavar="MINUTES",
                        help="repeat every MINUTES until every PUMA is done, in this terminal")
    parser.add_argument("--start", action="store_true",
                        help="submit the driver job, which repeats every --every minutes "
                             "until every PUMA is done, and return")
    parser.add_argument("--stop", action="store_true", help="cancel the driver job")
    parser.add_argument("--tick", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--every", type=float, default=20.0, metavar="MINUTES",
                        help="the driver job's interval (default 20)")
    parser.add_argument("--driver-sbatch", default=DRIVER_SBATCH, metavar="OPTIONS",
                        help="the driver job's sbatch options, given with '=' since they "
                             f"start with '--' (default --driver-sbatch=\"{DRIVER_SBATCH}\")")
    args = parser.parse_args()
    if sum(map(bool, (args.start, args.stop, args.tick, args.status,
                      args.watch is not None))) > 1:
        parser.error("--start, --stop, --status and --watch are alternatives")
    return args


# -- the configuration ----------------------------------------------------------


def pairs(values: list[str], flag: str) -> list[tuple[str, str]]:
    """``NAME=VALUE`` arguments as pairs."""
    out = []
    for value in values:
        name, sep, rest = value.partition("=")
        if not sep or not name:
            raise SystemExit(f"{flag} {value!r}: expected NAME=VALUE")
        out.append((name, rest))
    return out


def queue_config(args: argparse.Namespace) -> dict:
    """:data:`DEFAULT_QUEUE`, then ``--queue``'s file, then the flags."""
    queue = copy.deepcopy(DEFAULT_QUEUE)
    if args.queue is not None:
        given = json.loads(args.queue.read_text())
        defaults = {p["name"]: p for p in queue["partitions"]}
        for key, value in given.items():
            if key == "partitions":
                queue["partitions"] = [{**defaults.get(p["name"], {}), **p} for p in value]
            elif key not in DEFAULT_QUEUE:
                raise SystemExit(f"{args.queue}: unknown key {key!r} (known: "
                                 f"{', '.join(DEFAULT_QUEUE)})")
            else:
                queue[key] = value
    by_name = {p["name"]: p for p in queue["partitions"]}

    def partition(name: str, flag: str) -> dict:
        if name not in by_name:
            raise SystemExit(f"{flag} {name}: no such partition in the configuration "
                             f"({', '.join(by_name)})")
        return by_name[name]

    for name, value in pairs(args.max_submitted, "--max-submitted"):
        partition(name, "--max-submitted")["max_submitted"] = int(value)
    for name, value in pairs(args.max_running, "--max-running"):
        partition(name, "--max-running")["max_running"] = int(value)
    for name, value in pairs(args.sbatch, "--sbatch"):
        entry = partition(name, "--sbatch")
        entry["sbatch"] = rx.override(entry.get("sbatch", []), value.split())
    if args.partitions is not None:
        queue["partitions"] = [partition(name, "--partitions") for name in args.partitions]
    if args.account is not None:
        queue["account"] = args.account
    for name, value in pairs(args.per_job, "--per-job"):
        queue["per_job"][name] = int(value)
    for entry in queue["partitions"]:
        for key in ("max_submitted", "sbatch"):
            if key not in entry:
                raise SystemExit(f"partition {entry['name']}: no {key}")
    return queue


# -- the partitions -------------------------------------------------------------


class Partitions:
    """The partitions' free submit slots for one call, in order of preference."""

    def __init__(self, queue: dict):
        self.entries = queue["partitions"]
        self.free: dict[str, int] = {}
        self.counts: dict[str, tuple[int, int] | None] = {}
        for entry in self.entries:
            jobs = rx.user_jobs(entry["name"])
            if jobs is None:
                self.counts[entry["name"]] = None
                self.free[entry["name"]] = 0  # squeue did not answer: submit nothing there
                continue
            running = sum(1 for _, state in jobs if state == "RUNNING")
            self.counts[entry["name"]] = (len(jobs), running)
            self.free[entry["name"]] = max(0, int(entry["max_submitted"]) - len(jobs))

    def lines(self) -> list[str]:
        """Each partition's jobs against its limits."""
        out = []
        for entry in self.entries:
            counts = self.counts[entry["name"]]
            if counts is None:
                out.append(f"  {entry['name']:<14} squeue did not answer; nothing submitted there")
                continue
            out.append(f"  {entry['name']:<14} {counts[0]} of {entry['max_submitted']} submitted, "
                       f"{counts[1]} of {entry.get('max_running', '?')} running; "
                       f"{self.free[entry['name']]} free slot(s)")
        return out

    def first_free(self) -> dict | None:
        """The preferred partition with a free submit slot (none once it has refused a job)."""
        for entry in self.entries:
            if self.free[entry["name"]] > 0:
                return entry
        return None


# -- one experiment -------------------------------------------------------------


def per_job(spec: dict, queue: dict) -> int:
    """PUMAs per bundle job: the queue's ``per_job`` for the experiment, else its
    grid split over ``bundle.jobs`` jobs, as run_experiment.py splits it."""
    if spec["name"] in queue["per_job"]:
        return max(1, int(queue["per_job"][spec["name"]]))
    pumas = len(rx.grid(spec)["pumas"])
    jobs = max(1, min(int(spec["bundle"].get("jobs", 1)), pumas))
    return max(1, math.ceil(pumas / jobs))


def job_options(spec: dict, queue: dict, entry: dict) -> list[str]:
    """A bundle job's ``sbatch`` options on the partition ``entry``: the experiment's
    ``slurm.bundle`` options named in ``keep``, the partition's options over them,
    then the time-limit signal (:func:`run_experiment.with_signal`)."""
    own = rx.slurm_options(spec, "bundle", rx.FITS_SLURM)
    keep = set(queue["keep"])
    kept = [o for o in own if o.split("=", 1)[0] in keep]
    return rx.with_signal(rx.override(kept, list(entry["sbatch"])))


def job_partition(job: dict) -> str:
    """The partition a recorded job was submitted to: its ``partition``, else its
    command's ``--partition`` (a job ``run_experiment.py`` submitted)."""
    if job.get("partition"):
        return job["partition"]
    for part in job.get("command", []):
        if part.startswith("--partition="):
            return part.split("=", 1)[1]
    return "?"


class Experiment:
    """One experiment of the queue: its spec, folder and what is left of it."""

    def __init__(self, path: Path, root: Path):
        self.path = path
        self.spec = rx.load_experiment(path)
        self.name = self.spec["name"]
        self.exp = root / self.name

    def unfinished(self) -> list[str]:
        """The PUMAs not complete and not failed."""
        return [p for p in self.spec["pumas"] if not rx.puma_complete(self.spec, self.exp, p)
                and not rpb.is_failed(self.exp, p)]

    def failed(self) -> list[str]:
        return [p for p in self.spec["pumas"] if not rx.puma_complete(self.spec, self.exp, p)
                and rpb.is_failed(self.exp, p)]

    def advance(self, queue: dict, partitions: Partitions, dry_run: bool) -> tuple[int, int]:
        """Submit this experiment's waiting PUMAs to free partitions; the jobs
        submitted and the jobs left for a later call."""
        spec, exp = self.spec, self.exp
        print(f"== {self.name} ({self.path})")
        if not spec.get("bundle"):
            print("  no bundle block: run it with run_experiment.py")
            return 0, 0
        problems = rx.check_inputs(spec)
        if problems:
            print("  the data directory is not ready:\n    " + "\n    ".join(problems))
            if not dry_run:
                return 0, 0
        try:
            rx.prepare(exp, spec, dry_run)
        except SystemExit as error:
            print(f"  not run: {error}")
            return 0, 0
        todo = rx.bundle_todo(spec, exp, rx.active_for(exp), prefix="  bundles")
        if not todo:
            done = sum(rx.puma_complete(spec, exp, p) for p in spec["pumas"])
            print(f"  nothing to submit ({done} of {len(spec['pumas'])} PUMA(s) done)")
            return 0, 0
        size = per_job(spec, queue)
        jobs = math.ceil(len(todo) / size)
        groups = [todo[k::jobs] for k in range(jobs)]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        submitted = 0
        for k, group in enumerate(groups):
            while True:
                entry = partitions.first_free()
                if entry is None:
                    left = len(groups) - k
                    print(f"  no partition has a free slot; {left} job(s) "
                          f"({sum(len(g) for g in groups[k:])} PUMA(s)) wait for a later call")
                    return submitted, left
                sub = rx.Submitter(exp, spec, dry_run, [f"--account={queue['account']}"])
                print(f"  bundle {k + 1}/{len(groups)} on {entry['name']}: {len(group)} PUMA(s) "
                      f"{' '.join(group)}")
                try:
                    rx.submit_bundle(sub, spec, exp, group, f"{stamp}-{k + 1}",
                                     job_options(spec, queue, entry),
                                     extra={"partition": entry["name"]})
                except SystemExit as error:
                    print(f"  {entry['name']} refused it, closed for this call: {error}")
                    partitions.free[entry["name"]] = 0
                    continue
                partitions.free[entry["name"]] -= 1
                submitted += 1
                break
        return submitted, 0

    def status(self) -> list[str]:
        """Per PUMA: done, in a live job (where), claimed, failed, or waiting."""
        spec, exp = self.spec, self.exp
        active = rx.active_for(exp)
        live = rx.bundled(active)
        claims = rpb.claim_states(exp)
        done = sum(rx.puma_complete(spec, exp, p) for p in spec["pumas"])
        lines = [f"== {self.name}  (level {rx.level(spec)}; {done} of {len(spec['pumas'])} "
                 f"PUMA(s) done; {exp})"]
        for puma in spec["pumas"]:
            fit_paths = rx.fit_outputs(spec, exp, puma)
            reports = rx.hmc_outputs(spec, exp, puma) if spec["hmc"] else []
            progress = (f"fits {sum(p.exists() for p in fit_paths)}/{len(fit_paths)}, "
                        f"hmc {sum(p.exists() for p in reports)}/{len(reports)}")
            if rx.puma_complete(spec, exp, puma):
                where = "done"
            elif puma in live:
                job = live[puma]
                state = job["state"].lower()
                where = (f"{'running' if state == 'running' else state} on "
                         f"{job_partition(job)} (job {job['job_id']})")
            elif puma in claims and not claims[puma]["stale"]:
                claim = claims[puma]
                where = (f"claimed by job {claim.get('job_id', '?')} on {claim.get('host', '?')} "
                         f"(heartbeat {rpb.age_text(claim['age'])} ago)")
            elif rpb.is_failed(exp, puma):
                where = f"FAILED {rpb.MAX_FAILURES} times (run_experiment.py --retry-failed)"
            else:
                where = "waiting for a slot"
            checkpoint = spec["hmc"] and any(
                (out / f"{rx.fit_name(spec, puma, alpha)}_state.npz").exists()
                for _, alpha, out in rx.hmc_runs(spec, exp, puma))
            lines.append(f"  {puma:<14} {where}  [{progress}"
                         + ("; HMC checkpoint" if checkpoint and where != "done" else "") + "]")
        return lines


# -- the driver -----------------------------------------------------------------


def one_call(experiments: list[Experiment], queue: dict, dry_run: bool) -> tuple[int, bool]:
    """Submit what is left, experiment by experiment; the jobs submitted and
    whether any PUMA is still unfinished (failed ones aside)."""
    partitions = Partitions(queue)
    print("partitions (in order of preference):")
    print("\n".join(partitions.lines()))
    total = 0
    for experiment in experiments:
        submitted, _ = experiment.advance(queue, partitions, dry_run)
        total += submitted
    left = any(experiment.unfinished() for experiment in experiments)
    return total, left


def report(experiments: list[Experiment], queue: dict) -> None:
    """--status: the partitions, then every experiment's PUMAs."""
    partitions = Partitions(queue)
    print("partitions (in order of preference):")
    print("\n".join(partitions.lines()))
    for experiment in experiments:
        print("\n".join(experiment.status()))


# -- the driver job --------------------------------------------------------------


class Lock:
    """``<root>/run_queue/lock``: one call submits at a time."""

    def __init__(self, root: Path) -> None:
        self.path = root / "run_queue" / "lock"

    def __enter__(self) -> "Lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and time.time() - self.path.stat().st_mtime > STALE_LOCK:
            self.path.unlink()
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise SystemExit(f"{self.path} is held ({self.path.read_text().strip()}); another "
                             "call is submitting -- try again in a minute") from None
        os.write(fd, f"{socket.gethostname()} pid {os.getpid()} {time.ctime()}\n".encode())
        os.close(fd)
        return self

    def __exit__(self, *exc) -> None:
        self.path.unlink(missing_ok=True)


class Driver:
    """The driver's records, ``<root>/run_queue/jobs.jsonl``: its jobs (``job``)
    and the outcome of each call it ran (``tick``)."""

    def __init__(self, root: Path) -> None:
        self.dir = root / "run_queue"
        self.path = self.dir / "jobs.jsonl"
        self.records = ([json.loads(line) for line in self.path.read_text().splitlines()
                         if line.strip()] if self.path.exists() else [])

    def append(self, record: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}
        with open(self.path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        self.records.append(record)

    def live(self) -> dict[str, str]:
        """The driver's jobs queued or running: job id -> state."""
        ids = [r["job_id"] for r in self.records if r.get("kind") == "job"]
        return rx.active_jobs(ids) if ids else {}

    def failures_in_a_row(self) -> int:
        count = 0
        for record in reversed([r for r in self.records if r.get("kind") == "tick"]):
            if record["ok"]:
                break
            count += 1
        return count

    def last_tick(self) -> dict | None:
        ticks = [r for r in self.records if r.get("kind") == "tick"]
        return ticks[-1] if ticks else None


def passthrough(args: argparse.Namespace) -> list[str]:
    """The experiments and queue options, for the driver job's own call."""
    out = [str(p.resolve()) for p in args.experiments] + ["--root", str(args.root.resolve())]
    if args.queue is not None:
        out += ["--queue", str(args.queue.resolve())]
    for flag, values in (("--max-submitted", args.max_submitted),
                         ("--max-running", args.max_running), ("--sbatch", args.sbatch),
                         ("--per-job", args.per_job)):
        for value in values:
            out.append(f"{flag}={value}")  # "=": a value may itself start with "--"
    if args.partitions:
        out += ["--partitions", *args.partitions]
    if args.account:
        out += ["--account", args.account]
    out += ["--every", f"{args.every:g}", f"--driver-sbatch={args.driver_sbatch}"]
    return out


def schedule(args: argparse.Namespace, driver: Driver, now: bool) -> str:
    """Submit the driver job: at once (``--start``) or ``--every`` minutes from now."""
    account = args.account or queue_config(args)["account"]
    logs = driver.dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = ["sbatch", "--parsable", f"--account={account}", "--job-name=run-queue",
               f"--output={logs / 'driver-%j.log'}", *shlex.split(args.driver_sbatch)]
    if not now:
        command.append(f"--begin=now+{max(1, round(args.every))}minutes")
    command += [str(rx.EXPERIMENTS / "run_queue.sbatch"), *passthrough(args), "--tick"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"sbatch failed: {result.stderr.strip()}\n{' '.join(command)}")
    job_id = result.stdout.strip().split(";")[0]
    driver.append({"kind": "job", "job_id": job_id, "command": command})
    print(f"driver job {job_id} submitted" + ("" if now else f", to start in {args.every:g} min"))
    return job_id


def tick(args: argparse.Namespace, experiments: list, queue: dict) -> int:
    """One call as the driver job; then the next job while work remains."""
    driver = Driver(args.root)
    ok, left = False, True
    try:
        with Lock(args.root):
            count, left = one_call(experiments, queue, False)
            print(f"{count} job(s) submitted", flush=True)
        ok = True
    except (Exception, SystemExit):
        traceback.print_exc()
    driver.append({"kind": "tick", "ok": ok, "left": bool(left),
                   "job_id": os.environ.get("SLURM_JOB_ID", "")})
    if ok and not left:
        failed = {e.name: e.failed() for e in experiments if e.failed()}
        print("every PUMA is done" + (f", but failed: {failed} (run_experiment.py "
                                      "--retry-failed, then --start again)" if failed else "")
              + "; the driver stops")
        return 0
    if driver.failures_in_a_row() >= MAX_TICK_FAILURES:
        print(f"{MAX_TICK_FAILURES} calls in a row failed; the driver stops (see the logs in "
              f"{driver.dir / 'logs'}, then --start again)")
        return 1
    schedule(args, driver, now=False)
    return 0 if ok else 1


def driver_status(args: argparse.Namespace) -> None:
    driver = Driver(args.root)
    live = driver.live()
    last = driver.last_tick()
    line = ("driver: " + (", ".join(f"job {j} {s}" for j, s in live.items()) if live
                          else "no driver job queued or running"))
    if last:
        line += (f"; last call {last['time']} " + ("ok" if last["ok"] else "FAILED")
                 + ("" if last.get("left", True) else ", nothing left"))
    print(line)


def main() -> None:
    args = parse_args()
    queue = queue_config(args)
    experiments = [Experiment(path, args.root) for path in args.experiments]
    names = [e.name for e in experiments]
    if len(set(names)) != len(names):
        raise SystemExit(f"an experiment is named twice: {names}")
    if args.status:
        driver_status(args)
        report(experiments, queue)
        return
    if args.stop:
        live = Driver(args.root).live()
        if not live:
            print("no driver job queued or running")
            return
        subprocess.run(["scancel", *live], check=False)
        print(f"cancelled driver job(s) {', '.join(live)}")
        return
    if args.tick:
        sys.exit(tick(args, experiments, queue))
    if args.start:
        driver = Driver(args.root)
        live = driver.live()
        if live:
            raise SystemExit(f"a driver job is already queued or running: {', '.join(live)} "
                             "(--stop first)")
        if args.dry_run:
            print("would submit the driver job: run_queue.sbatch " + " ".join(passthrough(args)))
            return
        schedule(args, driver, now=True)
        return
    if args.watch is None or args.dry_run:
        with (Lock(args.root) if not args.dry_run else _nolock()):
            count, left = one_call(experiments, queue, args.dry_run)
        print(f"{count} job(s) {'would be ' if args.dry_run else ''}submitted"
              + ("" if left else "; every PUMA is done or failed") + "; see --status")
        return
    while True:
        print(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        try:
            with Lock(args.root):
                count, left = one_call(experiments, queue, False)
            print(f"{count} job(s) submitted", flush=True)
            if not left:
                failed = {e.name: e.failed() for e in experiments if e.failed()}
                print("every PUMA is done" + (f", but failed: {failed} (run_experiment.py "
                                              "--retry-failed, then watch again)" if failed else ""))
                return
        except (Exception, SystemExit):  # a round that fails (squeue, the file system, the lock)
            traceback.print_exc()
        sys.stdout.flush()
        try:
            time.sleep(max(1.0, args.watch * 60))
        except KeyboardInterrupt:
            print("stopped")
            return


class _nolock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


if __name__ == "__main__":
    main()
