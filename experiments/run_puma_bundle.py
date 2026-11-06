"""Run many PUMAs in one job: MAP, the VB fits and their HMC runs, one PUMA per GPU at a time.

Written for queues where a job waits days but a PUMA takes minutes, so that a
few jobs carry a whole study area. ``run_experiment.py`` writes the work file
and submits it with ``run_puma_bundle.sbatch``; by hand::

    $CONDA_PREFIX/bin/python experiments/run_puma_bundle.py WORK.json

The work file (JSON) is a **bundle**, a fixed list of PUMAs::

    {"name": AREA_NAME, "fits_args": [run_map.py options shared by every PUMA],
     "families": ["gaussian"], "draws": 64, "device": "cuda",
     "experiment": "<root>/<experiment name>",
     "pumas": [{"puma": "4701502", "out": ".../tn01_gpu/4701502",
                "fit_outputs": [every fit's .npz],
                "hmc": [{"out": ".../hmc_short", "args": [run_mcmc.py options],
                         "report": ".../hmc_short/<name>_report.txt"}]}, ...]}

or a **pilot**: the same keys with ``"spec"`` (the experiment as
``run_experiment.py`` loaded it) in place of ``"pumas"``. A pilot has no list:
each of its streams repeatedly takes the unfinished PUMA, not failed and not
claimed by another job, that is estimated slowest (the most zones x units in
its ``manifest.json``), builds that PUMA's entry with
:func:`run_experiment.bundle_item`, and runs it; it exits when none is left,
so a pilot that starts after the bundles have done everything exits at once.
A pilot tries each PUMA at most once.

**One stream per GPU.** The GPUs are those Slurm gives the job
(``CUDA_VISIBLE_DEVICES``); each stream takes the next PUMA and runs it with
that one GPU visible, so the fits and the sampler see it as ``cuda``. The
job's cores are split evenly between the streams. With no GPU, one stream
runs on the CPU.

**Per PUMA**, in order: ``run_map.py solve`` (the MAP, into ``<out>/map``),
``run_map.py vb`` for each family (into ``<out>/vb_<family>``), then for each
HMC run ``run_mcmc.py sample`` and ``report``. Finished work is skipped: the
fits whose ``.npz`` exists (run_map.py's own check) and the HMC runs whose
report exists; an unfinished sampler resumes from its checkpoint. Those files
appear whole or not at all (written aside and renamed), so a killed job leaves
nothing that looks finished and is not. A PUMA whose fits fail gets no HMC; a
failure does not stop the other PUMAs, and the job exits non-zero if any PUMA
failed. Each PUMA's output goes to ``<out>/bundle.log``; this script's own log
says which stream ran which PUMA, and for how long.

**Claims.** So that several jobs can work on one experiment at once (short
bundle jobs and a long pilot), a job runs a PUMA only while it holds the
PUMA's claim, the directory ``<experiment>/claims/<puma>/``: ``os.mkdir`` is
atomic, also on Lustre, so of two jobs only one creates it. It holds
``claim.json`` (Slurm job id, or ``local-<host>-<pid>`` outside Slurm; host;
start time; attempt; a random token) and a ``heartbeat`` file, touched every
``PMEDM_VB_HEARTBEAT`` seconds (300) while the job holds the claim. A claim is
**stale** when its job is not running or pending in ``squeue`` (asked by job
id; when ``squeue`` cannot answer, the heartbeat alone decides) or its
heartbeat is older than ``PMEDM_VB_CLAIM_STALE`` seconds (1800). A stale claim
is taken over by renaming it to a unique ``claims/.stale-<puma>-<uuid>`` --
only one renamer succeeds -- and creating a fresh one; a job whose claim was
taken over (it finds another token, or none, at its next heartbeat) stops that
PUMA. A bundle skips the PUMAs another live job holds.

**Failures.** On success the claim is removed. A PUMA that fails has a line
(time, job, host, failed steps, log) appended to
``<experiment>/failures/<puma>.jsonl`` -- outside the claim, so it outlives
it -- and its claim is released. After :data:`MAX_FAILURES` the PUMA is
*failed*: no job takes it until ``run_experiment.py --retry-failed`` clears
its record. A job killed or out of time is not a failure: its PUMA resumes
in the next job.

**Signals.** Jobs are submitted with ``--signal=B:USR1@600`` and the batch
script ``exec``-s this one, so ten minutes before the time limit it gets
``USR1``: it takes no new PUMA and lets the current ones run on. At the limit
Slurm sends ``SIGTERM``: it stops its PUMAs' processes, releases its claims
and exits, so the next job need not wait for them to go stale.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")

#: Seconds between a claim's heartbeats, and the heartbeat age that makes it stale.
HEARTBEAT = float(os.environ.get("PMEDM_VB_HEARTBEAT", "300"))
STALE_AFTER = float(os.environ.get("PMEDM_VB_CLAIM_STALE", "1800"))
#: Failures after which a PUMA is no longer taken.
MAX_FAILURES = 2
#: Job states (``squeue``) in which a claim's job still counts as holding it.
LIVE_STATES = {"PENDING", "CONFIGURING", "RUNNING", "SUSPENDED"}
#: Seconds the driver gives its streams to wind down after SIGTERM.
TERM_GRACE = 20


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {message}", flush=True)


def streams() -> list[str | None]:
    """The GPU ids to run on, one stream each: ``CUDA_VISIBLE_DEVICES``, else one CPU stream."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpus = [g for g in visible.split(",") if g.strip()] if visible else []
    return gpus or [None]


# -- claims and failures ------------------------------------------------------


def job_identity() -> str:
    """This job's id for claims: ``SLURM_JOB_ID``, else ``local-<host>-<pid>``."""
    return os.environ.get("SLURM_JOB_ID") or f"local-{socket.gethostname()}-{os.getpid()}"


def job_states(job_ids) -> dict[str, str] | None:
    """``{job id: state}`` for the Slurm job ids ``squeue`` knows; ``None`` when it
    cannot answer (no ``squeue``, a timeout, the controller down), so that the
    caller falls back on heartbeats rather than take every claim for stale.
    A job ``squeue`` does not list has ended; with every id unknown it exits
    non-zero with "Invalid job id", which is an answer too."""
    ids = sorted({j for j in job_ids if j and j.isdigit()})
    if not ids:
        return {}
    try:
        result = subprocess.run(["squeue", "-h", "-o", "%A %T", "-j", ",".join(ids)],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 and "invalid job id" not in result.stderr.lower():
        return None
    return dict(line.split()[:2] for line in result.stdout.splitlines() if len(line.split()) >= 2)


def claims_dir(exp: Path) -> Path:
    return Path(exp) / "claims"


def read_claim(path: Path) -> dict | None:
    """A claim's ``claim.json`` plus ``age``, seconds since its last sign of life
    (heartbeat, ``claim.json`` or the directory, whichever is newest); ``None``
    when there is no claim. A claim whose ``claim.json`` is missing or torn (its
    job died while creating it) has no job id, so only its age can make it stale."""
    try:
        times = [path.stat().st_mtime]
    except FileNotFoundError:
        return None
    info: dict = {}
    for name in ("claim.json", "heartbeat"):
        try:
            times.append((path / name).stat().st_mtime)
        except FileNotFoundError:
            pass
    try:
        info = json.loads((path / "claim.json").read_text())
    except (OSError, ValueError):
        pass
    return {**info, "age": time.time() - max(times)}


def age_text(seconds: float) -> str:
    """``45 s``, ``12 min`` or ``3.1 h``."""
    return (f"{seconds:.0f} s" if seconds < 120 else f"{seconds / 60:.0f} min" if seconds < 7200
            else f"{seconds / 3600:.1f} h")


def staleness(info: dict, states: dict[str, str] | None) -> str:
    """Why the claim ``info`` is stale (see the module docstring), or ``""`` if it is valid."""
    if info["age"] > STALE_AFTER:
        return f"heartbeat {age_text(info['age'])} old"
    job = str(info.get("job_id") or "")
    if states is not None and job.isdigit() and states.get(job) not in LIVE_STATES:
        return f"job {job} {states.get(job, 'ended').lower()}"
    return ""


def claim_states(exp: Path) -> dict[str, dict]:
    """``{puma: claim}`` for every claim in the experiment, each with ``age`` and
    ``stale`` (the reason, or ``""``); one ``squeue`` call for all of them."""
    folder = claims_dir(exp)
    if not folder.is_dir():
        return {}
    claims = {}
    for path in sorted(folder.iterdir()):
        if path.name.startswith("."):  # a tombstone on its way out
            continue
        info = read_claim(path)
        if info is not None:
            claims[path.name] = info
    states = job_states(str(info.get("job_id") or "") for info in claims.values())
    return {puma: {**info, "stale": staleness(info, states)} for puma, info in claims.items()}


def failures_path(exp: Path, puma: str) -> Path:
    return Path(exp) / "failures" / f"{puma}.jsonl"


def failure_records(exp: Path, puma: str) -> list[dict]:
    path = failures_path(exp, puma)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def is_failed(exp: Path, puma: str) -> bool:
    """Whether the PUMA has failed :data:`MAX_FAILURES` times and is left alone."""
    return len(failure_records(exp, puma)) >= MAX_FAILURES


def record_failure(exp: Path, puma: str, record: dict) -> int:
    """Append a failure to the PUMA's record; the failures recorded so far."""
    path = failures_path(exp, puma)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(record) + "\n")
    return len(failure_records(exp, puma))


def clear_failures(exp: Path, pumas: list[str] | None = None) -> list[str]:
    """Clear the failure records of ``pumas`` (every PUMA with one when ``None``),
    so jobs take them again; the PUMAs cleared. The records are moved to
    ``failures/cleared/``, not deleted, to keep their history."""
    folder = Path(exp) / "failures"
    paths = sorted(folder.glob("*.jsonl")) if folder.is_dir() else []
    if pumas is not None:
        paths = [p for p in paths if p.stem in set(pumas)]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for path in paths:
        (folder / "cleared").mkdir(exist_ok=True)
        os.replace(path, folder / "cleared" / f"{path.stem}-{stamp}.jsonl")
    return [p.stem for p in paths]


def remove_dir(path: Path, prefix: str) -> None:
    """Remove a claim directory: rename it out of the way first, so that it
    disappears at once rather than file by file."""
    tomb = path.parent / f".{prefix}-{path.name}-{uuid.uuid4().hex}"
    try:
        os.rename(path, tomb)
    except FileNotFoundError:
        return
    shutil.rmtree(tomb, ignore_errors=True)


class Claim:
    """This job's claim on one PUMA (see the module docstring)."""

    def __init__(self, exp: Path, puma: str):
        self.path = claims_dir(exp) / puma
        self.puma = puma
        self.token = uuid.uuid4().hex

    @classmethod
    def take(cls, exp: Path, puma: str, attempt: int, mode: str) -> Claim | None:
        """Claim the PUMA, taking over a stale claim; ``None`` if another job holds it.

        A takeover renames the stale claim to a tombstone, then checks that the
        tombstone holds the token judged stale: if a third job took the claim
        over in between, what was moved is that job's fresh claim, which is put
        back."""
        claim = cls(exp, puma)
        claim.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                os.mkdir(claim.path)
            except FileExistsError:
                info = read_claim(claim.path)
                if info is None:
                    continue  # released just now
                why = staleness(info, job_states([str(info.get("job_id") or "")]))
                if not why:
                    return None
                tomb = claim.path.parent / f".stale-{puma}-{uuid.uuid4().hex}"
                try:
                    os.rename(claim.path, tomb)
                except OSError:
                    continue  # another job got there first
                moved = read_claim(tomb) or {}
                if moved.get("token") != info.get("token"):
                    try:
                        os.rename(tomb, claim.path)
                    except OSError:
                        shutil.rmtree(tomb, ignore_errors=True)
                    return None
                shutil.rmtree(tomb, ignore_errors=True)
                log(f"{puma}: took over the claim of job {info.get('job_id', '?')} "
                    f"on {info.get('host', '?')} ({why})")
                continue
            record = {"job_id": job_identity(), "host": socket.gethostname(), "pid": os.getpid(),
                      "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "attempt": attempt,
                      "mode": mode, "token": claim.token}
            part = claim.path / ".claim.json.part"
            part.write_text(json.dumps(record) + "\n")
            os.replace(part, claim.path / "claim.json")
            claim.touch()
            return claim
        return None

    def touch(self) -> None:
        heartbeat = self.path / "heartbeat"
        try:
            os.utime(heartbeat)
        except FileNotFoundError:
            heartbeat.touch()

    def ours(self) -> bool | None:
        """Whether the claim is still this job's; ``None`` when that cannot be read."""
        try:
            return json.loads((self.path / "claim.json").read_text()).get("token") == self.token
        except FileNotFoundError:
            return False
        except (OSError, ValueError):
            return None

    def release(self) -> None:
        if self.ours():
            remove_dir(self.path, "released")


# -- running a PUMA -----------------------------------------------------------


class Worker:
    """This job's claims, heartbeat and PUMA processes, and its response to signals.

    ``stopping`` (USR1 or SIGTERM): take no new PUMA. ``terminating``
    (SIGTERM): stop the running ones too. A PUMA whose claim was lost is
    stopped the same way, alone."""

    def __init__(self, exp: Path | None, mode: str):
        self.exp, self.mode = exp, mode
        self.lock = threading.RLock()  # re-entrant: the signal handlers take it too
        self.held: dict[str, Claim] = {}
        self.procs: dict[str, subprocess.Popen] = {}
        self.lost: set[str] = set()
        self.stopping = threading.Event()
        self.terminating = threading.Event()
        self.done = threading.Event()

    def claim(self, puma: str) -> bool:
        """Claim the PUMA for this job (always granted without an experiment to claim in)."""
        if self.exp is None:
            return True
        claim = Claim.take(self.exp, puma, len(failure_records(self.exp, puma)) + 1, self.mode)
        if claim is None:
            return False
        with self.lock:
            self.held[puma] = claim
        return True

    def release(self, puma: str) -> None:
        with self.lock:
            claim = self.held.pop(puma, None)
        if claim is not None:
            try:
                claim.release()
            except OSError as error:
                log(f"{puma}: could not release the claim ({error!r}); it goes stale")

    def cancelled(self, puma: str) -> bool:
        return self.terminating.is_set() or puma in self.lost

    def stop(self, puma: str | None = None) -> None:
        """Signal the process group of one PUMA's running step (or every PUMA's)."""
        with self.lock:
            procs = [p for key, p in self.procs.items() if puma is None or key == puma]
        for proc in procs:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

    def terminate(self) -> None:
        self.stopping.set()
        self.terminating.set()
        self.stop()

    def heartbeat(self) -> None:
        """Touch every held claim each :data:`HEARTBEAT` seconds; stop a PUMA whose
        claim another job has taken over."""
        while not self.done.wait(HEARTBEAT):
            with self.lock:
                held = dict(self.held)
            for puma, claim in held.items():
                ours = claim.ours()
                if ours:
                    try:
                        claim.touch()
                    except OSError as error:
                        log(f"{puma}: heartbeat failed ({error!r})")
                elif ours is False:
                    log(f"{puma}: CLAIM LOST to another job; stopping this PUMA")
                    with self.lock:
                        self.lost.add(puma)
                        self.held.pop(puma, None)
                    self.stop(puma)

    def run(self, puma: str, command: list[str], env: dict[str, str], logfile: Path) -> bool:
        """Run ``command`` with its output appended to ``logfile``; whether it
        succeeded. It runs in its own process group, so that stopping it also
        stops its workers; nothing is started once the PUMA is cancelled."""
        if self.cancelled(puma):
            return False
        with open(logfile, "a") as handle:
            handle.write(f"\n=== {time.strftime('%H:%M:%S')} {' '.join(command)}\n")
            handle.flush()
            proc = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, cwd=REPO, start_new_session=True)
            with self.lock:
                self.procs[puma] = proc
            if self.cancelled(puma):  # a signal between the check and the start
                self.stop(puma)
            code = proc.wait()
            with self.lock:
                self.procs.pop(puma, None)
        return code == 0


def run_puma(item: dict, bundle: dict, gpu: str | None, cores: int, worker: Worker) -> list[str]:
    """Fit one PUMA and sample it; the steps that failed."""
    py = sys.executable
    puma = item["puma"]
    out = Path(item["out"])
    out.mkdir(parents=True, exist_ok=True)
    logfile = out / "bundle.log"
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    device = bundle.get("device", "cpu") if gpu is not None else "cpu"
    common = ["--name", bundle["name"], "--puma", puma, "--cores", str(cores),
              *bundle["fits_args"]]
    failed = []

    # run_map.py sets its workers' BLAS threads itself; one for its driver process.
    fits_env = {**env, **{v: "1" for v in THREAD_VARIABLES}}
    if not worker.run(puma, [py, "-u", str(EXPERIMENTS / "run_map.py"), "solve", *common,
                             "--out", str(out / "map")], fits_env, logfile):
        failed.append("map")
    for family in bundle["families"]:
        if not worker.run(puma, [py, "-u", str(EXPERIMENTS / "run_map.py"), "vb", *common,
                                 "--family", family, "--draws", str(bundle.get("draws", 64)),
                                 "--device", device, "--out", str(out / f"vb_{family}")],
                          fits_env, logfile):
            failed.append(f"vb_{family}")
    # run_map.py records a failed fit in summary.csv, so check the outputs too.
    missing = [path for path in item.get("fit_outputs", []) if not Path(path).exists()]
    if missing:
        failed.append(f"fits missing {len(missing)} output(s)")
    if failed:
        return failed + ["hmc (not run: fits incomplete)"]

    hmc_env = {**env, **{v: str(cores) for v in THREAD_VARIABLES}}
    for run_spec in item.get("hmc", []):
        report = Path(run_spec["report"])
        if report.exists():
            continue
        hmc_out = Path(run_spec["out"])
        hmc_out.mkdir(parents=True, exist_ok=True)
        args = [*run_spec["args"], "--out", str(hmc_out)]
        if gpu is not None:
            args += ["--device", "cuda"]
        ok = (worker.run(puma, [py, "-u", str(EXPERIMENTS / "run_mcmc.py"), "sample", *args],
                         hmc_env, logfile)
              and worker.run(puma, [py, "-u", str(EXPERIMENTS / "run_mcmc.py"), "report", *args],
                             hmc_env, logfile))
        if not ok or not report.exists():
            failed.append(f"hmc {hmc_out.name}")
    return failed


def item_complete(item: dict) -> bool:
    """Whether every fit and HMC report of a bundle entry exists."""
    paths = item.get("fit_outputs", []) + [run["report"] for run in item.get("hmc", [])]
    return bool(paths) and all(Path(path).exists() for path in paths)


# -- where the PUMAs come from ------------------------------------------------


class FixedList:
    """A bundle's PUMAs, in its order. One that is complete, failed, or claimed
    by another live job is skipped."""

    def __init__(self, bundle: dict, worker: Worker):
        self.worker = worker
        self.work: queue.Queue = queue.Queue()
        for item in bundle["pumas"]:
            self.work.put(item)
        self.total = len(bundle["pumas"])

    def next(self) -> dict | None:
        exp = self.worker.exp
        while not self.worker.stopping.is_set():
            try:
                item = self.work.get_nowait()
            except queue.Empty:
                return None
            puma = item["puma"]
            if exp is not None and item_complete(item):
                log(f"{puma}: already complete; skipped")
            elif exp is not None and is_failed(exp, puma):
                log(f"{puma}: failed {MAX_FAILURES} times (run_experiment.py --retry-failed); "
                    "skipped")
            elif not self.worker.claim(puma):
                log(f"{puma}: claimed by another live job; skipped")
            elif exp is not None and item_complete(item):  # finished just before the claim
                self.worker.release(puma)
                log(f"{puma}: already complete; skipped")
            else:
                return item
        return None


class Pilot:
    """An experiment's unfinished PUMAs, slowest first, each taken once."""

    def __init__(self, bundle: dict, worker: Worker):
        import run_experiment as rx

        self.rx, self.spec, self.worker = rx, bundle["spec"], worker
        self.exp = worker.exp
        self.lock = threading.Lock()
        self.tried: set[str] = set()
        self.order = sorted(self.spec["pumas"], key=self.size, reverse=True)
        self.total = len(self.order)

    def size(self, puma: str) -> int:
        """zones x units of the PUMA's assembled inputs: the slowest go first."""
        try:
            manifest = json.loads((self.rx.inputs_dir(self.spec) / puma / "manifest.json")
                                  .read_text())
            return int(manifest["shapes"]["n_zones"]) * int(manifest["shapes"]["n_units"])
        except (OSError, ValueError, KeyError):
            return 0

    def next(self) -> dict | None:
        with self.lock:
            # One squeue call for every claim, rather than one per claimed PUMA.
            held = {puma for puma, info in claim_states(self.exp).items() if not info["stale"]}
            for puma in self.order:
                if self.worker.stopping.is_set():
                    return None
                if puma in self.tried or puma in held or is_failed(self.exp, puma):
                    continue
                if self.rx.puma_complete(self.spec, self.exp, puma):
                    continue
                if not self.worker.claim(puma):
                    continue
                self.tried.add(puma)
                # Finished by another job between the check and the claim?
                if self.rx.puma_complete(self.spec, self.exp, puma):
                    self.worker.release(puma)
                    continue
                return self.rx.bundle_item(self.spec, self.exp, puma)
        return None


# -- the driver ---------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path, help="the work file (JSON): a bundle or a pilot")
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    exp = Path(bundle["experiment"]) if bundle.get("experiment") else None
    mode = "pilot" if "spec" in bundle else "bundle"
    if mode == "pilot" and exp is None:
        raise SystemExit(f"{args.bundle}: a pilot needs its 'experiment'")
    worker = Worker(exp, mode)
    source = Pilot(bundle, worker) if mode == "pilot" else FixedList(bundle, worker)

    gpus = streams()
    total = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    cores = max(1, total // len(gpus))
    log(f"{mode} {args.bundle} (job {job_identity()}): {source.total} PUMA(s), "
        f"{len(gpus)} stream(s) "
        f"({'GPUs ' + ','.join(gpus) if gpus[0] is not None else 'CPU'}), {cores} cores each"
        + ("" if exp is not None else "; no experiment, so no claims"))
    if exp is not None:
        for tomb in claims_dir(exp).glob(".*-*"):  # left by a job killed while removing it
            if time.time() - tomb.stat().st_mtime > STALE_AFTER:
                shutil.rmtree(tomb, ignore_errors=True)

    def on_usr1(signum, frame) -> None:
        log("USR1: the time limit is near; no new PUMA will be started")
        worker.stopping.set()

    def on_term(signum, frame) -> None:
        log(f"{signal.Signals(signum).name}: stopping the running PUMAs and releasing their claims")
        worker.terminate()

    signal.signal(signal.SIGUSR1, on_usr1)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    failures: dict[str, list[str]] = {}
    finished: list[str] = []
    lock = threading.Lock()

    def stream(gpu: str | None) -> None:
        while not worker.stopping.is_set():
            try:
                item = source.next()
            except OSError as error:  # the file system failing under the claims
                log(f"stream ended: could not take the next PUMA ({error!r})")
                return
            if item is None:
                return
            puma = item["puma"]
            label = f"[{'gpu ' + gpu if gpu is not None else 'cpu'}] {puma}"
            log(f"{label} started")
            start = time.perf_counter()
            try:
                failed = run_puma(item, bundle, gpu, cores, worker)
            except Exception as error:  # a broken PUMA must not stop its stream
                failed = [repr(error)]
            minutes = (time.perf_counter() - start) / 60
            logfile = Path(item["out"]) / "bundle.log"
            if failed and worker.cancelled(puma):
                # Killed, out of time or taken over: not the PUMA's fault; it resumes later.
                why = "claim lost" if puma in worker.lost else "stopped"
                log(f"{label} {why} after {minutes:.1f} min; not counted as a failure")
                worker.release(puma)
                continue
            if failed:
                with lock:
                    failures[puma] = failed
                count = (record_failure(exp, puma, {
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "job_id": job_identity(),
                    "host": socket.gethostname(), "failed": failed, "log": str(logfile)})
                    if exp is not None else 0)
                log(f"{label} FAILED after {minutes:.1f} min: {'; '.join(failed)} (see {logfile})"
                    + (f"; failure {count} of {MAX_FAILURES}" if exp is not None else ""))
            else:
                with lock:
                    finished.append(puma)
                log(f"{label} done in {minutes:.1f} min")
            worker.release(puma)

    beat = threading.Thread(target=worker.heartbeat, daemon=True)
    beat.start()
    threads = [threading.Thread(target=stream, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    deadline = None
    while any(thread.is_alive() for thread in threads):
        for thread in threads:
            thread.join(timeout=1)
        if worker.terminating.is_set():
            deadline = deadline or time.monotonic() + TERM_GRACE
            if time.monotonic() > deadline:
                break
    worker.done.set()
    with worker.lock:
        held = list(worker.held)
    for puma in held:  # streams that did not wind down in time
        worker.release(puma)
    log(f"finished: {len(finished)} PUMA(s) done, {len(failures)} failed"
        f"{': ' + ', '.join(sorted(failures)) if failures else ''}"
        + ("; stopped by SIGTERM" if worker.terminating.is_set() else
           "; stopped early by USR1" if worker.stopping.is_set() else ""))
    sys.exit(143 if worker.terminating.is_set() else 1 if failures else 0)


if __name__ == "__main__":
    main()
