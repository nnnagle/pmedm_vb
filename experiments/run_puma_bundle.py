"""Run many PUMAs in one job: MAP, the VB fits and their HMC runs, one PUMA per GPU at a time.

Written for queues where a job waits days but a PUMA takes minutes, so that a
few jobs carry a whole study area. ``run_experiment.py`` writes the bundle
file and submits it with ``run_puma_bundle.sbatch``; by hand::

    $CONDA_PREFIX/bin/python experiments/run_puma_bundle.py BUNDLE.json

The bundle file (JSON) holds::

    {"name": AREA_NAME, "fits_args": [run_map.py options shared by every PUMA],
     "families": ["gaussian"], "draws": 64, "device": "cuda",
     "pumas": [{"puma": "4701502", "out": ".../tn01_gpu/4701502",
                "hmc": [{"out": ".../hmc_short", "args": [run_mcmc.py options],
                         "report": ".../hmc_short/<name>_report.txt"}]}, ...]}

**One stream per GPU.** The GPUs are those Slurm gives the job
(``CUDA_VISIBLE_DEVICES``); each stream takes the next PUMA from a shared
queue and runs it with that one GPU visible, so the fits and the sampler see
it as ``cuda``. The job's cores are split evenly between the streams. With no
GPU, one stream runs on the CPU.

**Per PUMA**, in order: ``run_map.py solve`` (the MAP, into ``<out>/map``),
``run_map.py vb`` for each family (into ``<out>/vb_<family>``), then for each
HMC run ``run_mcmc.py sample`` and ``report``. Finished work is skipped: the
fits whose ``.npz`` exists (run_map.py's own check) and the HMC runs whose
report exists; an unfinished sampler resumes from its checkpoint. So a
bundle that runs out of time is finished by submitting the same file again.
A PUMA whose fits fail gets no HMC; a failure does not stop the other PUMAs,
and the job exits non-zero if any PUMA failed.

Each PUMA's output goes to ``<out>/bundle.log``; this script's own log says
which stream ran which PUMA, and for how long.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS")


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {message}", flush=True)


def streams() -> list[str | None]:
    """The GPU ids to run on, one stream each: ``CUDA_VISIBLE_DEVICES``, else one CPU stream."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpus = [g for g in visible.split(",") if g.strip()] if visible else []
    return gpus or [None]


def run(command: list[str], env: dict[str, str], logfile: Path) -> bool:
    """Run ``command`` with its output appended to ``logfile``; whether it succeeded."""
    with open(logfile, "a") as handle:
        handle.write(f"\n=== {time.strftime('%H:%M:%S')} {' '.join(command)}\n")
        handle.flush()
        result = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, cwd=REPO)
    return result.returncode == 0


def run_puma(item: dict, bundle: dict, gpu: str | None, cores: int) -> list[str]:
    """Fit one PUMA and sample it; the steps that failed."""
    py = sys.executable
    out = Path(item["out"])
    out.mkdir(parents=True, exist_ok=True)
    logfile = out / "bundle.log"
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    device = bundle.get("device", "cpu") if gpu is not None else "cpu"
    common = ["--name", bundle["name"], "--puma", item["puma"], "--cores", str(cores),
              *bundle["fits_args"]]
    failed = []

    # run_map.py sets its workers' BLAS threads itself; one for its driver process.
    fits_env = {**env, **{v: "1" for v in THREAD_VARIABLES}}
    if not run([py, "-u", str(EXPERIMENTS / "run_map.py"), "solve", *common,
                "--out", str(out / "map")], fits_env, logfile):
        failed.append("map")
    for family in bundle["families"]:
        if not run([py, "-u", str(EXPERIMENTS / "run_map.py"), "vb", *common,
                    "--family", family, "--draws", str(bundle.get("draws", 64)),
                    "--device", device, "--out", str(out / f"vb_{family}")], fits_env, logfile):
            failed.append(f"vb_{family}")
    # run_map.py records a failed fit in summary.csv and exits 0, so check the outputs.
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
        ok = (run([py, "-u", str(EXPERIMENTS / "run_mcmc.py"), "sample", *args], hmc_env, logfile)
              and run([py, "-u", str(EXPERIMENTS / "run_mcmc.py"), "report", *args], hmc_env,
                      logfile))
        if not ok or not report.exists():
            failed.append(f"hmc {hmc_out.name}")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path, help="the bundle file (JSON)")
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())

    gpus = streams()
    total = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    cores = max(1, total // len(gpus))
    log(f"bundle {args.bundle}: {len(bundle['pumas'])} PUMA(s), {len(gpus)} stream(s) "
        f"({'GPUs ' + ','.join(gpus) if gpus[0] is not None else 'CPU'}), {cores} cores each")

    work: queue.Queue = queue.Queue()
    for item in bundle["pumas"]:
        work.put(item)
    failures: dict[str, list[str]] = {}
    lock = threading.Lock()

    def stream(gpu: str | None) -> None:
        while True:
            try:
                item = work.get_nowait()
            except queue.Empty:
                return
            label = f"[{'gpu ' + gpu if gpu is not None else 'cpu'}] {item['puma']}"
            log(f"{label} started")
            start = time.perf_counter()
            try:
                failed = run_puma(item, bundle, gpu, cores)
            except Exception as error:  # a broken PUMA must not stop its stream
                failed = [repr(error)]
            minutes = (time.perf_counter() - start) / 60
            if failed:
                with lock:
                    failures[item["puma"]] = failed
                log(f"{label} FAILED after {minutes:.1f} min: {'; '.join(failed)} "
                    f"(see {Path(item['out']) / 'bundle.log'})")
            else:
                log(f"{label} done in {minutes:.1f} min")

    threads = [threading.Thread(target=stream, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    log(f"finished: {len(bundle['pumas']) - len(failures)} PUMA(s) done, "
        f"{len(failures)} failed{': ' + ', '.join(sorted(failures)) if failures else ''}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
