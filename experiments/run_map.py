"""Run the MAP or VB sweep for a study area: prefetch, assemble, solve / vb.

Separate steps, because ISAAC's compute nodes cannot reach census.gov::

    # login or data transfer node: download whatever is not cached yet
    $CONDA_PREFIX/bin/python experiments/run_map.py prefetch

    # compute node (see run_map.sbatch): build one problem per PUMA, then fit
    $CONDA_PREFIX/bin/python experiments/run_map.py assemble
    $CONDA_PREFIX/bin/python experiments/run_map.py solve --out DIR   # MAP
    $CONDA_PREFIX/bin/python experiments/run_map.py vb --out DIR      # VB

**assemble** builds each PUMA in its own process and saves it with
:meth:`~pmedm_vb.assemble.inputs.PMEDMInputs.save` under
``$PMEDM_VB_DATA/processed/inputs/<area>/<puma>``. A PUMA already saved is
skipped; ``--rebuild`` rebuilds it, which is needed after changing the
constraint tables or the assembly code.

**solve** and **vb** run every (PUMA, taper, alpha) combination in a process
pool. ``vb`` fits the MAP solution first, in the same worker, and starts from
its Laplace approximation (see :mod:`pmedm_vb.solvers.vb`).
Problems are started largest first, so a big one does not start last and set
the finishing time. Each process gets ``cores // workers`` BLAS threads -- one
each when there are at least as many solves as cores. Every solve writes
``<puma>_<taper>_a<alpha>.npz`` (``lam``, ``W``, the trace and the scalars), and
``summary.csv`` is rewritten after each one, so a job that hits its time limit
keeps what finished. Fits whose ``.npz`` already exists in ``--out`` are
skipped, so re-submitting with the same ``--out`` resumes. A fit that raises
is recorded in ``summary.csv`` with its error and does not stop the others.

**Statewide support.** ``--epsilon E`` (all steps but prefetch) builds and
solves the problems whose support is every record in the state, the PUMA's own
keeping ``1 - E`` of the prior (see :mod:`pmedm_vb.assemble.build`). They live
in their own inputs directory, ``<area>-state-e<E>``, so the PUMA-only inputs
are untouched; pass that name as ``--area`` to the diagnostics. Assembly writes
``support_summary.csv`` there, one row per PUMA: records and merged rows, how
many rows are new to the support and the share of the prior on them. Each
PUMA's directory also gets ``support_columns.parquet``, the rows and records
carrying each constraint category with and without the wider support.

**Logs.** Each worker task writes its progress -- every Newton or VB iteration --
to its own file, ``<out>/logs/<puma>_<taper>_a<alpha>.log`` for a fit and
``<inputs>/logs/assemble_<puma>.log`` for assembly, rather than interleaving
with the others. The main log carries one line per finished task, naming the
file to read when one fails.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.config import StudyArea, processed_dir
from pmedm_vb.progress import logger, record_run

#: Environment variables that set BLAS/OpenMP thread counts in a fresh process.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=["prefetch", "assemble", "solve", "vb"])
    parser.add_argument("--name", default="knox")
    parser.add_argument("--state", default="47", help="state FIPS")
    parser.add_argument(
        "--county", action="append", default=None,
        help="county FIPS; repeatable (default: 093, Knox)",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--span", type=int, default=5)
    parser.add_argument(
        "--cores", type=int, default=None,
        help="cores to use (default: $SLURM_CPUS_PER_TASK, else all)",
    )
    parser.add_argument("--rebuild", action="store_true", help="assemble: rebuild saved PUMAs")
    parser.add_argument(
        "--alpha", type=float, nargs="+", default=[1.0, 0.7, 0.5, 0.3, 0.1],
        help="solve: shrinkage values to sweep",
    )
    parser.add_argument(
        "--taper", nargs="+", default=["tract", "none"], choices=["tract", "none"],
        help="solve: Sigma tapers to sweep",
    )
    parser.add_argument("--out", type=Path, default=None, help="solve, vb: results directory")
    parser.add_argument(
        "--variance-floor", type=floor_spec, default=None,
        help="solve, vb: 'zero' floors each cell's variance at its area's zero-count "
             "variance; a number floors at that value; 'none' (default) leaves them",
    )
    parser.add_argument(
        "--family", choices=["gaussian", "skewed", "sumdiff"], default="gaussian",
        help="vb: 'skewed' adds a second stage fitting a per-coordinate skew; 'sumdiff' "
             "adds skew layers on tract+block group sums and differences as well",
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="vb: iteration cap (per stage)")
    parser.add_argument("--draws", type=int, default=8, help="vb: Monte Carlo draws per step")
    parser.add_argument("--learning-rate", type=float, default=0.02, help="vb: Adam step")
    parser.add_argument(
        "--epsilon", type=float, default=None,
        help="assemble, solve, vb: statewide support with this prior share off the "
             "PUMA's own records (default: the PUMA's records only)",
    )
    return parser.parse_args()


def floor_spec(text: str) -> str | float | None:
    """``--variance-floor``: ``none``, ``zero`` or a positive number."""
    if text == "none":
        return None
    if text == "zero":
        return "zero"
    return float(text)


def study_area(args: argparse.Namespace) -> StudyArea:
    return StudyArea(
        name=args.name,
        state=args.state,
        year=args.year,
        counties=tuple(args.county or ("093",)),
        span=args.span,
    )


def inputs_root(area: StudyArea, epsilon: float | None = None) -> Path:
    name = area.slug if epsilon is None else f"{area.slug}-state-e{epsilon:g}"
    return processed_dir() / "inputs" / name


def available_cores(requested: int | None) -> int:
    if requested:
        return requested
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def log_to_file(path: Path) -> None:
    """Send this process's log lines to ``path``, and only there.

    Pool processes are reused, so whatever handler the previous task left --
    the default stderr one, or the previous task's file -- is closed first.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(path, mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)


def pool(workers: int, threads: int) -> ProcessPoolExecutor:
    """A process pool whose workers start with ``threads`` BLAS threads each.

    The variables must be in place before a worker imports numpy, so they are
    set in this process's environment, which ``spawn`` hands to each worker
    as it starts.
    """
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(threads)
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )


# -- prefetch ----------------------------------------------------------------


def run_prefetch(area: StudyArea) -> None:
    from pmedm_vb.assemble.constraints import default_tables
    from pmedm_vb.data.prefetch import prefetch

    prefetch(area, default_tables(area))


# -- assemble ----------------------------------------------------------------


def assemble_one(
    area: StudyArea, puma: str, path: Path, epsilon: float | None = None
) -> tuple[str, float]:
    """Worker: build and save one PUMA's problem."""
    from pmedm_vb.assemble.build import build_puma
    from pmedm_vb.assemble.constraints import default_tables
    from pmedm_vb.data.geography import puma_crosswalk_whole

    log_to_file(path.parent / "logs" / f"assemble_{puma}.log")
    start = time.perf_counter()
    zones = puma_crosswalk_whole(area)
    build_puma(area, puma, default_tables(area), zones=zones, epsilon=epsilon).save(path)
    return puma, time.perf_counter() - start


def run_assemble(
    args: argparse.Namespace,
    area: StudyArea, cores: int, rebuild: bool, epsilon: float | None = None
) -> None:
    from pmedm_vb.data.geography import puma_crosswalk_whole

    pumas = sorted(str(p) for p in puma_crosswalk_whole(area)["puma_geoid"].unique())
    root = inputs_root(area, epsilon)
    record_run(root, args)
    todo = [p for p in pumas if rebuild or not (root / p / "manifest.json").exists()]
    logger.info(
        "assemble %s: %d PUMAs, %d to build, into %s", area.slug, len(pumas), len(todo), root
    )
    if not todo:
        write_support_summary(root, pumas, epsilon)
        return
    workers = min(cores, len(todo))
    failed = []
    with pool(workers, max(1, cores // workers)) as executor:
        futures = {executor.submit(assemble_one, area, p, root / p, epsilon): p for p in todo}
        for future in as_completed(futures):
            try:
                puma, seconds = future.result()
                logger.info("assembled PUMA %s in %.0fs", puma, seconds)
            except Exception:
                puma = futures[future]
                failed.append(puma)
                logger.error(
                    "PUMA %s failed; see %s\n%s", puma,
                    root / "logs" / f"assemble_{puma}.log", traceback.format_exc(),
                )
    write_support_summary(root, pumas, epsilon)
    if failed:
        raise SystemExit(f"assembly failed for PUMA(s) {failed}")


def write_support_summary(root: Path, pumas: list[str], epsilon: float | None) -> None:
    """Collect each PUMA's support counts into ``<root>/support_summary.csv`` and log them."""
    import json

    if epsilon is None:
        return
    rows = []
    for puma in pumas:
        manifest = root / puma / "manifest.json"
        if manifest.exists():
            support = json.loads(manifest.read_text()).get("support")
            if support:
                rows.append({"puma": puma, **support})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "support_summary.csv", index=False)
    logger.info("support (epsilon=%g), also in %s:\n%s",
                epsilon, root / "support_summary.csv", frame.to_string(index=False))


# -- solve -------------------------------------------------------------------


def estimated_cost(manifest: dict, taper: str | None) -> float:
    """Relative cost of one solve, for ordering only.

    Per Newton iteration: one Cholesky per tract, cubic in the block size;
    forming those blocks and the dual, roughly zones x units x cells per tract;
    and, untapered, the 81-column Woodbury correction. Iteration counts vary
    little (8 on every real solve so far), so they are left out.
    """
    shapes = manifest["shapes"]
    m, tracts = shapes["n_constraints"], shapes["n_tracts"]
    per_tract = m / tracts
    cost = tracts * per_tract**3 + shapes["n_zones"] * shapes["n_units"] * per_tract
    if taper is None:
        cost += m * 81**2
    return cost


def result_name(puma: str, taper: str | None, alpha: float) -> str:
    return f"{puma}_{taper or 'none'}_a{alpha:g}"


def solve_one(path: Path, taper: str | None, alpha: float, out: Path, options: dict) -> dict:
    """Worker: solve one (PUMA, taper, alpha) and save it. Never raises."""
    from pmedm_vb.assemble.inputs import PMEDMInputs
    from pmedm_vb.solvers.map_dual import solve_map

    puma = path.name
    log_to_file(out / "logs" / f"{result_name(puma, taper, alpha)}.log")
    floor = options["variance_floor"]
    row = {"puma": puma, "taper": taper or "none", "alpha": alpha,
           "variance_floor": "none" if floor is None else str(floor)}
    start = time.perf_counter()
    try:
        inputs = PMEDMInputs.load(path)
        result = solve_map(inputs, alpha=alpha, taper=taper, variance_floor=floor)
        row.update(
            n_constraints=inputs.n_constraints,
            converged=result.converged,
            n_iter=result.n_iter,
            newton_decrement=result.newton_decrement,
            objective=result.objective,
            max_abs_z=float(result.trace["max_abs_z"][-1]),
            mahalanobis=float(result.trace["mahalanobis"][-1]),
            log_evidence=result.log_evidence,
            seconds=time.perf_counter() - start,
            error="",
        )
        np.savez(
            out / f"{result_name(puma, taper, alpha)}.npz",
            lam=result.lam,
            W=result.W,
            **{f"trace_{key}": value for key, value in result.trace.items()},
            **{key: np.asarray(value) for key, value in row.items()},
        )
    except Exception as error:
        row.update(seconds=time.perf_counter() - start, error=repr(error))
        logger.error("failed:\n%s", traceback.format_exc())
    return row


def vb_one(path: Path, taper: str | None, alpha: float, out: Path, options: dict) -> dict:
    """Worker: MAP, then VB from its Laplace approximation, and save. Never raises."""
    import torch

    from pmedm_vb.assemble.inputs import PMEDMInputs
    from pmedm_vb.solvers.map_dual import solve_map
    from pmedm_vb.solvers.vb import solve_vb

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    puma = path.name
    log_to_file(out / "logs" / f"{result_name(puma, taper, alpha)}.log")
    floor = options["variance_floor"]
    row = {"puma": puma, "taper": taper or "none", "alpha": alpha,
           "variance_floor": "none" if floor is None else str(floor)}
    vb_options = {k: v for k, v in options.items() if k != "variance_floor"}
    start = time.perf_counter()
    try:
        inputs = PMEDMInputs.load(path)
        start_map = solve_map(inputs, alpha=alpha, taper=taper, variance_floor=floor)
        fit = solve_vb(inputs, alpha=alpha, taper=taper, variance_floor=floor,
                       init=start_map, **vb_options)
        row.update(
            n_constraints=inputs.n_constraints,
            converged=fit.converged,
            n_iter=fit.n_iter,
            elbo=fit.elbo,
            elbo_se=fit.elbo_se,
            laplace_elbo=fit.laplace_elbo,
            laplace_elbo_se=fit.laplace_elbo_se,
            family=fit.family,
            gaussian_elbo=fit.gaussian_elbo,
            gaussian_elbo_se=fit.gaussian_elbo_se,
            gain=fit.elbo - fit.laplace_elbo,
            seconds=time.perf_counter() - start,
            error="",
        )
        np.savez(
            out / f"{result_name(puma, taper, alpha)}.npz",
            mean=fit.q.mean,
            W=fit.q.W,
            V=fit.q.V,
            map_lam=fit.map_result.lam,
            elbo_trace=fit.elbo_trace,
            **{f"rows_{i}": rows for i, rows in enumerate(fit.q.rows)},
            **{f"block_{i}": block for i, block in enumerate(fit.q.blocks)},
            **fit.q.skew_params(),
            **{key: np.asarray(value) for key, value in row.items()},
        )
    except Exception as error:
        row.update(seconds=time.perf_counter() - start, error=repr(error))
        logger.error("failed:\n%s", traceback.format_exc())
    return row


#: Summary columns per step, in order; read back from a saved result on resume.
SUMMARY_KEYS = {
    "solve": ["puma", "taper", "alpha", "variance_floor", "n_constraints", "converged", "n_iter",
              "newton_decrement", "objective", "max_abs_z", "mahalanobis",
              "log_evidence", "seconds", "error"],
    "vb": ["puma", "taper", "alpha", "variance_floor", "family", "n_constraints", "converged",
           "n_iter", "elbo", "elbo_se", "gaussian_elbo", "gaussian_elbo_se", "laplace_elbo",
           "laplace_elbo_se", "gain", "seconds", "error"],
}


def saved_row(path: Path, step: str) -> dict:
    """The summary row stored in an existing result, for resuming."""
    keys = SUMMARY_KEYS[step]
    with np.load(path) as saved:
        # Results written before a column existed resume with it blank.
        return {key: saved[key].item() if key in saved else np.nan for key in keys}


def run_sweep(
    args: argparse.Namespace,
    step: str,
    area: StudyArea,
    cores: int,
    alphas: list[float],
    tapers: list[str],
    out: Path | None,
    options: dict,
) -> None:
    """Run ``solve`` or ``vb`` over every (PUMA, taper, alpha) not already in ``out``."""
    import json

    root = inputs_root(area, options.pop("epsilon", None))
    pumas = sorted(p for p in root.glob("*") if (p / "manifest.json").exists())
    if not pumas:
        raise SystemExit(f"no assembled problems under {root}; run the assemble step first")
    out = out or processed_dir() / "runs" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    record_run(out, args)

    rows, tasks = [], []
    for path in pumas:
        manifest = json.loads((path / "manifest.json").read_text())
        for taper in (None if t == "none" else t for t in tapers):
            for alpha in alphas:
                done = out / f"{result_name(path.name, taper, alpha)}.npz"
                if done.exists():
                    rows.append(saved_row(done, step))
                else:
                    tasks.append((estimated_cost(manifest, taper), path, taper, alpha))
    tasks.sort(key=lambda task: task[0], reverse=True)

    logger.info(
        "%s %s: %d to run, %d already in %s; per-task logs in %s",
        step, area.slug, len(tasks), len(rows), out, out / "logs",
    )
    if tasks:
        workers = min(cores, len(tasks))
        threads = max(1, cores // workers)
        logger.info("%d workers x %d BLAS thread(s), largest problems first", workers, threads)
        with pool(workers, threads) as executor:
            futures = [
                executor.submit(solve_one, path, taper, alpha, out, options) if step == "solve"
                else executor.submit(vb_one, path, taper, alpha, out, options)
                for _, path, taper, alpha in tasks
            ]
            for count, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                logger.info(
                    "%d of %d done: %s %s a=%g %s",
                    count, len(tasks), row["puma"], row["taper"], row["alpha"],
                    f"FAILED {row['error']} -- see "
                    f"{out / 'logs' / (result_name(row['puma'], row['taper'], row['alpha']) + '.log')}"
                    if row["error"] else
                    f"converged={row['converged']} iters={row['n_iter']} {row['seconds']:.0f}s",
                )
                write_summary(rows, out)
    write_summary(rows, out)
    failures = sum(bool(row["error"]) for row in rows)
    logger.info("summary: %s (%d failed)", out / "summary.csv", failures)


def write_summary(rows: list[dict], out: Path) -> None:
    frame = pd.DataFrame(rows).sort_values(["puma", "taper", "alpha"], ascending=[True, True, False])
    frame.to_csv(out / "summary.csv", index=False)


def main() -> None:
    args = parse_args()
    area = study_area(args)
    cores = available_cores(args.cores)
    if args.step == "prefetch":
        run_prefetch(area)
    elif args.step == "assemble":
        run_assemble(args, area, cores, args.rebuild, args.epsilon)
    else:
        options = {"max_iter": args.max_iter, "draws": args.draws,
                   "learning_rate": args.learning_rate, "variance_floor": args.variance_floor,
                   "epsilon": args.epsilon,
                   **({"family": args.family} if args.step == "vb" else {})}
        run_sweep(args, args.step, area, cores, args.alpha, args.taper, args.out, options)


if __name__ == "__main__":
    main()
