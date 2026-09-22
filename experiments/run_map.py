"""Run the MAP sweep for a study area: prefetch, assemble, solve.

Three steps, because ISAAC's compute nodes cannot reach census.gov::

    # login or data transfer node: download whatever is not cached yet
    $CONDA_PREFIX/bin/python experiments/run_map.py prefetch

    # compute node (see run_map.sbatch): build one problem per PUMA, then solve
    $CONDA_PREFIX/bin/python experiments/run_map.py assemble
    $CONDA_PREFIX/bin/python experiments/run_map.py solve --out DIR

**assemble** builds each PUMA in its own process and saves it with
:meth:`~pmedm_vb.assemble.inputs.PMEDMInputs.save` under
``$PMEDM_VB_DATA/processed/inputs/<area>/<puma>``. A PUMA already saved is
skipped; ``--rebuild`` rebuilds it, which is needed after changing the
constraint tables or the assembly code.

**solve** runs every (PUMA, taper, alpha) combination in a process pool.
Problems are started largest first, so a big one does not start last and set
the finishing time. Each process gets ``cores // workers`` BLAS threads -- one
each when there are at least as many solves as cores. Every solve writes
``<puma>_<taper>_a<alpha>.npz`` (``lam``, ``W``, the trace and the scalars), and
``summary.csv`` is rewritten after each one, so a job that hits its time limit
keeps what finished. Solves whose ``.npz`` already exists in ``--out`` are
skipped, so re-submitting with the same ``--out`` resumes. A solve that raises
is recorded in ``summary.csv`` with its error and does not stop the others.

Log lines from worker processes carry a tag -- ``[4701502 tract a=0.7]`` -- since
they arrive interleaved.
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
from pmedm_vb.progress import logger

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
    parser.add_argument("step", choices=["prefetch", "assemble", "solve"])
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
    parser.add_argument("--out", type=Path, default=None, help="solve: results directory")
    return parser.parse_args()


def study_area(args: argparse.Namespace) -> StudyArea:
    return StudyArea(
        name=args.name,
        state=args.state,
        year=args.year,
        counties=tuple(args.county or ("093",)),
        span=args.span,
    )


def inputs_root(area: StudyArea) -> Path:
    return processed_dir() / "inputs" / area.slug


def available_cores(requested: int | None) -> int:
    if requested:
        return requested
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def tag_logs(tag: str) -> None:
    """Prefix this process's log lines, since workers' lines interleave."""
    for handler in logger.handlers:
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s pmedm_vb [{tag}]  %(message)s", "%H:%M:%S")
        )


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


def assemble_one(area: StudyArea, puma: str, path: Path) -> tuple[str, float]:
    """Worker: build and save one PUMA's problem."""
    from pmedm_vb.assemble.build import build_puma
    from pmedm_vb.assemble.constraints import default_tables
    from pmedm_vb.data.geography import puma_crosswalk_whole

    tag_logs(puma)
    start = time.perf_counter()
    zones = puma_crosswalk_whole(area)
    build_puma(area, puma, default_tables(area), zones=zones).save(path)
    return puma, time.perf_counter() - start


def run_assemble(area: StudyArea, cores: int, rebuild: bool) -> None:
    from pmedm_vb.data.geography import puma_crosswalk_whole

    pumas = sorted(str(p) for p in puma_crosswalk_whole(area)["puma_geoid"].unique())
    root = inputs_root(area)
    todo = [p for p in pumas if rebuild or not (root / p / "manifest.json").exists()]
    logger.info(
        "assemble %s: %d PUMAs, %d to build, into %s", area.slug, len(pumas), len(todo), root
    )
    if not todo:
        return
    workers = min(cores, len(todo))
    failed = []
    with pool(workers, max(1, cores // workers)) as executor:
        futures = {executor.submit(assemble_one, area, p, root / p): p for p in todo}
        for future in as_completed(futures):
            try:
                puma, seconds = future.result()
                logger.info("assembled PUMA %s in %.0fs", puma, seconds)
            except Exception:
                failed.append(futures[future])
                logger.error("PUMA %s failed:\n%s", futures[future], traceback.format_exc())
    if failed:
        raise SystemExit(f"assembly failed for PUMA(s) {failed}")


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


def solve_one(path: Path, taper: str | None, alpha: float, out: Path) -> dict:
    """Worker: solve one (PUMA, taper, alpha) and save it. Never raises."""
    from pmedm_vb.assemble.inputs import PMEDMInputs
    from pmedm_vb.solvers.map_dual import solve_map

    puma = path.name
    tag_logs(f"{puma} {taper or 'none'} a={alpha:g}")
    row = {"puma": puma, "taper": taper or "none", "alpha": alpha}
    start = time.perf_counter()
    try:
        inputs = PMEDMInputs.load(path)
        result = solve_map(inputs, alpha=alpha, taper=taper)
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


def saved_row(path: Path) -> dict:
    """The summary row stored in an existing result, for resuming."""
    keys = ["puma", "taper", "alpha", "n_constraints", "converged", "n_iter",
            "newton_decrement", "objective", "max_abs_z", "mahalanobis", "log_evidence", "seconds", "error"]
    with np.load(path) as saved:
        # Results written before a column existed resume with it blank.
        return {key: saved[key].item() if key in saved else np.nan for key in keys}


def run_solve(
    area: StudyArea, cores: int, alphas: list[float], tapers: list[str], out: Path | None
) -> None:
    import json

    root = inputs_root(area)
    pumas = sorted(p for p in root.glob("*") if (p / "manifest.json").exists())
    if not pumas:
        raise SystemExit(f"no assembled problems under {root}; run the assemble step first")
    out = out or processed_dir() / "runs" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    rows, tasks = [], []
    for path in pumas:
        manifest = json.loads((path / "manifest.json").read_text())
        for taper in (None if t == "none" else t for t in tapers):
            for alpha in alphas:
                done = out / f"{result_name(path.name, taper, alpha)}.npz"
                if done.exists():
                    rows.append(saved_row(done))
                else:
                    tasks.append((estimated_cost(manifest, taper), path, taper, alpha))
    tasks.sort(key=lambda task: task[0], reverse=True)

    logger.info(
        "solve %s: %d to run, %d already in %s", area.slug, len(tasks), len(rows), out
    )
    if tasks:
        workers = min(cores, len(tasks))
        threads = max(1, cores // workers)
        logger.info("%d workers x %d BLAS thread(s), largest problems first", workers, threads)
        with pool(workers, threads) as executor:
            futures = [executor.submit(solve_one, path, taper, alpha, out)
                       for _, path, taper, alpha in tasks]
            for count, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                logger.info(
                    "%d of %d done: %s %s a=%g %s",
                    count, len(tasks), row["puma"], row["taper"], row["alpha"],
                    f"FAILED {row['error']}" if row["error"] else
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
        run_assemble(area, cores, args.rebuild)
    else:
        run_solve(area, cores, args.alpha, args.taper, args.out)


if __name__ == "__main__":
    main()
