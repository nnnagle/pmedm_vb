"""Step 1 of the latent-population PMEDM: locating each PUMA in type space
(``PLAN_latent_step1.md``; model ``MODEL_latent.qmd``).

    python experiments/latent/step1_reference.py grid --config experiments/embed/stage1_v2.json \
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d16_g0.5_r0.01 \
        --grid experiments/latent/step1_grid.json --out $PMEDM_VB_RUNS/latent/step1 --workers 16
    python experiments/latent/collect_step1.py --root $PMEDM_VB_RUNS/latent/step1
    python experiments/latent/step1_reference.py final --config ... --run ... --grid ... \
        --out $PMEDM_VB_RUNS/latent/step1 --workers 16
    python experiments/latent/collect_step1.py --root $PMEDM_VB_RUNS/latent/step1

**grid** (the cross-fitted reproduction check, §3-§5). Builds, once:

* the atoms (:mod:`pmedm_vb.latent.atoms`) and their checks (``inputs/atoms.json``);
* ``T`` (``inputs/T/``), the PUMA distances (``inputs/distances.csv``,
  ``inputs/distances.json``) and every basis (``inputs/bases/``: K, dropped
  columns, condition number, standardization, kernel centres).

Then, per PUMA and basis (a work unit, run in ``--workers`` processes), for
each of the ``folds`` held-out folds: every (γ, σ_θ) of the grid
(:func:`~pmedm_vb.latent.reference.fit_reference`; within a γ the σ_θ are
fitted in increasing order, each from the previous ``θ̂``) and, in a unit of
its own per PUMA, the baselines (``θ = 0`` at each baseline γ), each scored
against the held-out fold (:mod:`pmedm_vb.latent.reference_check`). A unit
writes ``grid/<basis>/<puma>/``: ``fits.csv`` (one row per fit: iterations,
final gradient, convergence, ESS, share outside the PUMA, seconds),
``cells.csv``, ``tables.csv``, ``rare.csv`` and ``theta.npz`` (``θ̂`` of every
fit), then ``done.json``; a unit with ``done.json`` is skipped, so a
timed-out job can be resubmitted unchanged. ``q^P`` of the cross-fitted fits
is not stored (it follows from ``θ̂``, the basis and γ).

**final** (§5, §8). At the chosen (h, γ, σ_θ) -- ``decisions.json`` written by
``collect_step1.py``, or ``--h``, ``--gamma``, ``--sigma`` -- fits ``θ_P`` on all
of each PUMA's records (support: every atom) and writes ``final/``:
``reference.npz`` (PUMAs, ``θ̂`` per PUMA, ``q^P`` per PUMA over the atoms,
the atoms' ``SERIALNO``), ``fits.csv``, ``rare.csv`` (in-sample: ``q^P``
against the PUMA's own weighted totals) and ``settings.json``. The optional
secondary check (§4.6) adds the published PUMA-level tables to ``rare.csv``
when they are in the API cache; it reads the cache only and is skipped, with a
log message, when they are not.

Runs offline: set ``PMEDM_VB_OFFLINE=1`` (the sbatch script does).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from pmedm_vb.data.cache import OFFLINE_ENV
from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.puma_check import (
    RARE,
    load_check_data,
    published_tables,
    rare_groups,
)
from pmedm_vb.latent.atoms import (
    Contributions,
    atom_checks,
    build_contributions,
    load_atoms,
)
from pmedm_vb.latent.basis import build_basis
from pmedm_vb.latent.geography import atom_distance, distance_checks, load_distances
from pmedm_vb.latent.reference import fit_reference, log_base
from pmedm_vb.latent.reference_check import (
    assign_folds,
    cell_groups,
    heldout_cells,
    rare_columns,
    score_fit,
)
from pmedm_vb.progress import logger, record_run, save_npz, stage, write_text

#: Filled before the worker pool forks, so workers share it without pickling.
_SHARED: dict = {}


def load_inputs(args, grid: dict, bases: list[str]) -> None:
    """Data, atoms, T, distances and the requested bases into ``_SHARED``;
    the inputs and their checks written under ``<out>/inputs``."""
    config = load_config(args.config)
    area = config.area
    root = args.out / "inputs"
    root.mkdir(parents=True, exist_ok=True)
    with stage("load PUMS, tables and replicate weights"):
        data = load_check_data(config)
    with stage("atoms"):
        atoms = load_atoms(data, args.run)
    checks = atom_checks(atoms)
    write_text(root / "atoms.json", json.dumps(checks, indent=1))
    logger.info("atoms: M = %s, z mean max |.| %.2e, cov max |. - I| %.2e", f"{checks['M']:,}",
                checks["z_mean_max_abs"], checks["z_cov_max_abs_dev_from_I"])
    t_dir = root / "T"
    if (t_dir / "T_cells.json").exists():
        contributions = Contributions.load(t_dir)
        if contributions.serial is None or not contributions.serial.equals(
                pd.Index(atoms.serial.astype(str), name="SERIALNO")):
            raise SystemExit(f"{t_dir}: T's rows are not the atoms; remove it to rebuild")
    else:
        contributions = build_contributions(area, atoms.serial, grid.get("tables", "all"))
        contributions.save(t_dir)
    D = load_distances(area)
    dcheck = distance_checks(D)
    D.to_csv(root / "distances.csv")
    write_text(root / "distances.json", json.dumps(dcheck, indent=1))
    if not (dcheck["symmetric"] and dcheck["zero_diagonal"] and dcheck["connected"]):
        raise SystemExit(f"PUMA distances fail their checks: {dcheck}")
    missing = sorted(set(atoms.pumas) - set(D.index))
    if missing:
        raise SystemExit(f"PUMAs without boundaries: {missing}")
    built = {}
    for name in bases:
        with stage(f"basis {name}") as step:
            basis = build_basis(name, atoms, args.run, seed=grid.get("kmeans_seed", 1),
                                kmeans_iter=grid.get("kmeans_iterations", 100),
                                tol=grid.get("collinearity_tol", 1e-9))
            basis.save(root / "bases")
            built[name] = basis
            step.detail = f"K = {basis.K}, dropped {len(basis.dropped)}, " \
                          f"condition {basis.condition:.3g}"
    _SHARED.update(area=area, data=data, atoms=atoms, contributions=contributions, D=D,
                   bases=built, groups=cell_groups(data, contributions),
                   rare_cols=rare_columns(data), grid=grid, args=args)


def _write_frames(directory: Path, frames: dict[str, list[dict]]) -> None:
    for name, rows in frames.items():
        write_text(directory / f"{name}.csv", pd.DataFrame(rows).to_csv(index=False))


def run_unit(unit: tuple[str, str]) -> tuple[str, str, float]:
    """One PUMA and one basis (or ``baseline``) over every fold; see the module notes."""
    name, puma = unit
    s = _SHARED
    data, atoms, grid, args = s["data"], s["atoms"], s["grid"], s["args"]
    out = args.out / "grid" / name / puma
    if (out / "done.json").exists():
        return name, puma, 0.0
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    q = atoms.q
    in_puma = atoms.puma == puma
    distance = atom_distance(s["D"], puma, atoms.puma)
    k = int(grid["folds"])
    fold = assign_folds(atoms, puma, k, grid.get("fold_seed", 1))
    frames = {"fits": [], "cells": [], "tables": [], "rare": []}
    thetas, keys = [], []

    def record(base: dict, p: np.ndarray, info: dict, obs, held) -> None:
        scored = score_fit(data, p, s["groups"], obs, held, puma, s["rare_cols"])
        frames["fits"].append({**base, **info})
        for kind in ("cells", "tables", "rare"):
            frames[kind].extend({**base, **r} for r in scored[kind])

    for f in range(k):
        held = in_puma & (fold == f)
        train = in_puma & (fold != f)
        obs = heldout_cells(atoms, s["groups"], held)
        common = {"puma": puma, "fold": f, "n_train": int(train.sum()), "n_held": obs.n}
        if name == "baseline":
            for label, gamma in grid["baselines"].items():
                t0 = time.perf_counter()
                logb = log_base(q, distance, float(gamma), exclude=held)
                p = np.exp(logb - logsumexp(logb))
                info = {"iterations": 0, "converged": True, "grad_max": np.nan,
                        "ess": float(1 / np.square(p).sum()),
                        "share_outside": float(p[~in_puma].sum()),
                        "seconds": time.perf_counter() - t0}
                record({"basis": f"baseline_{label}", "gamma": float(gamma), "sigma": np.nan,
                        **common}, p, info, obs, held)
            continue
        basis = s["bases"][name]
        H = basis.H
        w_train = atoms.weight[train]
        mu = (w_train / w_train.sum()) @ H[train]
        n = int(train.sum())
        for gamma in grid["gammas"]:
            gamma = float(gamma)
            logb = log_base(q, distance, gamma, exclude=held)
            rows = np.flatnonzero(train) if gamma == 0 else None
            theta0 = None
            for sigma in sorted(float(x) for x in grid["sigmas"]):
                fit = fit_reference(H, logb, mu, n, sigma, rows=rows, theta0=theta0,
                                    in_puma=in_puma, tol=grid.get("newton_tol", 1e-8),
                                    max_iter=grid.get("newton_max_iter", 100))
                theta0 = fit.theta
                info = {"iterations": fit.iterations, "converged": fit.converged,
                        "grad_max": fit.grad_max, "objective": fit.objective,
                        "ess": fit.ess, "share_outside": fit.share_outside,
                        "theta_norm": float(np.linalg.norm(fit.theta)),
                        "seconds": fit.seconds, "message": fit.message}
                record({"basis": name, "gamma": gamma, "sigma": sigma, **common}, fit.p, info,
                       obs, held)
                thetas.append(fit.theta)
                keys.append(f"{f}|{gamma:g}|{sigma:g}")
                if not fit.converged:
                    logger.warning("PUMA %s %s fold %d gamma %g sigma %g: %s after %d "
                                   "iterations (gradient %.2e)", puma, name, f, gamma, sigma,
                                   fit.message, fit.iterations, fit.grad_max)
    _write_frames(out, frames)
    if thetas:
        save_npz(out / "theta.npz", theta=np.array(thetas), keys=np.array(keys),
                 columns=np.array(s["bases"][name].columns))
    seconds = time.perf_counter() - start
    write_text(out / "done.json", json.dumps({"seconds": seconds, "fits": len(frames["fits"])}))
    return name, puma, seconds


def run_grid(args, grid: dict) -> None:
    bases = args.bases or list(grid["bases"])
    load_inputs(args, grid, bases)
    atoms = _SHARED["atoms"]
    pumas = args.puma or atoms.pumas
    unknown = sorted(set(pumas) - set(atoms.pumas))
    if unknown:
        raise SystemExit(f"unknown PUMAs {unknown}")
    # Largest bases first, so the pool's last units are short.
    order = sorted(bases, key=lambda b: -_SHARED["bases"][b].K)
    units = [(b, p) for b in order for p in pumas] + [("baseline", p) for p in pumas]
    logger.info("grid: %d PUMAs x %d bases + baselines = %d units, %d workers", len(pumas),
                len(bases), len(units), args.workers)
    if args.workers > 1:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for name, puma, seconds in pool.map(run_unit, units):
                logger.info("unit %s %s done in %.0fs", name, puma, seconds)
    else:
        for unit in units:
            name, puma, seconds = run_unit(unit)
            logger.info("unit %s %s done in %.0fs", name, puma, seconds)


# --------------------------------------------------------------------------
# Final fits
# --------------------------------------------------------------------------


def cached_published(area) -> pd.DataFrame | None:
    """The published PUMA-level tables of the rare groups, from the API cache
    only (``None``, with a log message, when any is not cached)."""
    from pmedm_vb.data.acs_api import values_path

    tables = sorted({t for t, _ in RARE if t != "unit_type"})
    absent = [t for t in tables if not values_path(area, t, "puma").exists()]
    if absent:
        logger.info("secondary check skipped: published PUMA tables not cached for %s",
                    ", ".join(absent))
        return None
    os.environ[OFFLINE_ENV] = "1"  # read the cache only, whatever the caller set
    return published_tables(area)


def run_final_puma(puma: str) -> dict:
    s = _SHARED
    data, atoms, basis, choice = s["data"], s["atoms"], s["basis"], s["choice"]
    in_puma = atoms.puma == puma
    distance = atom_distance(s["D"], puma, atoms.puma)
    logb = log_base(atoms.q, distance, choice["gamma"])
    w = atoms.weight[in_puma]
    mu = (w / w.sum()) @ basis.H[in_puma]
    n = int(in_puma.sum())
    rows = np.flatnonzero(in_puma) if choice["gamma"] == 0 else None
    fit = fit_reference(basis.H, logb, mu, n, choice["sigma"], rows=rows, in_puma=in_puma,
                        tol=s["grid"].get("newton_tol", 1e-8),
                        max_iter=s["grid"].get("newton_max_iter", 100))
    N = float(w.sum())
    rare = [{"puma": puma, **r}
            for r in rare_groups(data, fit.p, N, in_puma, s["published"], puma)]
    for r in rare:
        if "z_pred" in r:
            r["moe_units_pred"] = r["z_pred"] / 1.645
    info = {"puma": puma, "n": n, "iterations": fit.iterations, "converged": fit.converged,
            "grad_max": fit.grad_max, "objective": fit.objective, "ess": fit.ess,
            "share_outside": fit.share_outside, "seconds": fit.seconds, "message": fit.message,
            "max_moment_error": float(np.abs(fit.p @ basis.H - mu).max())}
    return {"info": info, "rare": rare, "theta": fit.theta, "p": fit.p}


def run_final(args, grid: dict) -> None:
    decisions = args.out / "decisions.json"
    choice = json.loads(decisions.read_text()) if decisions.exists() else {}
    for key in ("h", "gamma", "sigma"):
        if getattr(args, key) is not None:
            choice[key] = getattr(args, key)
    if not all(k in choice for k in ("h", "gamma", "sigma")):
        raise SystemExit(f"no decisions: run collect_step1.py first ({decisions}) or give "
                         "--h, --gamma and --sigma")
    choice = {"h": str(choice["h"]), "gamma": float(choice["gamma"]),
              "sigma": float(choice["sigma"])}
    load_inputs(args, grid, [choice["h"]])
    s = _SHARED
    atoms = s["atoms"]
    s.update(basis=s["bases"][choice["h"]], choice=choice, published=cached_published(s["area"]))
    pumas = args.puma or atoms.pumas
    logger.info("final fits at h = %s, gamma = %g, sigma = %g for %d PUMAs", choice["h"],
                choice["gamma"], choice["sigma"], len(pumas))
    if args.workers > 1:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            results = list(pool.map(run_final_puma, pumas))
    else:
        results = [run_final_puma(p) for p in pumas]
    out = args.out / "final"
    out.mkdir(parents=True, exist_ok=True)
    save_npz(out / "reference.npz", pumas=np.array(pumas),
             theta=np.array([r["theta"] for r in results]),
             q=np.array([r["p"] for r in results]),
             serial=np.array([str(x) for x in atoms.serial]),
             columns=np.array(s["basis"].columns))
    write_text(out / "fits.csv", pd.DataFrame([r["info"] for r in results]).to_csv(index=False))
    write_text(out / "rare.csv", pd.DataFrame([x for r in results for x in r["rare"]])
               .to_csv(index=False))
    write_text(out / "settings.json", json.dumps({
        **choice, "basis": str(args.out / "inputs" / "bases" / choice["h"]),
        "run": str(args.run), "secondary_check": s["published"] is not None}, indent=1))
    logger.info("final fits written to %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=["grid", "final"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="stage 1 run directory (z)")
    parser.add_argument("--grid", required=True, type=Path, help="grid configuration (JSON)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--puma", nargs="+", default=None, help="subset of PUMAs")
    parser.add_argument("--bases", nargs="+", default=None, help="subset of the grid's bases")
    parser.add_argument("--h", default=None, help="final: basis (overrides decisions.json)")
    parser.add_argument("--gamma", type=float, default=None, help="final: gamma")
    parser.add_argument("--sigma", type=float, default=None, help="final: sigma_theta")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    grid = json.loads(args.grid.read_text())
    if args.bases:
        unknown = sorted(set(args.bases) - set(grid["bases"]))
        if unknown:
            raise SystemExit(f"bases {unknown} not in {args.grid}")
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    write_text(args.out / f"grid_{args.stage}.json", json.dumps(grid, indent=1))
    if args.stage == "grid":
        run_grid(args, grid)
    else:
        run_final(args, grid)


if __name__ == "__main__":
    main()
