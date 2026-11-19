"""Step 3, §4.3a: the MAP diagnostic, run before the full regional run
(``PLAN_latent_step3.md`` version 2; model ``MODEL_latent.qmd`` version 2).

    python experiments/latent/step3_diagnostic.py --config experiments/embed/stage1_v2.json \
        --regions experiments/latent/regions_tn.json \
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d16_g0.5_r0.01 \
        --step1 $PMEDM_VB_RUNS/latent/step1_sigma \
        --out $PMEDM_VB_RUNS/latent/step3_diagnostic --workers 12
    python experiments/latent/collect_step3_diagnostic.py \
        --root $PMEDM_VB_RUNS/latent/step3_diagnostic

Regions ``knoxville_core`` and ``upper_cumberland`` (``--region``),
leave-zone-out mode, the real φ; the same inputs as ``step3_regional.py``
(``θ̂_A`` on the region's records, ``Σ``, the engine). For each
``τ ∈ {0.5, 1, 2} × τ_ref``, eight fits (:mod:`pmedm_vb.latent.map_fit`):

* on the exact objective: plain L-BFGS with memory 10 (``exact_m10``) and 20
  (``exact_m20``), L-BFGS preconditioned by the Gauss–Newton Hessian refreshed
  every 30 iterations (``exact_gn``), and plain L-BFGS (memory 10) from the
  previous ``τ``'s solution (``exact_warm``; none for the first ``τ``);
* the same four on the fixed-S objective (``fixedS_*``; ``S`` held at
  ``S(0)``).

**The previous τ's solution** (a choice made here; the plan does not say which
of the previous ``τ``'s fits): the one with the lowest value of the same
objective among that ``τ``'s ``m10``, ``m20`` and ``gn`` fits.

**Work.** Phase 1: one unit per (region, objective, optimizer) for ``m10``,
``m20`` and ``gn``, each over the ``τ`` in increasing order, every fit from
``U = 0``. Phase 2: one unit per (region, objective, ``τ`` after the first),
the warm start. Units run in ``--workers`` processes. Each fit writes
``<out>/<region>/<fit>_t<m>.json`` (its :meth:`~pmedm_vb.latent.map_fit.MapFit.info`)
and ``.npz`` (``Û``); a fit already written is skipped, so a timed-out job can
be resubmitted unchanged.

**Comparison** (in the main process, after both phases), per region and
``τ``, against the fit with the lowest exact objective: ``‖ΔU‖`` and the
largest change in fitted zone shares on the scoring tables (``p_b``'s share
of each category within its table, over zones, tables and categories). The
ladder and scoring are not run. Written to ``<out>/fits.csv``, one row per
fit: final exact objective, fixed-S objective (fixed-S fits), largest
gradient component of the fitted objective and of the exact one, iterations,
evaluations, seconds, convergence, ``‖ΔU‖`` and the share change.
``collect_step3_diagnostic.py`` applies the agreement test of §4.3b.

Runs offline: ``PMEDM_VB_OFFLINE=1`` is set if the caller has not set it.
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

from pmedm_vb.data.cache import OFFLINE_ENV
from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.puma_check import load_check_data
from pmedm_vb.latent.area import load_basis, load_step1
from pmedm_vb.latent.atoms import load_atoms
from pmedm_vb.latent.collapsed import PairOperator
from pmedm_vb.latent.map_fit import OPTIMIZERS, run_fit
from pmedm_vb.latent.region_engine import (
    RegionLikelihood,
    region_area,
    region_contributions,
    region_sigma,
)
from pmedm_vb.latent.region_reference import fit_region_reference, zone_log_bases
from pmedm_vb.latent.regions import build_region, load_regions, transfer_sets
from pmedm_vb.progress import logger, record_run, save_npz, stage, write_text

_SHARED: dict = {}

REGIONS = ("knoxville_core", "upper_cumberland")
TAU_REF = 0.284
TAU_MULTIPLES = (0.5, 1.0, 2.0)
OBJECTIVES = ("exact", "fixedS")


def fit_path(region: str, name: str, m: float) -> Path:
    return _SHARED["args"].out / region / f"{name}_t{m:g}"


def save_fit(path: Path, fit, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(path.with_suffix(".npz"), U=fit.U)
    write_text(path.with_suffix(".json"), json.dumps({**extra, **fit.info()}, default=float))


def load_fit(path: Path) -> tuple[dict, np.ndarray]:
    return json.loads(path.with_suffix(".json").read_text()), \
        np.load(path.with_suffix(".npz"))["U"]


def kwargs() -> dict:
    a = _SHARED["args"]
    return {"gtol": a.map_gtol, "max_iter": a.max_iter, "refresh": a.refresh}


def phase1(unit: tuple[str, str, str]) -> str:
    region, objective, optimizer = unit
    s, args = _SHARED, _SHARED["args"]
    engine, theta = s["engines"][region], s["refs"][region].theta
    name = f"{objective}_{optimizer}"
    for m in args.tau_multiples:
        path = fit_path(region, name, m)
        if path.with_suffix(".json").exists():
            continue
        with stage(f"{region} {name} tau = {m:g} tau_ref") as step:
            fit = run_fit(engine, theta, m * args.tau_ref, objective, optimizer, **kwargs())
            step.detail = (f"F {fit.objective:.3f} (F_S {fit.fixed_s_objective:.3f}), "
                           f"{fit.evaluations} evaluations, {fit.seconds:.0f}s")
        save_fit(path, fit, {"region": region, "fit": name, "objective_kind": objective,
                             "tau_multiple": m})
    return f"{region} {name}"


def phase2(unit: tuple[str, str, int]) -> str:
    region, objective, k = unit
    s, args = _SHARED, _SHARED["args"]
    engine, theta = s["engines"][region], s["refs"][region].theta
    m, prev = args.tau_multiples[k], args.tau_multiples[k - 1]
    name = f"{objective}_warm"
    path = fit_path(region, name, m)
    if path.with_suffix(".json").exists():
        return f"{region} {name} {m:g}"
    key = "objective" if objective == "exact" else "fixed_s_objective"
    candidates = [load_fit(fit_path(region, f"{objective}_{o}", prev)) for o in OPTIMIZERS]
    info, U0 = min(candidates, key=lambda c: c[0][key])
    with stage(f"{region} {name} tau = {m:g} tau_ref (from {info['fit']} at {prev:g})") as step:
        fit = run_fit(engine, theta, m * args.tau_ref, objective, "m10", U0=U0, **kwargs())
        step.detail = (f"F {fit.objective:.3f} (F_S {fit.fixed_s_objective:.3f}), "
                       f"{fit.evaluations} evaluations, {fit.seconds:.0f}s")
    fit.origin = f"previous:{info['fit']}"
    save_fit(path, fit, {"region": region, "fit": name, "objective_kind": objective,
                         "tau_multiple": m})
    return f"{region} {name} {m:g}"


def zone_shares(data, engine, theta, U) -> np.ndarray:
    """Every scoring table's category shares per zone (zones × all categories)."""
    p = engine.probabilities(theta, U)
    out = []
    for name in data.sets.scoring:
        totals = np.asarray(data.counts[name].T @ p).T          # (Z, categories)
        out.append(totals / np.maximum(totals.sum(axis=1, keepdims=True), 1e-300))
    return np.hstack(out)


def compare(args) -> None:
    s = _SHARED
    rows = []
    for region in s["chosen"]:
        engine, theta = s["engines"][region], s["refs"][region].theta
        for m in args.tau_multiples:
            fits = []
            for objective in OBJECTIVES:
                for o in (*OPTIMIZERS, "warm"):
                    path = fit_path(region, f"{objective}_{o}", m)
                    if path.with_suffix(".json").exists():
                        fits.append(load_fit(path))
            if not fits:
                continue
            best_info, best_U = min(fits, key=lambda f: f[0]["objective"])
            ref_shares = zone_shares(s["data"], engine, theta, best_U)
            for info, U in fits:
                shares = zone_shares(s["data"], engine, theta, U)
                rows.append({**info, "best_exact_fit": best_info["fit"],
                             "dU_to_best": float(np.linalg.norm(U - best_U)),
                             "max_share_change": float(np.abs(shares - ref_shares).max())})
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["fitted_grad_max"] = np.where(frame["objective_kind"] == "fixedS",
                                            frame["fixed_s_grad_max"], frame["grad_max"])
    write_text(args.out / "fits.csv", frame.to_csv(index=False))
    logger.info("comparison of %d fits written to %s", len(frame), args.out / "fits.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="stage 1 run directory (z)")
    parser.add_argument("--step1", required=True, type=Path, help="step 1 run root")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--region", nargs="+", default=list(REGIONS))
    parser.add_argument("--mode", default="lzo", choices=["lzo", "own"])
    parser.add_argument("--tau-ref", type=float, default=TAU_REF)
    parser.add_argument("--tau-multiples", type=float, nargs="+", default=list(TAU_MULTIPLES))
    parser.add_argument("--sigma-alpha", type=float, default=0.01)
    parser.add_argument("--prior-sd", type=float, default=3.0)
    parser.add_argument("--map-gtol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=20000,
                        help="iteration cap per fit (reduce for a smoke run)")
    parser.add_argument("--refresh", type=int, default=30,
                        help="Gauss-Newton preconditioner refreshed every this many iterations")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    args.tau_multiples = sorted(args.tau_multiples)
    os.environ.setdefault(OFFLINE_ENV, "1")
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)

    config = load_config(args.config)
    regions = {r.id: r for r in load_regions(args.regions)}
    unknown = [r for r in args.region if r not in regions]
    if unknown:
        raise SystemExit(f"unknown regions {unknown}")
    step1 = load_step1(args.step1)
    with stage("load PUMS, tables and replicate weights"):
        data = load_check_data(config)
    with stage("atoms"):
        atoms = load_atoms(data, args.run)
    if not pd.Index(atoms.serial.astype(str)).equals(pd.Index(step1.serial)):
        raise SystemExit("the atoms are not step 1's (SERIALNO order differs)")
    H = load_basis(step1.root / "inputs" / "bases", step1.h, atoms, args.run)
    del atoms
    sets = transfer_sets(data)
    with stage("pair operator over the S1 cells"):
        pairs = PairOperator(region_contributions(data, sets["A"]), workers=args.threads)
    q = data.weight / data.weight.sum()
    refs, engines = {}, {}
    for rid in args.region:
        region = regions[rid]
        ref = fit_region_reference(region, H, q, data.weight, data.puma, step1.distances,
                                   step1.gamma, step1.sigma)
        problem = build_region(data, region, tables=sets["A"])
        area = region_area(problem, region_sigma(problem, args.sigma_alpha, args.prior_sd))
        engines[rid] = RegionLikelihood(area, H, zone_log_bases(ref.logb, ref.zone, area.B,
                                                                args.mode), pairs)
        refs[rid] = ref
        logger.info("region %s: %d zones, theta_A in %d steps", rid, area.B, ref.fit.iterations)
    write_text(args.out / "settings.json", json.dumps({
        "regions": args.region, "mode": args.mode, "tau_ref": args.tau_ref,
        "tau_multiples": args.tau_multiples, "h": step1.h, "gamma": step1.gamma,
        "sigma_theta": step1.sigma, "map_gtol": args.map_gtol, "max_iter": args.max_iter,
        "refresh": args.refresh, "step1": str(step1.root), "run": str(args.run)}, indent=1))
    _SHARED.update(data=data, args=args, chosen=args.region, refs=refs, engines=engines)

    phases = [
        (phase1, [(r, o, opt) for r in args.region for o in OBJECTIVES for opt in OPTIMIZERS]),
        (phase2, [(r, o, k) for r in args.region for o in OBJECTIVES
                  for k in range(1, len(args.tau_multiples))]),
    ]
    start = time.perf_counter()
    for fn, units in phases:
        if args.workers > 1 and len(units) > 1:
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
                for done in pool.map(fn, units):
                    logger.info("unit %s finished", done)
        else:
            for unit in units:
                fn(unit)
    compare(args)
    write_text(args.out / "done.json", json.dumps({"seconds": time.perf_counter() - start}))


if __name__ == "__main__":
    main()
