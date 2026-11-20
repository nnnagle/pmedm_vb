"""Step 3b of the latent-population PMEDM: the basis of the zone deviations, a
review of step 3 (``PLAN_latent_step3b.md`` version 1; model
``MODEL_latent.qmd`` version 3).

    python experiments/latent/step3b_basis.py --config experiments/embed/stage1_v2.json \
        --regions experiments/latent/regions_tn.json \
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d16_g0.5_r0.01 \
        --step1 $PMEDM_VB_RUNS/latent/step1_sigma --step3 $PMEDM_VB_RUNS/latent/step3 \
        --out $PMEDM_VB_RUNS/latent/step3b --workers 16 --threads 1
    python experiments/latent/collect_step3b.py --root $PMEDM_VB_RUNS/latent/step3b

The zone deviations get their own basis ``g``:
``p_b ∝ q_i γ^{d(A,i)} exp{θ_A' h(z_i) + u_b' g(z_i)}``, ``u_b ~ N(0, τ² I)``,
with ``h`` and ``θ_A`` as in step 3 (:mod:`pmedm_vb.latent.deviation_basis`,
:func:`~pmedm_vb.latent.region_engine.deviation_engine`). Leave-zone-out
mode only (§3).

**Before the work units** (main process):

* the bases ``g`` (``--bases``: ``kern30``, ``kern100``, ``kern200``; step 1's
  saved files where it has them, else built by step 1's recipe), recorded in
  ``<out>/bases.json``;
* ``τ^g_ref`` per basis from the 58 PUMAs' tilts on ``g``
  (``<out>/tau_ref.json``, ``<out>/tau_ref_<g>.npz``; reused when present);
* the split of transfer set B into B1 (choosing ``τ``) and B2 (scoring
  transfer), by table, read from ``--split`` (default
  ``experiments/latent/step3b_split.json``, fixed before the run) and checked
  to partition B; recorded in ``<out>/settings.json``;
* per region, from step 3's output root (``--step3``, not rerun): ``θ̂_A``
  (``<region>/reference.npz``; recomputed only as a check, the largest
  difference recorded) and the baselines ``pmedm_own``, ``pmedm_ref``,
  ``latent_u0`` and ``latent_u0:p`` (``<region>/lzo/``: their scores, rare
  groups, misfit, spread and summary rows as step 3 wrote them; their
  transfer re-scored on B1 and B2 from step 3's ``realized.npz``), written to
  ``<out>/<region>/baselines/``.

**Per region and basis** (a work unit): for each ``τ = m τ^g_ref``,
``m ∈ --tau-multiples`` (default ``{0.125, 0.25, 0.5, 1}``), in increasing
order, the MAP over step 3's ``all4`` starts with plain L-BFGS (memory 10)
(:func:`~pmedm_vb.latent.map_fit.fit_starts`; the lowest exact objective),
its Laplace evidence, the conditional PMEDM at ``p_b(Û)``, and the scores of
the realized MAP (``latent_<g>_t<m>``) and of ``p_b`` (``:p``): step 3's
scoring, and transfer misfit on B1 and on B2. Each ``τ`` is checkpointed in
``<out>/<region>/<g>/t<m>/`` (a ``done.json``), so a timed-out job resumes at
the next ``τ``. Then ``τ`` is chosen by the realized MAP's misfit on B1 (mean
over zones of the per-zone RMS z), and its rows are copied as
``latent_<g>`` (and ``latent_<g>:p``). The unit writes, under
``<out>/<region>/<g>/``, ``scores``, ``rare``, ``misfit``, ``spread``,
``summary``, ``transfer`` (``scored_set`` B1 or B2), ``fits``, ``starts``,
``tau.csv`` (per ``τ``: B1 misfit, evidence, the MAP's objective against
``U = 0``, gradient, evaluations, spread over starts, chosen) and
``done.json``.

Runs offline: ``PMEDM_VB_OFFLINE=1`` is set if the caller has not set it.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from pmedm_vb.data.cache import OFFLINE_ENV
from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.puma_check import load_check_data
from pmedm_vb.latent.area import load_basis, load_step1
from pmedm_vb.latent.atoms import load_atoms
from pmedm_vb.latent.collapsed import PairOperator
from pmedm_vb.latent.conditional import solve_conditional
from pmedm_vb.latent.deviation_basis import DEVIATION_BASES, deviation_basis, puma_tilt, tau_ref
from pmedm_vb.latent.map_fit import START_SETS, fit_starts, laplace_evidence
from pmedm_vb.latent.region_engine import (
    deviation_engine,
    region_area,
    region_contributions,
    region_sigma,
)
from pmedm_vb.latent.region_reference import (
    fit_region_reference,
    region_distance,
    zone_distributions,
    zone_log_bases,
    zone_of_atoms,
)
from pmedm_vb.latent.reference import log_base
from pmedm_vb.latent.regions import build_region, load_regions, transfer_sets
from pmedm_vb.latent.scoring import score_region, transfer_misfit
from pmedm_vb.progress import logger, record_run, save_npz, stage, write_text

_SHARED: dict = {}

MODE = "lzo"
TAU_MULTIPLES = (0.125, 0.25, 0.5, 1.0)
BASELINES = ("pmedm_own", "pmedm_ref", "latent_u0", "latent_u0:p")
FRAMES = ("scores", "rare", "misfit", "spread", "summary", "transfer")
SCORED = ("B1", "B2")
DEFAULT_SPLIT = Path(__file__).with_name("step3b_split.json")


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def statewide_view(problem, X_all, frame, zone_all):
    from dataclasses import replace

    inputs = replace(problem.inputs, X_B=X_all, units=frame)
    return replace(problem, inputs=inputs, units=np.arange(X_all.shape[0]),
                   zone_of_unit=zone_all)


def read_split(path: Path, data) -> dict[str, tuple[str, ...]]:
    split = json.loads(Path(path).read_text())
    B = transfer_sets(data)["B"]
    b1, b2 = tuple(split["B1"]), tuple(split["B2"])
    if set(b1) & set(b2) or set(b1) | set(b2) != set(B) or len(b1) + len(b2) != len(B):
        raise SystemExit(f"{path}: B1 and B2 do not partition transfer set B {B}")
    return {"B1": b1, "B2": b2}


def region_problems(region):
    s = _SHARED
    data, sets = s["data"], s["sets"]
    problems = {k: build_region(data, region, tables=sets[k]) for k in ("A", *SCORED)}
    return problems


# --------------------------------------------------------------------------
# Main-process stages
# --------------------------------------------------------------------------


def tilt_unit(unit: tuple[str, str]) -> dict:
    name, puma = unit
    s = _SHARED
    return {"basis": name, **puma_tilt(s["G"][name], s["q"], s["data"].weight, s["data"].puma,
                                       s["step1"].distances, puma, gamma=s["step1"].gamma,
                                       sigma=s["step1"].sigma)}


def run_tau_refs(args) -> dict[str, float]:
    path = args.out / "tau_ref.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    todo = [b for b in args.bases if b not in done]
    if todo:
        pumas = sorted(set(_SHARED["data"].puma))
        units = [(b, p) for b in todo for p in pumas]
        with stage(f"PUMA tilts on g for {todo} ({len(units)} fits)"):
            if args.workers > 1:
                context = multiprocessing.get_context("fork")
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
                    results = list(pool.map(tilt_unit, units))
            else:
                results = [tilt_unit(u) for u in units]
        for b in todo:
            rows = [r for r in results if r["basis"] == b]
            theta = np.array([r["theta"] for r in rows])
            done[b] = {"tau_ref": tau_ref(theta), "n_pumas": len(rows),
                       "converged": int(sum(r["converged"] for r in rows)),
                       "max_iterations": int(max(r["iterations"] for r in rows)),
                       "max_grad": float(max(r["grad_max"] for r in rows)),
                       "seconds": float(sum(r["seconds"] for r in rows))}
            save_npz(args.out / f"tau_ref_{b}.npz", theta=theta,
                     pumas=np.array([r["puma"] for r in rows]))
            logger.info("basis %s: tau_ref %.4f from %d PUMAs", b, done[b]["tau_ref"], len(rows))
        write_text(path, json.dumps(done, indent=1))
    return {b: float(done[b]["tau_ref"]) for b in args.bases}


def load_reference(args, region) -> dict:
    """θ̂_A from step 3; the reference rebuilt from it (and recomputed as a check)."""
    s = _SHARED
    data, H, step1 = s["data"], s["H"], s["step1"]
    saved = np.load(args.step3 / region.id / "reference.npz")
    theta = saved["theta"]
    distance = region_distance(step1.distances, region.pumas, data.puma)
    logb = log_base(s["q"], distance, step1.gamma)
    eta = logb + H @ theta
    qA = np.exp(eta - logsumexp(eta))
    check = fit_region_reference(region, H, s["q"], data.weight, data.puma, step1.distances,
                                 step1.gamma, step1.sigma)
    return {"theta": theta, "logb": logb, "q": qA, "zone": zone_of_atoms(region, data.puma),
            "theta_recomputed_max_abs_diff": float(np.abs(check.theta - theta).max())}


def run_baselines(args, region, ref) -> None:
    """Step 3's baselines for the region (module notes)."""
    out = args.out / region.id / "baselines"
    if (out / "done.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    s = _SHARED
    src = args.step3 / region.id / MODE
    problems = region_problems(region)
    Z = len(region.pumas)
    share = problems["A"].zone_weight / problems["A"].inputs.N
    views = {k: statewide_view(problems[k], s["X"][k], s["frame"], ref["zone"]) for k in SCORED}
    for name in FRAMES[:-1]:
        frame = pd.read_csv(src / f"{name}.csv", dtype={"zone": str})
        frame = frame[frame["model"].isin(BASELINES)].assign(basis="none")
        write_text(out / f"{name}.csv", frame.to_csv(index=False))
    realized = np.load(src / "realized.npz")
    p = {"pmedm_own": realized["pmedm_own"].astype(float),
         "pmedm_ref": realized["pmedm_ref"].astype(float),
         "latent_u0": share[:, None] * realized["latent_u0"].astype(float),
         "latent_u0:p": share[:, None] * zone_distributions(ref["q"], ref["zone"], Z, MODE)}
    rows = []
    for model, pm in p.items():
        for k in SCORED:
            other = problems[k] if model == "pmedm_own" else views[k]
            rows.append(transfer_misfit(other, pm, model, "A", k).assign(mode=MODE, basis="none"))
    write_text(out / "transfer.csv", pd.concat(rows, ignore_index=True).to_csv(index=False))
    write_text(out / "done.json", json.dumps({"source": str(src)}))


# --------------------------------------------------------------------------
# Work units
# --------------------------------------------------------------------------


def run_unit(unit: tuple[str, str]) -> tuple[str, str, float]:
    region_id, g = unit
    s, args = _SHARED, _SHARED["args"]
    out = args.out / region_id / g
    if (out / "done.json").exists():
        return region_id, g, 0.0
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    data, H, G = s["data"], s["H"], s["G"][g]
    region, ref = s["regions"][region_id], s["refs"][region_id]
    problems = region_problems(region)
    problem = problems["A"]
    inputs = problem.inputs
    Z = inputs.n_zones
    N_b = np.asarray(problem.zone_weight, float)
    share = N_b / float(inputs.N)
    sigma = region_sigma(problem, args.sigma_alpha, args.prior_sd)
    area = region_area(problem, sigma)
    views = {"A": statewide_view(problem, s["X"]["A"], s["frame"], ref["zone"]),
             **{k: statewide_view(problems[k], s["X"][k], s["frame"], ref["zone"])
                for k in SCORED}}
    se_cells = np.sqrt(inputs.sigma_v)
    engine = deviation_engine(area, H, ref["theta"], G,
                              zone_log_bases(ref["logb"], ref["zone"], Z, MODE), s["pairs"])
    theta0 = np.zeros(G.shape[1])
    tau_g = s["tau_refs"][g]
    tag = {"region": region_id, "mode": MODE, "basis": g}
    kw = {"gtol": args.map_gtol, "max_iter": args.map_max_iter, "optimizer": args.optimizer}
    previous, rows = None, []
    for m in args.tau_multiples:
        tdir = out / f"t{m:g}"
        model = f"latent_{g}_t{m:g}"
        if (tdir / "done.json").exists():
            previous = np.load(tdir / "map.npz")["U"]
            rows.append(json.loads((tdir / "done.json").read_text()))
            continue
        tdir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        tau = m * tau_g
        with stage(f"region {region_id} g = {g}: MAP at tau = {m:g} tau_ref^g "
                   f"({tau:.4g})") as step:
            fit, starts = fit_starts(engine, theta0, tau, args.map_starts, previous, **kw)
            laplace_evidence(engine, theta0, fit)
            values = [f.objective for f in starts]
            step.detail = (f"MAP from {fit.start}; spread {max(values) - min(values):.2f}; "
                           f"{sum(f.evaluations for f in starts)} evaluations")
        p_b = engine.probabilities(theta0, fit.U).T
        res = solve_conditional(s["X"]["A"], p_b, N_b, inputs.targets(), sigma,
                                tol=args.cond_tol, max_iter=args.cond_max_iter)
        frames = {k: [] for k in FRAMES}
        for name, pm in ((model, share[:, None] * res.r), (f"{model}:p", share[:, None] * p_b)):
            scored = score_region(data, views["A"], pm, name, se_cells)
            scored["summary"]["allocation_accuracy"] = np.nan
            scored["transfer"] = pd.concat([transfer_misfit(views[k], pm, name, "A", k)
                                            for k in SCORED], ignore_index=True)
            for k, v in scored.items():
                frames[k].append(v.assign(mode=MODE, basis=g, tau_multiple=m))
        for k, parts in frames.items():
            write_text(tdir / f"{k}.csv", pd.concat(parts, ignore_index=True).to_csv(index=False))
        write_text(tdir / "starts.csv", pd.DataFrame(
            [{**tag, "tau_multiple": m, **f.info(), "selected": f is fit} for f in starts])
            .to_csv(index=False))
        write_text(tdir / "fits.csv", pd.DataFrame(
            [{**tag, "model": model, "kind": "conditional", **res.info()}]).to_csv(index=False))
        save_npz(tdir / "map.npz", U=fit.U, r=res.r.astype(np.float32), lam=res.lam)
        transfer = frames["transfer"]
        b1 = pd.concat(transfer)
        b1 = b1[(b1["model"] == model) & (b1["scored_set"] == "B1")]["rms_z"].mean()
        row = {**tag, "tau_multiple": m, "tau": tau, "tau_ref_g": tau_g, "K_g": G.shape[1],
               **fit.info(), "n_starts": len(starts), "spread": max(values) - min(values),
               "evaluations_all_starts": sum(f.evaluations for f in starts),
               "seconds_all_starts": sum(f.seconds for f in starts),
               "b1_rms_z": float(b1), "conditional_converged": res.converged,
               "seconds_tau": time.perf_counter() - t0}
        write_text(tdir / "done.json", json.dumps(row, default=float))
        rows.append(row)
        previous = fit.U

    # τ chosen by the realized MAP's misfit on B1; its rows copied as latent_<g>.
    chosen = int(np.argmin([r["b1_rms_z"] for r in rows]))
    by_evidence = int(np.argmax([r["log_evidence"] for r in rows]))
    m_star = args.tau_multiples[chosen]
    for k, row in enumerate(rows):
        row["chosen_b1"], row["chosen_evidence"] = k == chosen, k == by_evidence
    for name in (*FRAMES, "starts", "fits"):
        parts = [pd.read_csv(out / f"t{m:g}" / f"{name}.csv", dtype={"zone": str})
                 for m in args.tau_multiples]
        frame = pd.concat(parts, ignore_index=True)
        if name in FRAMES:
            src = f"latent_{g}_t{m_star:g}"
            extra = [frame[frame["model"] == src + suffix].assign(model=f"latent_{g}" + suffix)
                     for suffix in ("", ":p")]
            frame = pd.concat([frame, *extra], ignore_index=True)
        write_text(out / f"{name}.csv", frame.to_csv(index=False))
    write_text(out / "tau.csv", pd.DataFrame(rows).to_csv(index=False))
    seconds = time.perf_counter() - start
    write_text(out / "done.json", json.dumps({
        **tag, "seconds_this_job": seconds, "chosen_tau_multiple_b1": m_star,
        "chosen_tau_multiple_evidence": args.tau_multiples[by_evidence],
        "seconds_all_taus": float(sum(r["seconds_tau"] for r in rows)),
        "peak_gb": peak_gb(), "Z": Z, "K_g": int(G.shape[1]), "threads": args.threads}))
    logger.info("region %s g = %s: done (chosen tau = %g tau_ref^g by B1)", region_id, g, m_star)
    return region_id, g, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="stage 1 run directory (z)")
    parser.add_argument("--step1", required=True, type=Path, help="step 1 run root")
    parser.add_argument("--step3", required=True, type=Path, help="step 3 output root")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--region", nargs="+", default=None, help="subset of region ids")
    parser.add_argument("--bases", nargs="+", default=list(DEVIATION_BASES),
                        choices=list(DEVIATION_BASES))
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--tau-multiples", type=float, nargs="+", default=list(TAU_MULTIPLES))
    parser.add_argument("--map-starts", default="all4", choices=list(START_SETS))
    parser.add_argument("--optimizer", default="m10", choices=["m10", "lm"])
    parser.add_argument("--sigma-alpha", type=float, default=0.01)
    parser.add_argument("--prior-sd", type=float, default=3.0)
    parser.add_argument("--map-gtol", type=float, default=1e-4)
    parser.add_argument("--map-max-iter", type=int, default=20000)
    parser.add_argument("--cond-tol", type=float, default=1e-10)
    parser.add_argument("--cond-max-iter", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    args.tau_multiples = sorted(args.tau_multiples)
    os.environ.setdefault(OFFLINE_ENV, "1")
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)

    config = load_config(args.config)
    regions = {r.id: r for r in load_regions(args.regions)}
    chosen = args.region or list(regions)
    unknown = [r for r in chosen if r not in regions]
    if unknown:
        raise SystemExit(f"unknown regions {unknown}")
    missing = [r for r in chosen if not (args.step3 / r / MODE / "done.json").exists()]
    if missing:
        raise SystemExit(f"step 3 has no finished {MODE} results for {missing} under {args.step3}")
    step1 = load_step1(args.step1)
    with stage("load PUMS, tables and replicate weights"):
        data = load_check_data(config)
    with stage("atoms"):
        atoms = load_atoms(data, args.run)
    if not pd.Index(atoms.serial.astype(str)).equals(pd.Index(step1.serial)):
        raise SystemExit("the atoms are not step 1's (SERIALNO order differs)")
    bases_dir = step1.root / "inputs" / "bases"
    H = load_basis(bases_dir, step1.h, atoms, args.run)
    G, basis_info = {}, {}
    for name in args.bases:
        with stage(f"deviation basis {name}") as step:
            G[name], basis_info[name] = deviation_basis(name, atoms, bases_dir, args.run,
                                                        save_to=args.out / "inputs" / "bases")
            step.detail = f"K_g = {G[name].shape[1]} ({basis_info[name]['source']})"
    del atoms
    write_text(args.out / "bases.json", json.dumps(basis_info, indent=1, default=str))
    sets = {"A": transfer_sets(data)["A"], **read_split(args.split, data)}
    X = {k: region_contributions(data, t) for k, t in sets.items()}
    with stage("pair operator over the S1 cells"):
        pairs = PairOperator(X["A"], workers=args.threads)
    frame = pd.DataFrame({"SERIALNO": data.units.index.astype(str), "weight": data.weight,
                          "is_group_quarters": data.units["is_group_quarters"].to_numpy()})
    q = data.weight / data.weight.sum()
    _SHARED.update(data=data, args=args, regions=regions, step1=step1, H=H, G=G, sets=sets,
                   X=X, pairs=pairs, frame=frame, q=q)
    tau_refs = run_tau_refs(args)
    refs, checks = {}, {}
    with stage(f"region references and baselines ({len(chosen)} regions)"):
        for rid in chosen:
            refs[rid] = load_reference(args, regions[rid])
            checks[rid] = refs[rid]["theta_recomputed_max_abs_diff"]
            run_baselines(args, regions[rid], refs[rid])
    cells = {k: int(X[k].shape[1]) for k in sets}
    write_text(args.out / "settings.json", json.dumps({
        "mode": MODE, "h": step1.h, "gamma": step1.gamma, "sigma_theta": step1.sigma,
        "bases": basis_info, "tau_ref_g": tau_refs, "tau_multiples": args.tau_multiples,
        "split": {k: list(v) for k, v in sets.items() if k != "A"}, "cells": cells,
        "split_file": str(args.split), "map_starts": args.map_starts,
        "optimizer": args.optimizer, "map_gtol": args.map_gtol,
        "map_max_iter": args.map_max_iter, "step1": str(step1.root), "step3": str(args.step3),
        "run": str(args.run), "theta_A_recomputed_max_abs_diff": checks}, indent=1, default=str))
    _SHARED.update(refs=refs, tau_refs=tau_refs)
    # Largest units first (zones x K_g), so the pool's last units are short.
    units = sorted(((r, g) for r in chosen for g in args.bases),
                   key=lambda u: -len(regions[u[0]].pumas) * G[u[1]].shape[1])
    logger.info("%d work units (%d regions x %d bases), %d processes x %d threads", len(units),
                len(chosen), len(args.bases), args.workers, args.threads)
    if args.workers > 1 and len(units) > 1:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for region_id, g, seconds in pool.map(run_unit, units):
                logger.info("unit %s %s finished (%.0fs)", region_id, g, seconds)
    else:
        for unit in units:
            run_unit(unit)


if __name__ == "__main__":
    main()
