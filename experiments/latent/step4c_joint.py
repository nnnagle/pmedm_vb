"""Step 4c of the latent-population PMEDM: ``τ`` and ``κ`` chosen together
(``PLAN_latent_step4c.md`` version 2; model ``MODEL_latent.qmd`` version 4).

    python experiments/latent/step4c_joint.py --config experiments/embed/stage1_v2.json \
        --regions experiments/latent/regions_tn.json \
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d16_g0.5_r0.01 \
        --step1 $PMEDM_VB_RUNS/latent/step1_sigma --step3 $PMEDM_VB_RUNS/latent/step3 \
        --step3b $PMEDM_VB_RUNS/latent/step3b --step4 $PMEDM_VB_RUNS/latent/step4 \
        --step4b $PMEDM_VB_RUNS/latent/step4b --out $PMEDM_VB_RUNS/latent/step4c \
        --workers 16 --threads 1 [--scal] [--p0] [--extras-only]
    python experiments/latent/collect_step4c.py --root $PMEDM_VB_RUNS/latent/step4c

**Design** (§2): as step 4b's K2 arm (regional run, leave-zone-out, ``g`` =
``kern30``, model version 4 with ``κ = c n_A``, simulations S and G, the
same replicates, seeds (step 4's, from ``--step4``'s ``settings.json``) and
realized truths, the same functionals including the zone totals).

**Selection** (``<out>/select/<sim>/<region>/c<c>/r<k>/t<m>/``): for each
simulation, region, ``c ∈ --c-grid`` (``{0.125, 0.25, 0.5}``) and the first
``--select-replicates`` (3) replicates, the ``τ`` in ``--tau-multiples``
(``{0.25, 0.5, 1, 2} × τ^g_ref``) in increasing order: the ``U`` MAP
(``all4`` starts, plain L-BFGS, the lowest exact objective; the warm start is
the previous ``τ``'s MAP for the same ``c``), its Laplace evidence (``κ`` in
the model), the conditional PMEDM over pairs and the realized MAP's scores,
and the posterior with ``--select-draws`` (50) draws: its interval statistics
and the B1 posterior predictive log density. **Reuse:** where step 4b
(``--step4b``) fitted the same ``c`` and replicate and its B1-chosen ``τ`` is
this ``τ``, its fit (MAP, posterior with its 200 draws, scores) is copied
instead (marked ``reused_from`` in ``done.json``); its Laplace evidence is
computed here.

**Choice** (``<out>/choice.json``): per simulation and region, the pair with
the largest B1 log density summed over the selection replicates; reported
beside it: the summed Laplace evidence of each pair and the pair it prefers,
the truth-best pair (lowest mean log-odds-ratio error of the realized MAP over
the selection replicates), and a flag when the chosen ``τ`` is the grid's
top.

**Final fits** (``<out>/final/<sim>/<region>/r<k>/``): the other replicates
at the chosen pair, with ``--draws`` (200) draws (the chain of ``τ`` up to the
chosen one for the warm start, the posterior at the chosen one); reused from
step 4b where the pair and replicate match.

**S-cal** (``--scal``, off by default; ``<out>/Scal/<region>/r<k>/``):
simulation S's sample calibrated to region-level controls
(:func:`~pmedm_vb.latent.calibration.draw_calibrated`), fitted with its
linearization covariance plus ``ω² I``
(:class:`~pmedm_vb.latent.calibration.FullSigma`), at S's chosen pair for
the region, all replicates, ``--draws`` draws, fitted from scratch: the MAP at
the chosen ``τ`` alone from ``U = 0`` (the ``all4`` set without its warm
start), no start from S's fits.

**P0** (``--p0``, off by default; ``<out>/P0/<region>/r<k>/``): simulation
S's data (the same ``Y`` and ``Σ̂``) with prior shares from an independent
source: ``π⁰_b = N̂ind_b / Σ N̂ind``, ``N̂ind_b`` the zone's unit total
(households plus group-quarters persons, from the unit-type cells) in a
second, independent 1-in-8 Bernoulli sample of the same realized population,
seeded by (seed, region, replicate, :data:`P0_TAG`). At S's chosen pair, all
replicates, ``--draws`` draws, from scratch as S-cal.

``--extras-only`` runs only the requested S-cal and P0 arms (the main run's
``choice.json`` must exist), so they can be separate jobs.

Every unit writes step 4's files (``posterior.csv.gz``, ``scores``, ``rare``,
``misfit``, ``transfer``, ``fits``, ``starts``, ``map.npz``) and
``done.json``; finished units are skipped, so a timed-out job can be
resubmitted unchanged.

Runs offline: ``PMEDM_VB_OFFLINE=1`` is set if the caller has not set it.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import resource
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.prior import prior_variance
from pmedm_vb.data.cache import OFFLINE_ENV
from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.puma_check import load_check_data
from pmedm_vb.latent.area import load_basis, load_step1
from pmedm_vb.latent.atoms import load_atoms
from pmedm_vb.latent.calibration import (
    BlockSigma,
    FullSigma,
    calibration_variables,
    draw_bernoulli,
    draw_calibrated,
    draw_gaussian,
    draw_truth,
    interval_stats,
    with_tables,
)
from pmedm_vb.latent.collapsed import PairOperator
from pmedm_vb.latent.deviation_basis import DEVIATION_BASES, deviation_basis
from pmedm_vb.latent.map_fit import MapFit, fit_starts, laplace_evidence, run_fit
from pmedm_vb.latent.overdispersed import (
    dirmult_phi,
    pair_engine,
    pair_posterior_samples,
    predictive_log_density,
    prior_shares,
    solve_conditional_pairs,
)
from pmedm_vb.latent.posterior import Functionals, build_functionals, zone_truth
from pmedm_vb.latent.reference import log_base
from pmedm_vb.latent.region_engine import region_area, region_contributions, region_sigma
from pmedm_vb.latent.region_reference import region_distance, zone_log_bases, zone_of_atoms
from pmedm_vb.latent.regions import build_region, load_regions, transfer_sets
from pmedm_vb.latent.scoring import score_region, transfer_misfit
from pmedm_vb.progress import logger, record_run, save_npz, write_text

_SHARED: dict = {}

MODE = "lzo"
TAU_MULTIPLES = (0.25, 0.5, 1.0, 2.0)
C_GRID = (0.125, 0.25, 0.5)
SPLIT_FILE = Path(__file__).with_name("step3b_split.json")
SCORED = ("B1", "B2")
UNIT_FILES = ("posterior.csv.gz", "scores.csv", "rare.csv", "misfit.csv", "transfer.csv",
              "fits.csv", "starts.csv", "map.npz")
SIM_INDEX = {"G": 0, "S": 1}     # step 4's seeds (SIMULATIONS order); S-cal and P0 use S's
#: Tag of P0's independent sample's seed (seed, region, replicate, P0_TAG).
P0_TAG = 4003
#: Simulations that reuse S's data (and S's chosen pair), fitted from scratch.
EXTRAS = {"Scal": "Scal", "P0": "P0"}


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def statewide_view(problem, X_all, frame, zone_all):
    inputs = replace(problem.inputs, X_B=X_all, units=frame)
    return replace(problem, inputs=inputs, units=np.arange(X_all.shape[0]),
                   zone_of_unit=zone_all)


def step4b_label(c: float) -> str:
    return "K1" if c == 1.0 else f"K2_c{c:g}"


class Replicate:
    """One (simulation, region, replicate)'s data, ``Σ`` and pair engine at a
    given ``c`` (step 4b's construction, with step 4's seeds)."""

    def __init__(self, sim: str, region_id: str, k: int):
        s, args = _SHARED, _SHARED["args"]
        data, sets, X = s["data"], s["sets"], s["X"]
        self.region, self.ref = s["regions"][region_id], s["refs"][region_id]
        index = list(s["regions"]).index(region_id)
        self.sim, self.k, self.index = sim, k, index
        truth_rng = np.random.default_rng([args.seed, index, k])
        rng = np.random.default_rng([args.seed, SIM_INDEX.get(sim, SIM_INDEX["S"]), index, k])
        problems = {key: build_region(data, self.region, tables=sets[key])
                    for key in ("A", *SCORED)}
        joint = build_region(data, self.region,
                             tables=(*sets["A"], *sets["B1"], *sets["B2"]))
        true_pop = draw_truth(joint, truth_rng)
        sizes = [len(problems[key].cells) for key in ("A", *SCORED)]
        self.cuts = np.cumsum([0, *sizes])
        if sim == "G":
            draw = draw_gaussian(joint, true_pop, rng)
        elif sim in ("S", "P0"):
            draw = draw_bernoulli(joint, true_pop, rng)
        else:
            Xcal, _ = calibration_variables(data, joint.units)
            draw = draw_calibrated(joint, true_pop, rng, Xcal, sizes[0])
        self.draw, self.true_pop, self.joint = draw, true_pop, joint
        reps = {key: with_tables(problems[key], np.arange(self.cuts[j], self.cuts[j + 1]), draw)
                for j, key in enumerate(("A", *SCORED))}
        self.pA = replace(reps["A"], zone_weight=true_pop.N.astype(float))
        inputs = self.pA.inputs
        self.Z = inputs.n_zones
        self.N_A = float(true_pop.N.sum())
        self.n_A = int(inputs.n)
        unit_cells = np.array([j for j, name in enumerate(problems["A"].cells)
                               if name.startswith("unit_type:")])
        self.shares, self.N_hat = prior_shares(draw.Y[:, :sizes[0]], unit_cells)
        self.ind = None
        if sim == "P0":   # π⁰ from an independent 1-in-8 sample of the same population
            ind = draw_bernoulli(joint, true_pop, np.random.default_rng([args.seed, index, k, P0_TAG]))
            self.shares, self.N_hat = prior_shares(ind.Y[:, :sizes[0]], unit_cells)
            self.ind = ind
        omega2 = prior_variance(args.prior_sd, inputs.n, inputs.N)
        if sim == "G":
            self.sigma = region_sigma(self.pA, args.sigma_alpha, args.prior_sd)
        elif sim in ("S", "P0"):
            blocks = [B[:sizes[0], :sizes[0]] for B in draw.blocks]
            self.sigma = BlockSigma(blocks, draw.floor[:, :sizes[0]], self.Z, omega2)
        else:
            self.sigma = FullSigma(draw.full, draw.floor[:, :sizes[0]].T.ravel(), omega2)
        self.sigma_dense = self.sigma.to_dense()
        self.area = region_area(self.pA, self.sigma)
        self.views = {key: statewide_view(reps[key], X[key], s["frame"], self.ref["zone"])
                      for key in ("A", *SCORED)}
        self.theta0 = np.zeros(s["G"].shape[1])

    def engine(self, c: float):
        s = _SHARED
        kappa = c * self.n_A
        return pair_engine(self.area, s["H"], self.ref["theta"], s["G"],
                           zone_log_bases(self.ref["logb"], self.ref["zone"], self.Z, MODE),
                           s["pairs"], self.shares, self.N_A, kappa), kappa


def fit_one(rep: Replicate, engine, kappa: float, m: float, previous, out: Path, tag: dict,
            draws: int, with_posterior: bool, warm: bool = True) -> dict:
    """MAP at ``τ = m τ^g_ref``, evidence, conditional PMEDM, scores; with
    ``with_posterior`` the Laplace posterior, its interval statistics and the
    B1 log density. Writes the unit files under ``out``."""
    s, args = _SHARED, _SHARED["args"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tau = m * s["tau_ref_g"]
    kw = {"gtol": args.map_gtol, "max_iter": args.map_max_iter, "optimizer": "m10"}
    starts_set = "all4" if warm else None
    if starts_set:
        fit, starts = fit_starts(engine, rep.theta0, tau, "all4", previous, **kw)
    else:   # from scratch: all4's starts from U = 0 (no warm start)
        starts = [run_fit(engine, rep.theta0, tau, "exact", o, gtol=args.map_gtol,
                          max_iter=args.map_max_iter) for o in ("m10", "m20", "gn")]
        for f, name in zip(starts, ("exact_m10", "exact_m20", "exact_gn")):
            f.start = name
        fit = min(starts, key=lambda f: f.objective)
    laplace_evidence(engine, rep.theta0, fit)
    seconds_map = time.perf_counter() - t0
    phi = dirmult_phi(rep.N_A, kappa)
    q = rep.shares[:, None] * engine.probabilities(rep.theta0, fit.U).T
    res = solve_conditional_pairs(s["X"]["A"], q, rep.N_A, rep.N_A / phi,
                                  rep.pA.inputs.targets(), rep.sigma_dense,
                                  tol=args.cond_tol, max_iter=args.cond_max_iter)
    tag = {**tag, "tau_multiple": m, "tau": tau, "kappa": kappa, "phi": phi}
    b1 = float(transfer_misfit(rep.views["B1"], res.r, "x", "A", "B1")["rms_z"].mean())
    se_cells = np.sqrt(rep.pA.inputs.sigma_v)
    parts = {key: [] for key in ("scores", "rare", "misfit", "transfer")}
    for name, pm in (("latent_kern30", res.r), ("latent_kern30:p", q)):
        scored = score_region(s["data"], rep.views["A"], pm, name, se_cells)
        scored["transfer"] = pd.concat([transfer_misfit(rep.views[key], pm, name, "A", key)
                                        for key in SCORED], ignore_index=True)
        for key in parts:
            parts[key].append(scored[key].assign(**tag))
    for key, p in parts.items():
        write_text(out / f"{key}.csv", pd.concat(p, ignore_index=True).to_csv(index=False))
    values = [f.objective for f in starts]
    write_text(out / "starts.csv", pd.DataFrame(
        [{**tag, **f.info(), "selected": f is fit} for f in starts]).to_csv(index=False))
    write_text(out / "fits.csv", pd.DataFrame(
        [{**tag, "model": "conditional_pairs", **res.info()}]).to_csv(index=False))
    info = {**tag, "map_start": fit.start, "map_objective": fit.objective,
            "map_grad_max": fit.grad_max, "map_converged": fit.converged,
            "spread": max(values) - min(values),
            "evaluations_all_starts": sum(f.evaluations for f in starts),
            "log_evidence": fit.log_evidence, "b1_rms_z": b1, "seconds_map": seconds_map,
            "conditional_converged": res.converged}
    if with_posterior:
        t1 = time.perf_counter()
        index = rep.index
        post_rng = np.random.default_rng(
            [args.seed, SIM_INDEX.get(rep.sim, SIM_INDEX["S"]), index, rep.k, 2,
             int(round(m * 1000)), int(round(kappa))])
        fun = s["fun"]
        sample = pair_posterior_samples(engine, rep.theta0, fit.U, tau, fun, s["X"]["A"],
                                        draws, post_rng)
        counts = np.zeros(s["data"].weight.size)
        counts[rep.joint.units] = rep.true_pop.counts
        truth = zone_truth(fun, rep.ref["zone"], counts, rep.Z)
        st = interval_stats(sample.samples, truth)
        cuts, draw = rep.cuts, rep.draw
        info["b1_logdens"] = predictive_log_density(sample.samples[:, :, fun.of("B1")],
                                                    draw.Y[:, cuts[1]:cuts[2]],
                                                    draw.var[:, cuts[1]:cuts[2]])
        frame = pd.DataFrame({
            "zone": np.repeat(np.array(rep.region.pumas), fun.F),
            "cls": np.tile(fun.cls, rep.Z), "functional": np.tile(fun.names, rep.Z),
            "truth": truth.ravel(), **{key: np.asarray(v).ravel() for key, v in st.items()}})
        fitted = fun.of("fitted")
        se = np.full((rep.Z, fun.F), np.nan)
        pub = np.full((rep.Z, fun.F), np.nan)
        se[:, fitted] = np.sqrt(draw.var[:, :cuts[1]])
        pub[:, fitted] = draw.Y[:, :cuts[1]]
        frame["se"], frame["published"] = se.ravel(), pub.ravel()
        frame["direct_cover90"] = np.where(frame["se"].notna(),
                                           np.abs(frame["published"] - frame["truth"])
                                           <= 1.645 * frame["se"], np.nan)
        frame["width90_over_se"] = frame["width90"] / (2 * frame["se"])
        frame["width90_over_truth"] = np.where(frame["truth"] > 0,
                                               frame["width90"] / frame["truth"], np.nan)
        frame.assign(**tag).to_csv(out / "posterior.csv.gz", index=False)
        info.update(draws=draws, seconds_posterior=time.perf_counter() - t1,
                    seconds_per_draw=sample.seconds_per_draw,
                    negative_variances=sample.negative_variances)
    save_npz(out / "map.npz", U=fit.U, shares=rep.shares, N_hat=rep.N_hat,
             N_true=rep.true_pop.N, Y=rep.draw.Y, var=rep.draw.var)
    info.update(seconds=time.perf_counter() - t0, peak_gb=peak_gb())
    if rep.draw.info is not None:
        info["calibration"] = rep.draw.info
    if rep.ind is not None:
        info["p0"] = {"N_hat_ind": rep.N_hat.tolist(),
                      "N_hat_survey": prior_shares(rep.draw.Y[:, :rep.cuts[1]], np.array(
                          [j for j, n in enumerate(rep.pA.cells) if n.startswith("unit_type:")]))[1]
                      .tolist(), "N_true": rep.true_pop.N.tolist(),
                      "sampled_units_equal_fraction": float(np.mean(
                          rep.ind.sampled_units == rep.draw.sampled_units))}
    write_text(out / "done.json", json.dumps(info, default=float))
    return info


def reuse_4b(c: float, sim: str, region_id: str, k: int, m: float, out: Path, rep, engine,
             kappa: float, tag: dict) -> np.ndarray | None:
    """Copy step 4b's fit when its unit has this ``c``, replicate and B1-chosen
    ``τ``; returns its ``Û`` (or ``None``)."""
    args = _SHARED["args"]
    if args.step4b is None:
        return None
    src = args.step4b / step4b_label(c) / sim / region_id / f"r{k}"
    if not (src / "done.json").exists():
        return None
    done = json.loads((src / "done.json").read_text())
    if float(done["chosen_tau_multiple"]) != float(m):
        return None
    out.mkdir(parents=True, exist_ok=True)
    for name in UNIT_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    U = np.load(src / "map.npz")["U"]
    tau = m * _SHARED["tau_ref_g"]
    ev = engine.evaluate(rep.theta0, U, gradient=False)
    fit = MapFit(tau=tau, U=U, objective=-ev.loglik + 0.5 * float(np.sum(U * U)) / tau ** 2,
                 objective_at_zero=np.nan, loglik=ev.loglik, grad_max=np.nan, iterations=0,
                 evaluations=0, converged=True, message="reused", seconds=0.0)
    laplace_evidence(engine, rep.theta0, fit)
    tau_rows = pd.read_csv(src / "tau.csv")
    row = tau_rows[tau_rows["tau_multiple"] == m].iloc[0]
    info = {**tag, "tau_multiple": m, "tau": tau, "kappa": kappa,
            "phi": dirmult_phi(rep.N_A, kappa), "map_start": row.get("start"),
            "map_objective": fit.objective, "map_grad_max": row.get("grad_max"),
            "spread": row.get("spread"), "log_evidence": fit.log_evidence,
            "b1_rms_z": row.get("b1_rms_z"), "b1_logdens": done["b1_logdens"],
            "draws": done.get("draws"), "seconds": done.get("seconds"),
            "seconds_per_draw": done.get("seconds_per_draw"),
            "negative_variances": done.get("negative_variances"), "reused_from": str(src)}
    write_text(out / "done.json", json.dumps(info, default=float))
    return U


def select_unit(unit: tuple) -> str:
    sim, region_id, c, k = unit
    args = _SHARED["args"]
    base = args.out / "select" / sim / region_id / f"c{c:g}" / f"r{k}"
    if all((base / f"t{m:g}" / "done.json").exists() for m in args.tau_multiples):
        return f"{sim} {region_id} c{c:g} r{k}: done"
    rep = Replicate(sim, region_id, k)
    engine, kappa = rep.engine(c)
    tag = {"simulation": sim, "region": region_id, "replicate": k, "c": c, "phase": "select"}
    previous = None
    for m in args.tau_multiples:
        out = base / f"t{m:g}"
        if (out / "done.json").exists():
            previous = np.load(out / "map.npz")["U"]
            continue
        U = reuse_4b(c, sim, region_id, k, m, out, rep, engine, kappa, tag)
        if U is None:
            info = fit_one(rep, engine, kappa, m, previous, out, tag, args.select_draws, True)
            U = np.load(out / "map.npz")["U"]
            logger.info("select %s %s c%g r%d tau %g: B1 log density %.1f, evidence %.1f, "
                        "%.0fs", sim, region_id, c, k, m, info["b1_logdens"],
                        info["log_evidence"], info["seconds"])
        else:
            logger.info("select %s %s c%g r%d tau %g: reused step 4b's fit", sim, region_id,
                        c, k, m)
        previous = U
    return f"{sim} {region_id} c{c:g} r{k}"


def final_unit(unit: tuple) -> str:
    sim, region_id, c, m_star, k = unit
    args = _SHARED["args"]
    out = (args.out / EXTRAS[sim] / region_id / f"r{k}") if sim in EXTRAS else \
        (args.out / "final" / sim / region_id / f"r{k}")
    if (out / "done.json").exists():
        return f"{sim} {region_id} r{k}: done"
    rep = Replicate(sim, region_id, k)
    engine, kappa = rep.engine(c)
    tag = {"simulation": sim, "region": region_id, "replicate": k, "c": c, "phase": "final"}
    if sim in EXTRAS:   # from scratch: the chosen τ alone, no warm start
        fit_one(rep, engine, kappa, m_star, None, out, tag, args.draws, True, warm=False)
        return f"{sim} {region_id} r{k}"
    if reuse_4b(c, sim, region_id, k, m_star, out, rep, engine, kappa, tag) is not None:
        return f"{sim} {region_id} r{k}: reused step 4b"
    previous = None
    for m in [x for x in args.tau_multiples if x < m_star]:     # the warm-start chain
        work = out / "chain" / f"t{m:g}"
        if (work / "done.json").exists():
            previous = np.load(work / "map.npz")["U"]
            continue
        fit_one(rep, engine, kappa, m, previous, work, tag, 0, False)
        previous = np.load(work / "map.npz")["U"]
    fit_one(rep, engine, kappa, m_star, previous, out, tag, args.draws, True)
    return f"{sim} {region_id} r{k}"


def run_units(fn, units: list, args) -> None:
    if args.workers > 1 and len(units) > 1:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for msg in pool.map(fn, units):
                logger.info("unit %s finished", msg)
    else:
        for unit in units:
            fn(unit)


def choose(args, sims, chosen) -> dict:
    """The joint choice per simulation and region (module notes)."""
    out = {}
    top = max(args.tau_multiples)
    for sim in sims:
        for rid in chosen:
            rows = []
            for c in args.c_grid:
                for m in args.tau_multiples:
                    logd, ev, lor = [], [], []
                    for k in range(min(args.select_replicates, args.replicates)):
                        d = args.out / "select" / sim / rid / f"c{c:g}" / f"r{k}" / f"t{m:g}"
                        info = json.loads((d / "done.json").read_text())
                        logd.append(info["b1_logdens"])
                        ev.append(info["log_evidence"])
                        sc = pd.read_csv(d / "scores.csv")
                        lor.append(sc.loc[sc["model"] == "latent_kern30", "lor_mae"].mean())
                    rows.append({"c": c, "tau_multiple": m, "b1_logdens": float(np.sum(logd)),
                                 "log_evidence": float(np.sum(ev)),
                                 "lor_mae": float(np.mean(lor))})
            f = pd.DataFrame(rows)
            best = f.loc[f["b1_logdens"].idxmax()]
            ev_best = f.loc[f["log_evidence"].idxmax()]
            truth_best = f.loc[f["lor_mae"].idxmin()]
            out[f"{sim}/{rid}"] = {
                "c": float(best["c"]), "tau_multiple": float(best["tau_multiple"]),
                "tau_at_top": bool(best["tau_multiple"] == top),
                "evidence_pair": [float(ev_best["c"]), float(ev_best["tau_multiple"])],
                "truth_best_pair": [float(truth_best["c"]), float(truth_best["tau_multiple"])],
                "pairs": rows}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="stage 1 run directory (z)")
    parser.add_argument("--step1", required=True, type=Path)
    parser.add_argument("--step3", required=True, type=Path)
    parser.add_argument("--step3b", required=True, type=Path)
    parser.add_argument("--step4", required=True, type=Path, help="step 4 output root (seed)")
    parser.add_argument("--step4b", type=Path, default=None,
                        help="step 4b output root, whose matching fits are reused")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--region", nargs="+", default=None)
    parser.add_argument("--simulations", nargs="+", default=["S", "G"], choices=["S", "G"])
    parser.add_argument("--scal", action="store_true",
                        help="also run simulation S-cal at S's chosen pairs (off by default)")
    parser.add_argument("--p0", action="store_true",
                        help="also run arm P0 (prior shares from an independent sample) at "
                             "S's chosen pairs (off by default)")
    parser.add_argument("--extras-only", action="store_true",
                        help="run only the requested --scal / --p0 arms (needs choice.json)")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--select-replicates", type=int, default=3)
    parser.add_argument("--select-draws", type=int, default=50)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(C_GRID))
    parser.add_argument("--tau-multiples", type=float, nargs="+", default=list(TAU_MULTIPLES))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--basis", default="kern30", choices=list(DEVIATION_BASES))
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
    s4 = json.loads((args.step4 / "settings.json").read_text())
    args.seed = int(s4["seed"]) if args.seed is None else args.seed
    record_run(args.out, args)

    config = load_config(args.config)
    regions = {r.id: r for r in load_regions(args.regions)}
    chosen = args.region or list(regions)
    step1 = load_step1(args.step1)
    tau_ref_g = float(json.loads((args.step3b / "tau_ref.json").read_text())[args.basis]["tau_ref"])
    data = load_check_data(config)
    atoms = load_atoms(data, args.run)
    if not pd.Index(atoms.serial.astype(str)).equals(pd.Index(step1.serial)):
        raise SystemExit("the atoms are not step 1's (SERIALNO order differs)")
    bases_dir = step1.root / "inputs" / "bases"
    H = load_basis(bases_dir, step1.h, atoms, args.run)
    G, basis_info = deviation_basis(args.basis, atoms, bases_dir, args.run)
    del atoms
    split = json.loads(SPLIT_FILE.read_text())
    sets = {"A": transfer_sets(data)["A"], "B1": tuple(split["B1"]), "B2": tuple(split["B2"])}
    X = {key: region_contributions(data, t) for key, t in sets.items()}
    pairs = PairOperator(X["A"], workers=args.threads)
    base = build_functionals(data, sets)
    ones = sp.csc_matrix(np.ones((base.G.shape[0], 1)))
    fun = Functionals(G=sp.hstack([base.G, ones]).tocsc(), cls=np.append(base.cls, "total"),
                      names=np.append(base.names, "units:total"))
    frame = pd.DataFrame({"SERIALNO": data.units.index.astype(str), "weight": data.weight,
                          "is_group_quarters": data.units["is_group_quarters"].to_numpy()})
    q = data.weight / data.weight.sum()
    refs = {}
    for rid in chosen:
        region = regions[rid]
        theta = np.load(args.step3 / rid / "reference.npz")["theta"]
        logb = log_base(q, region_distance(step1.distances, region.pumas, data.puma),
                        step1.gamma)
        refs[rid] = {"theta": theta, "logb": logb, "zone": zone_of_atoms(region, data.puma)}
    write_text(args.out / "settings.json", json.dumps({
        "mode": MODE, "basis": basis_info, "tau_ref_g": tau_ref_g,
        "tau_multiples": args.tau_multiples, "c_grid": args.c_grid,
        "select_replicates": args.select_replicates, "select_draws": args.select_draws,
        "draws": args.draws, "replicates": args.replicates, "simulations": args.simulations,
        "scal": bool(args.scal), "p0": bool(args.p0), "p0_tag": P0_TAG, "seed": args.seed,
        "step4b": None if args.step4b is None else str(args.step4b),
        "split": {key: list(v) for key, v in sets.items()}, "map_starts": "all4",
        "optimizer": "m10"}, indent=1, default=str))
    _SHARED.update(data=data, args=args, regions=regions, refs=refs, H=H, G=G, sets=sets, X=X,
                   pairs=pairs, fun=fun, frame=frame, tau_ref_g=tau_ref_g)

    def order(units, pos):
        return sorted(units, key=lambda u: (-len(regions[u[pos]].pumas), str(u)))

    n_sel = min(args.select_replicates, args.replicates)
    if not args.extras_only:
        sel = [(sim, r, c, k) for sim in args.simulations for r in chosen for c in args.c_grid
               for k in range(n_sel)]
        logger.info("selection: %d units (%d pairs each)", len(sel), len(args.tau_multiples))
        run_units(select_unit, order(sel, 1), args)
        choice_path = args.out / "choice.json"
        choice = json.loads(choice_path.read_text()) if choice_path.exists() else {}
        choice.update(choose(args, args.simulations, chosen))     # merged: jobs may split S/G
        write_text(choice_path, json.dumps(choice, indent=1))
        fin = [(sim, r, choice[f"{sim}/{r}"]["c"], choice[f"{sim}/{r}"]["tau_multiple"], k)
               for sim in args.simulations for r in chosen for k in range(n_sel, args.replicates)]
        logger.info("final fits: %d units", len(fin))
        run_units(final_unit, order(fin, 1), args)
    extras = [x for x, on in (("Scal", args.scal), ("P0", args.p0)) if on]
    if extras:
        choice = json.loads((args.out / "choice.json").read_text())
        units = [(x, r, choice[f"S/{r}"]["c"], choice[f"S/{r}"]["tau_multiple"], k)
                 for x in extras for r in chosen for k in range(args.replicates)]
        logger.info("%s: %d units at S's chosen pairs", " and ".join(extras), len(units))
        run_units(final_unit, order(units, 1), args)


if __name__ == "__main__":
    main()
