"""Step 4d of the latent-population PMEDM: the free-shares arm FS
(``PLAN_latent_step4d.md`` version 1; model ``MODEL_latent_v5_draft.md``).

    python experiments/latent/step4d_free_shares.py --config experiments/embed/stage1_v2.json \
        --regions experiments/latent/regions_tn.json \
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d16_g0.5_r0.01 \
        --step1 $PMEDM_VB_RUNS/latent/step1_sigma --step3 $PMEDM_VB_RUNS/latent/step3 \
        --step3b $PMEDM_VB_RUNS/latent/step3b --step4 $PMEDM_VB_RUNS/latent/step4 \
        --step4c $PMEDM_VB_RUNS/latent/step4c --out $PMEDM_VB_RUNS/latent/step4d \
        --workers 16 --threads 1
    python experiments/latent/collect_step4d.py --root $PMEDM_VB_RUNS/latent/step4d \
        --step4c $PMEDM_VB_RUNS/latent/step4c

**Design** (§2): simulation S's data exactly as step 4c built them (its
:class:`Replicate`, the same seeds, realized truths, ``Y`` and ``Σ̂``), at the
``(τ, c)`` pair step 4c chose for S in each region (``--step4c``'s
``choice.json``), all ``--replicates`` (10) replicates, ``--draws`` (200)
posterior draws, fitted from scratch. Only the realized level changes: model
version 5's free shares (:mod:`pmedm_vb.latent.free_shares`), with a
``Dirichlet(--share-prior)`` (½) prior; the survey's shares are a starting
value only.

**Per unit** (``<out>/FS/<region>/r<k>/``):

* the MAP of ``(U, ψ)``: two starts at ``U = 0`` with the Gauss–Newton-
  preconditioned L-BFGS, from the survey's shares and from equal shares, the
  lower objective;
  its Laplace evidence (reported only);
* the realized MAP with the shares free (SQUAREM-accelerated alternation of
  the conditional PMEDM) and its scores, as step 4c;
* the posterior (Laplace draws of ``(U, ψ)``, Gaussian conditioning per zone
  given the shares): interval statistics of every functional including the
  zone totals, and the B1 predictive log density;
* ``map.npz`` with the MAP shares, the survey's shares, the realized MAP's
  shares and the true totals; ``done.json``.

Finished units are skipped, so a timed-out job can be resubmitted unchanged.
Runs offline: ``PMEDM_VB_OFFLINE=1`` is set if the caller has not set it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import step4c_joint as s4c
from pmedm_vb.data.cache import OFFLINE_ENV
from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.puma_check import load_check_data
from pmedm_vb.latent.area import load_basis, load_step1
from pmedm_vb.latent.atoms import load_atoms
from pmedm_vb.latent.calibration import interval_stats
from pmedm_vb.latent.collapsed import PairOperator
from pmedm_vb.latent.deviation_basis import DEVIATION_BASES, deviation_basis
from pmedm_vb.latent.free_shares import (
    SHARE_PRIOR,
    free_share_engine,
    free_share_evidence,
    free_share_posterior_samples,
    free_share_starts,
    solve_free_share_map,
)
from pmedm_vb.latent.overdispersed import dirmult_phi, predictive_log_density
from pmedm_vb.latent.posterior import Functionals, build_functionals, zone_truth
from pmedm_vb.latent.reference import log_base
from pmedm_vb.latent.region_engine import region_contributions
from pmedm_vb.latent.region_reference import region_distance, zone_log_bases, zone_of_atoms
from pmedm_vb.latent.regions import load_regions, transfer_sets
from pmedm_vb.latent.scoring import score_region, transfer_misfit
from pmedm_vb.progress import logger, record_run, save_npz, write_text

ARM = "FS"
#: Tag of the posterior draws' seed, apart from step 4c's streams.
FS_TAG = 4004
_SHARED = s4c._SHARED       # step 4c's Replicate reads this dictionary


def fit_unit(unit: tuple) -> str:
    region_id, c, m, k = unit
    s, args = _SHARED, _SHARED["args"]
    out = args.out / ARM / region_id / f"r{k}"
    if (out / "done.json").exists():
        return f"{region_id} r{k}: done"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rep = s4c.Replicate("S", region_id, k)           # S's data, seeds and Σ̂
    kappa = c * rep.n_A
    tau = m * s["tau_ref_g"]
    engine = free_share_engine(rep.area, s["H"], rep.ref["theta"], s["G"],
                               zone_log_bases(rep.ref["logb"], rep.ref["zone"], rep.Z, s4c.MODE),
                               s["pairs"], rep.N_A, kappa, args.share_prior)
    fit, starts = free_share_starts(engine, rep.theta0, tau, rep.shares, gtol=args.map_gtol,
                                    max_iter=args.map_max_iter)
    free_share_evidence(engine, rep.theta0, fit)
    seconds_map = time.perf_counter() - t0
    phi = dirmult_phi(rep.N_A, kappa)
    p = engine.probabilities(rep.theta0, fit.U)                           # (M, Z)
    realized = solve_free_share_map(s["X"]["A"], p, fit.shares, rep.N_A, rep.N_A / phi,
                                    rep.pA.inputs.targets(), rep.sigma_dense,
                                    tol=args.cond_tol, max_iter=args.cond_max_iter,
                                    unit_tol=args.unit_tol)
    res = realized.result
    q = fit.shares[:, None] * p.T
    tag = {"simulation": ARM, "region": region_id, "replicate": k, "c": c, "phase": "final",
           "tau_multiple": m, "tau": tau, "kappa": kappa, "phi": phi,
           "share_prior": args.share_prior}
    b1 = float(transfer_misfit(rep.views["B1"], res.r, "x", "A", "B1")["rms_z"].mean())
    se_cells = np.sqrt(rep.pA.inputs.sigma_v)
    parts = {key: [] for key in ("scores", "rare", "misfit", "transfer")}
    for name, pm in (("latent_kern30", res.r), ("latent_kern30:p", q)):
        scored = score_region(s["data"], rep.views["A"], pm, name, se_cells)
        scored["transfer"] = pd.concat([transfer_misfit(rep.views[key], pm, name, "A", key)
                                        for key in s4c.SCORED], ignore_index=True)
        for key in parts:
            parts[key].append(scored[key].assign(**tag))
    for key, part in parts.items():
        write_text(out / f"{key}.csv", pd.concat(part, ignore_index=True).to_csv(index=False))
    values = [f.objective for f in starts]
    write_text(out / "starts.csv", pd.DataFrame(
        [{**tag, **f.info(), "selected": f is fit} for f in starts]).to_csv(index=False))
    write_text(out / "fits.csv", pd.DataFrame(
        [{**tag, "model": "free_share_pairs", **realized.info()}]).to_csv(index=False))

    t1 = time.perf_counter()
    fun = s["fun"]
    post_rng = np.random.default_rng([args.seed, s4c.SIM_INDEX["S"], rep.index, k, 2,
                                      int(round(m * 1000)), int(round(kappa)), FS_TAG])
    sample = free_share_posterior_samples(engine, rep.theta0, fit, fun, s["X"]["A"],
                                          args.draws, post_rng)
    counts = np.zeros(s["data"].weight.size)
    counts[rep.joint.units] = rep.true_pop.counts
    truth = zone_truth(fun, rep.ref["zone"], counts, rep.Z)
    st = interval_stats(sample.samples, truth)
    cuts, draw = rep.cuts, rep.draw
    b1_logdens = predictive_log_density(sample.samples[:, :, fun.of("B1")],
                                        draw.Y[:, cuts[1]:cuts[2]], draw.var[:, cuts[1]:cuts[2]])
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

    N_true = rep.true_pop.N.astype(float)
    N_map, N_survey = rep.N_A * fit.shares, rep.N_A * rep.shares
    N_realized = rep.N_A * realized.shares
    sd_shares = sample.shares.std(axis=0, ddof=1) * rep.N_A
    save_npz(out / "map.npz", U=fit.U, psi=fit.psi, shares=fit.shares,
             shares_survey=rep.shares, shares_realized=realized.shares, N_hat=rep.N_hat,
             N_true=N_true, N_posterior_sd=sd_shares, Y=rep.draw.Y, var=rep.draw.var)

    def rel_rms(est):
        return float(np.sqrt(np.mean(((est - N_true) / N_true) ** 2)))

    info = {**tag, "map_start": fit.start, "map_objective": fit.objective,
            "map_grad_max": fit.grad_max, "map_grad_max_U": fit.grad_max_U,
            "map_grad_max_psi": fit.grad_max_psi, "map_converged": fit.converged,
            "spread": max(values) - min(values),
            "evaluations_all_starts": sum(f.evaluations for f in starts),
            "log_evidence": fit.log_evidence, "b1_rms_z": b1, "b1_logdens": b1_logdens,
            "seconds_map": seconds_map, "realized_outer_iterations": realized.outer_iterations,
            "realized_converged": realized.converged,
            "realized_share_change_units": realized.share_change * rep.N_A,
            "conditional_converged": bool(res.converged),
            "totals_rel_rms": {"map": rel_rms(N_map), "survey": rel_rms(N_survey),
                               "realized_map": rel_rms(N_realized)},
            "totals_posterior_sd_rel_mean": float(np.mean(sd_shares / N_true)),
            "draws": args.draws, "seconds_posterior": time.perf_counter() - t1,
            "seconds_per_draw": sample.seconds_per_draw,
            "negative_variances": sample.negative_variances,
            "seconds": time.perf_counter() - t0, "peak_gb": s4c.peak_gb()}
    write_text(out / "done.json", json.dumps(info, default=float))
    logger.info("FS %s r%d: B1 log density %.1f, totals rel. RMS map %.4f survey %.4f, %.0fs",
                region_id, k, b1_logdens, info["totals_rel_rms"]["map"],
                info["totals_rel_rms"]["survey"], info["seconds"])
    return f"{region_id} r{k}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="stage 1 run directory (z)")
    parser.add_argument("--step1", required=True, type=Path)
    parser.add_argument("--step3", required=True, type=Path)
    parser.add_argument("--step3b", required=True, type=Path)
    parser.add_argument("--step4", required=True, type=Path, help="step 4 output root (seed)")
    parser.add_argument("--step4c", required=True, type=Path,
                        help="step 4c output root (its choice.json: S's pair per region)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--region", nargs="+", default=None)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--share-prior", type=float, default=SHARE_PRIOR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--basis", default="kern30", choices=list(DEVIATION_BASES))
    parser.add_argument("--sigma-alpha", type=float, default=0.01)
    parser.add_argument("--prior-sd", type=float, default=3.0)
    parser.add_argument("--map-gtol", type=float, default=1e-4)
    parser.add_argument("--map-max-iter", type=int, default=20000)
    parser.add_argument("--cond-tol", type=float, default=1e-10)
    parser.add_argument("--cond-max-iter", type=int, default=200)
    parser.add_argument("--unit-tol", type=float, default=0.01,
                        help="realized MAP: stop when no zone total moves by more (units)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    os.environ.setdefault(OFFLINE_ENV, "1")
    args.out.mkdir(parents=True, exist_ok=True)
    s4 = json.loads((args.step4 / "settings.json").read_text())
    args.seed = int(s4["seed"]) if args.seed is None else args.seed
    record_run(args.out, args)
    choice = json.loads((args.step4c / "choice.json").read_text())

    # The setup of step4c_joint.main, unchanged, so that Replicate builds S's data.
    config = load_config(args.config)
    regions = {r.id: r for r in load_regions(args.regions)}
    chosen = args.region or list(regions)
    missing = [r for r in chosen if f"S/{r}" not in choice]
    if missing:
        raise SystemExit(f"step 4c chose no pair for S in {missing}")
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
    split = json.loads(s4c.SPLIT_FILE.read_text())
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
        "arm": ARM, "mode": s4c.MODE, "basis": basis_info, "tau_ref_g": tau_ref_g,
        "draws": args.draws, "replicates": args.replicates, "share_prior": args.share_prior,
        "seed": args.seed, "fs_tag": FS_TAG, "step4c": str(args.step4c),
        "pairs": {r: [choice[f"S/{r}"]["c"], choice[f"S/{r}"]["tau_multiple"]] for r in chosen},
        "split": {key: list(v) for key, v in sets.items()}, "unit_tol": args.unit_tol,
        "map_starts": ["survey_gn", "equal_gn"]}, indent=1, default=str))
    _SHARED.update(data=data, args=args, regions=regions, refs=refs, H=H, G=G, sets=sets, X=X,
                   pairs=pairs, fun=fun, frame=frame, tau_ref_g=tau_ref_g)

    units = [(r, choice[f"S/{r}"]["c"], choice[f"S/{r}"]["tau_multiple"], k)
             for r in chosen for k in range(args.replicates)]
    # Largest regions first, but every region's replicates interleaved, so a
    # partial run covers all regions rather than only the largest.
    units.sort(key=lambda u: (u[3], -len(regions[u[0]].pumas), u[0]))
    logger.info("FS: %d units at S's chosen pairs", len(units))
    s4c.run_units(fit_unit, units, args)


if __name__ == "__main__":
    main()
