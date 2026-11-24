"""Step 4e of the latent-population PMEDM: the cut arm FScut
(``PLAN_latent_step4e.md`` version 1; model ``MODEL_latent_v5_draft.md`` §7).

    python experiments/latent/step4e_cut.py --config experiments/embed/stage1_v2.json \
        --regions experiments/latent/regions_tn.json \
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d16_g0.5_r0.01 \
        --step1 $PMEDM_VB_RUNS/latent/step1_sigma --step3 $PMEDM_VB_RUNS/latent/step3 \
        --step3b $PMEDM_VB_RUNS/latent/step3b --step4 $PMEDM_VB_RUNS/latent/step4 \
        --step4c $PMEDM_VB_RUNS/latent/step4c --out $PMEDM_VB_RUNS/latent/step4e \
        --workers 16 --threads 1
    python experiments/latent/collect_step4e.py --root $PMEDM_VB_RUNS/latent/step4e \
        --step4d $PMEDM_VB_RUNS/latent/step4d --step4c $PMEDM_VB_RUNS/latent/step4c

**Design** (§2): as step 4d's arm FS (S's data built by step 4c's
``Replicate``, step 4c's pair for S in each region, all replicates, 200
draws, from scratch), with the cut form of model version 5:

* **totals module:** each zone's direct unit total ``x_b`` (the sum of its
  fitted ``unit_type:*`` cells) and its variance from S's ``Σ̂`` block (the
  cells' block summed, plus their zero-cell floors; zones independent), then
  Gaussian conditioning on ``Σ_b N_b = N_A``
  (:func:`~pmedm_vb.latent.free_shares.cut_totals`);
* **U given the totals:** the MAP of ``U`` with the shares fixed at the
  module's mean (two starts at ``U = 0``, ``gn`` and ``m10``; the lower),
  and its Laplace approximation;
* **realized MAP:** the conditional PMEDM with the zone totals fixed at the
  module's mean (:func:`~pmedm_vb.latent.free_shares.solve_conditional_fixed_shares`);
* **posterior:** for each draw, totals from the module, ``U`` from its
  Laplace approximation, each zone's functionals conditioned on ``Y`` given
  its total (:func:`~pmedm_vb.latent.free_shares.cut_posterior_samples`).

Writes step 4d's unit files under ``<out>/FScut/<region>/r<k>/``; finished
units are skipped. Runs offline (``PMEDM_VB_OFFLINE=1`` unless set).
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import step4c_joint as s4c
import step4d_free_shares as s4d
from pmedm_vb.latent.free_shares import (
    cut_posterior_samples,
    cut_totals,
    fit_cut_U,
    free_share_engine,
    shares_to_psi,
    solve_conditional_fixed_shares,
)
from pmedm_vb.latent.overdispersed import dirmult_phi
from pmedm_vb.latent.region_reference import zone_log_bases
from pmedm_vb.progress import logger, save_npz, write_text

ARM = "FScut"
#: Tag of the posterior draws' seed, apart from steps 4c and 4d.
CUT_TAG = 4005
_SHARED = s4c._SHARED


def direct_unit_totals(rep) -> tuple[np.ndarray, np.ndarray]:
    """``x`` (each zone's direct unit total) and its covariance (diagonal:
    simulation S samples each zone on its own)."""
    cells = np.array([j for j, name in enumerate(rep.pA.cells) if name.startswith("unit_type:")])
    if cells.size == 0:
        raise ValueError("no unit_type cells among the fitted cells")
    draw = rep.draw
    if draw.blocks is None:
        raise ValueError("the totals module expects simulation S's per-zone covariance blocks")
    x = draw.Y[:, cells].sum(axis=1)
    v = (np.array([B[np.ix_(cells, cells)].sum() for B in draw.blocks])
         + draw.floor[:, cells].sum(axis=1))
    return x, np.diag(v)


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
    x, Sx = direct_unit_totals(rep)
    mean, cov = cut_totals(x, Sx, rep.N_A)
    if not np.all(mean > 0):
        raise ValueError(f"{region_id} r{k}: the totals module gave a non-positive total")
    shares = mean / rep.N_A
    psi = shares_to_psi(shares)
    engine = free_share_engine(rep.area, s["H"], rep.ref["theta"], s["G"],
                               zone_log_bases(rep.ref["logb"], rep.ref["zone"], rep.Z, s4c.MODE),
                               s["pairs"], rep.N_A, kappa)
    fit, starts = fit_cut_U(engine, rep.theta0, tau, psi, gtol=args.map_gtol,
                            max_iter=args.map_max_iter)
    seconds_map = time.perf_counter() - t0
    phi = dirmult_phi(rep.N_A, kappa)
    p = engine.probabilities(rep.theta0, fit.U)                           # (M, Z)
    res = solve_conditional_fixed_shares(s["X"]["A"], p, shares, rep.N_A, rep.N_A / phi,
                                         rep.pA.inputs.targets(), rep.sigma_dense,
                                         tol=args.cond_tol, max_iter=args.cond_max_iter)
    q = shares[:, None] * p.T
    tag = {"simulation": ARM, "region": region_id, "replicate": k, "c": c, "phase": "final",
           "tau_multiple": m, "tau": tau, "kappa": kappa, "phi": phi}
    b1 = s4d.write_scores(rep, res.r, q, tag, out)
    values = [f.objective for f in starts]
    write_text(out / "starts.csv", pd.DataFrame(
        [{**tag, **f.info(), "selected": f is fit} for f in starts]).to_csv(index=False))
    write_text(out / "fits.csv", pd.DataFrame(
        [{**tag, "model": "cut_fixed_shares", **res.info()}]).to_csv(index=False))

    t1 = time.perf_counter()
    post_rng = np.random.default_rng([args.seed, s4c.SIM_INDEX["S"], rep.index, k, 2,
                                      int(round(m * 1000)), int(round(kappa)), CUT_TAG])
    sample = cut_posterior_samples(engine, rep.theta0, fit, mean, cov, s["fun"], s["X"]["A"],
                                   args.draws, post_rng)
    b1_logdens = s4d.write_posterior(rep, sample.samples, tag, out)

    N_true = rep.true_pop.N.astype(float)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    save_npz(out / "map.npz", U=fit.U, shares=shares, totals_mean=mean, totals_cov=cov,
             x_direct=x, x_var=np.diag(Sx), shares_survey=rep.shares,
             shares_realized=res.shares, N_hat=rep.N_hat, N_true=N_true, Y=rep.draw.Y,
             var=rep.draw.var)

    def rel_rms(est):
        return float(np.sqrt(np.mean(((est - N_true) / N_true) ** 2)))

    info = {**tag, "map_start": fit.start, "map_objective": fit.objective,
            "map_grad_max": fit.grad_max, "map_grad_max_U": fit.grad_max_U,
            "map_grad_max_psi": 0.0, "map_converged": fit.converged,
            "spread": max(values) - min(values),
            "evaluations_all_starts": sum(f.evaluations for f in starts),
            "b1_rms_z": b1, "b1_logdens": b1_logdens, "seconds_map": seconds_map,
            "realized_outer_iterations": 1, "realized_converged": bool(res.converged),
            "realized_share_change_units": float(np.abs(res.shares - shares).max() * rep.N_A),
            "conditional_converged": bool(res.converged), "conditional_gap": res.gap,
            "totals_rel_rms": {"map": rel_rms(mean), "survey": rel_rms(rep.N_A * rep.shares),
                               "realized_map": rel_rms(rep.N_A * res.shares),
                               "direct": rel_rms(x)},
            "totals_posterior_sd_rel_mean": float(np.mean(sd / N_true)),
            "draws": args.draws, "seconds_posterior": time.perf_counter() - t1,
            "seconds_per_draw": sample.seconds_per_draw,
            "negative_variances": sample.negative_variances,
            "seconds": time.perf_counter() - t0, "peak_gb": s4c.peak_gb()}
    write_text(out / "done.json", json.dumps(info, default=float))
    logger.info("FScut %s r%d: B1 log density %.1f, totals rel. RMS module %.4f, %.0fs",
                region_id, k, b1_logdens, info["totals_rel_rms"]["map"], info["seconds"])
    return f"{region_id} r{k}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    s4d.common_arguments(parser)
    args = parser.parse_args()
    regions, chosen, choice = s4d.setup(args, {
        "arm": ARM, "cut_tag": CUT_TAG, "map_starts": ["zero_gn", "zero_m10"],
        "totals_module": "unit_type cells + N_A, Gaussian conditioning, zones independent"})
    units = s4d.units_at_s_pairs(regions, chosen, choice, args.replicates)
    logger.info("FScut: %d units at S's chosen pairs", len(units))
    s4c.run_units(fit_unit, units, args)


if __name__ == "__main__":
    main()
