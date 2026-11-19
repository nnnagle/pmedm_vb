"""Step 3 of the latent-population PMEDM: MAP on the regional run, against the
standard PMEDM (``PLAN_latent_step3.md`` version 2; model ``MODEL_latent.qmd``
version 2).

    python experiments/latent/step3_regional.py --config experiments/embed/stage1_v2.json \
        --regions experiments/latent/regions_tn.json \
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d16_g0.5_r0.01 \
        --step1 $PMEDM_VB_RUNS/latent/step1_sigma --out $PMEDM_VB_RUNS/latent/step3 \
        --map-starts <fixedS|all4|plain2> --workers 16 --threads 1
    python experiments/latent/collect_step3.py --root $PMEDM_VB_RUNS/latent/step3

The region plays the PUMA's part and its PUMAs (zones) the block groups'
(§2-§3). From step 1's run root (``--step1``): h, γ and σ_θ of its final fits
(``final/settings.json``), the basis (``inputs/bases/<h>.*``) and the PUMA
distances; the atoms are rebuilt from ``--run`` and must be step 1's, in
order. The fitted tables are each zone's S1 cells from its own records
(:func:`~pmedm_vb.latent.regions.build_region`); ``Σ`` is
:func:`~pmedm_vb.latent.region_engine.region_sigma` (``α = 0.01``, untapered,
``build_region``'s one-case floor, plus ``ω² I``), the same for every
variant.

**Per region** (before the work units, in the main process; §4.1):
``θ̂_A`` on all the region's records
(:func:`~pmedm_vb.latent.region_reference.fit_region_reference`) and the
reproduction check, written to ``<out>/<region>/reference.{json,npz}`` and
``reference_check.csv``.

**Per region and support mode** (a work unit; ``--modes``, default ``lzo``
then ``own``; §3), the ladder of §4.5:

``pmedm_own``   the standard PMEDM on the region's own records with their
                design weights (:func:`~pmedm_vb.solvers.map_dual.solve_map`,
                ``α = 0.01``, untapered, ``hierarchy="none+prior3"``);
``pmedm_ref``   the same on every statewide atom with base ``q^A``;
``latent_u0``   the conditional PMEDM (:mod:`pmedm_vb.latent.conditional`),
                base ``p_b = q^A`` (``U = 0``), weight ``N_b``;
``latent_t<m>`` the conditional PMEDM at the MAP ``Û`` for ``τ = m τ_ref``
                (:mod:`pmedm_vb.latent.map_fit`; ``--tau-multiples``,
                ``τ_ref = 0.284`` from step 2): the lowest exact objective
                over the set of starts ``--map-starts`` (§4.3b: ``fixedS``,
                ``all4`` or ``plain2``, chosen from the diagnostic,
                ``step3_diagnostic.py``), the ``τ`` in increasing order so
                that a start can begin from the previous ``τ``'s MAP;
``latent``      the same at the ``τ`` chosen by Laplace evidence (a copy of
                that ``latent_t<m>``'s rows);
``latent_n``    as ``latent``, entropy weight ``n_A`` (``κ_b = n_A N_b / N_A``).

In ``lzo`` mode each zone's own records have zero base weight in every
variant (:mod:`pmedm_vb.latent.region_reference`). Every variant is scored
on its realized MAP (zone ``b``'s mass ``N_b r̂_b``; the standard PMEDM's
``W``), and ``latent_u0`` and every ``latent*`` also on ``p_b`` itself (model
names with ``:p``), with the regional run's scoring
(:func:`~pmedm_vb.latent.scoring.score_region`, and transfer to table set B,
:func:`~pmedm_vb.latent.scoring.transfer_misfit`). Allocation accuracy is
reported in ``own`` mode only (in ``lzo`` mode it is undefined, written NaN);
for a variant on the statewide atoms it is computed over the region's own
records (their share of mass in their true zone).

Each unit writes ``<out>/<region>/<mode>/``: ``scores``, ``rare``,
``misfit``, ``spread``, ``summary``, ``transfer`` (as the regional run),
``fits.csv`` (per variant: convergence, iterations, seconds, and for the
conditional PMEDM the duality gap and KKT residual), ``tau.csv`` (per ``τ``:
the MAP's start, objective against ``U = 0``, largest gradient component,
iterations, evaluations, seconds per evaluation, Laplace evidence, chosen,
and the spread of the starts' final exact objectives with their total
evaluations), ``starts.csv`` (every start's fit),
``map.npz`` (``θ̂_A``, ``Û`` per ``τ``, evidence), ``realized.npz`` (``r̂_b``
per variant in float32 and the multipliers; the standard PMEDM's ``W / N``),
``timings.json`` and ``done.json``; a unit with ``done.json`` is skipped, so a
timed-out job can be resubmitted unchanged.

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

from pmedm_vb.data.cache import OFFLINE_ENV
from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.puma_check import load_check_data
from pmedm_vb.latent.area import load_basis, load_step1
from pmedm_vb.latent.atoms import load_atoms
from pmedm_vb.latent.collapsed import PairOperator
from pmedm_vb.latent.conditional import solve_conditional
from pmedm_vb.latent.map_fit import START_SETS, choose_tau, fit_starts, laplace_evidence
from pmedm_vb.latent.region_engine import (
    RegionLikelihood,
    region_area,
    region_contributions,
    region_sigma,
)
from pmedm_vb.latent.region_reference import (
    MODES,
    fit_region_reference,
    pair_base,
    pmedm_inputs,
    reproduction_check,
    zone_distributions,
    zone_log_bases,
)
from pmedm_vb.latent.regions import build_region, load_regions, transfer_sets
from pmedm_vb.latent.scoring import allocation_accuracy, score_region, transfer_misfit
from pmedm_vb.progress import logger, record_run, save_npz, stage, write_text
from pmedm_vb.solvers.map_dual import solve_map

#: Filled before the worker pool forks, so workers share it without pickling.
_SHARED: dict = {}

#: τ_ref from step 2 (``STEP2_latent.qmd``).
TAU_REF = 0.284
TAU_MULTIPLES = (0.5, 1.0, 2.0)

FRAMES = ("scores", "rare", "misfit", "spread", "summary", "transfer")


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def tau_label(m: float) -> str:
    return f"latent_t{m:g}"


def statewide_view(problem, X_all, frame, zone_all):
    """``problem`` with its units replaced by every statewide atom (for scoring a
    distribution over the atoms): ``score_region`` and ``transfer_misfit`` read
    ``inputs.N``, ``X_B``, ``Y_B``, ``sigma_v``, ``units`` and the view's
    ``units`` / ``zone_of_unit`` (``-1`` outside the region)."""
    from dataclasses import replace

    inputs = replace(problem.inputs, X_B=X_all, units=frame)
    return replace(problem, inputs=inputs, units=np.arange(X_all.shape[0]),
                   zone_of_unit=zone_all)


def run_references(args) -> None:
    """θ̂_A and the reproduction check per region (§4.1), into ``_SHARED``."""
    s = _SHARED
    data, H, step1 = s["data"], s["H"], s["step1"]
    q = data.weight / data.weight.sum()
    refs = {}
    for region_id in s["chosen"]:
        region = s["regions"][region_id]
        out = args.out / region_id
        out.mkdir(parents=True, exist_ok=True)
        ref = fit_region_reference(region, H, q, data.weight, data.puma, step1.distances,
                                   step1.gamma, step1.sigma)
        check = reproduction_check(data, ref)
        info = ref.info()
        info["max_moment_error"] = float(np.abs(
            ref.q @ H - (data.weight[ref.in_region] / ref.N) @ H[ref.in_region]).max())
        write_text(out / "reference.json", json.dumps(info, indent=1, default=str))
        write_text(out / "reference_check.csv", check.to_csv(index=False))
        save_npz(out / "reference.npz", theta=ref.theta, q=ref.q.astype(np.float32),
                 zones=np.array(region.pumas))
        flagged = check[check["stop_group"] & (check["in_stop_range"] == 0)]
        logger.info("region %s: theta_A in %d Newton steps (gradient %.1e), ESS %.0f, "
                    "outside %.2f; AIAN/NHPI ratios %s%s", region_id, ref.fit.iterations,
                    ref.fit.grad_max, ref.fit.ess, ref.fit.share_outside,
                    ", ".join(f"{r:.3f}" for r in check.loc[check["stop_group"], "ratio"]),
                    " (outside [0.8, 1.25])" if len(flagged) else "")
        if not ref.fit.converged:
            logger.warning("region %s: theta_A not converged (%s)", region_id, ref.fit.message)
        refs[region_id] = ref
    s["refs"] = refs


def run_unit(unit: tuple[str, str]) -> tuple[str, str, float]:
    region_id, mode = unit
    s = _SHARED
    data, args, H = s["data"], s["args"], s["H"]
    out = args.out / region_id / mode
    if (out / "done.json").exists():
        logger.info("region %s %s: done, skipped", region_id, mode)
        return region_id, mode, 0.0
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    times: dict[str, float] = {}
    region, ref = s["regions"][region_id], s["refs"][region_id]
    sets, T_all, B_all = s["sets"], s["T_all"], s["B_all"]
    problems = {k: build_region(data, region, tables=sets[k]) for k in ("A", "B")}
    problem = problems["A"]
    inputs = problem.inputs
    Z = inputs.n_zones
    N_b = np.asarray(problem.zone_weight, float)
    N_A, n_A = float(inputs.N), int(inputs.n)
    share = N_b / N_A
    sigma = region_sigma(problem, args.sigma_alpha, args.prior_sd)
    area = region_area(problem, sigma)
    zone_all = ref.zone
    views = {"A": statewide_view(problem, T_all, s["frame"], zone_all),
             "B": statewide_view(problems["B"], B_all, s["frame"], zone_all)}
    se_cells = np.sqrt(inputs.sigma_v)
    frames = {k: [] for k in FRAMES}
    fits, realized = [], {}
    tag = {"region": region_id, "mode": mode}

    def score(model: str, p: np.ndarray, statewide: bool) -> None:
        pr, other = (views["A"], views["B"]) if statewide else (problem, problems["B"])
        scored = score_region(data, pr, p, model, se_cells)
        scored["transfer"] = transfer_misfit(other, p, model, "A", "B")
        acc = np.nan
        if mode == "own":
            acc = allocation_accuracy(problem, p[:, problem.units] if statewide else p)
        scored["summary"]["allocation_accuracy"] = acc
        for k, v in scored.items():
            frames[k].append(v.assign(mode=mode, fit_set="A"))

    def conditional(model: str, base: np.ndarray, kappa=None):
        t0 = time.perf_counter()
        res = solve_conditional(T_all, base, N_b, inputs.targets(), sigma, kappa=kappa,
                                tol=args.cond_tol, max_iter=args.cond_max_iter)
        fits.append({**tag, "model": model, "kind": "conditional", **res.info()})
        if not res.converged:
            logger.warning("region %s %s %s: conditional PMEDM not converged (decrement %.2e)",
                           region_id, mode, model, res.decrement)
        realized[model] = res.r.astype(np.float32)
        realized[f"{model}__lambda"] = res.lam
        times[model] = time.perf_counter() - t0
        return res

    # Standard PMEDM on the region's own records, then on the statewide atoms (base q^A).
    w = inputs.units["weight"].to_numpy()
    for model, q_pairs, X in (
            ("pmedm_own", pair_base(w, problem.zone_of_unit, Z, mode), None),
            ("pmedm_ref", pair_base(ref.q, zone_all, Z, mode), T_all)):
        t0 = time.perf_counter()
        with stage(f"region {region_id} {mode}: {model}"):
            pin = pmedm_inputs(problem, q_pairs, X=X, units=None if X is None else s["frame"])
            res = solve_map(pin, alpha=args.sigma_alpha, taper=None,
                            hierarchy=f"none+prior{args.prior_sd:g}", max_iter=200)
        times[model] = time.perf_counter() - t0
        p = res.W / pin.N
        fits.append({**tag, "model": model, "kind": "pmedm", "converged": bool(res.converged),
                     "iterations": int(res.n_iter), "objective": float(res.objective),
                     "decrement": float(res.newton_decrement), "seconds": times[model],
                     "zone_shares_max_abs_dev": float(np.abs(p.sum(axis=1) - share).max())})
        realized[model] = p.astype(np.float32)
        score(model, p, statewide=X is not None)

    # The latent ladder.
    logbz = zone_log_bases(ref.logb, zone_all, Z, mode)
    engine = RegionLikelihood(area, H, logbz, s["pairs"])
    p0 = zone_distributions(ref.q, zone_all, Z, mode)
    res = conditional("latent_u0", p0)
    score("latent_u0", share[:, None] * res.r, True)
    score("latent_u0:p", share[:, None] * p0, True)

    map_fits, bases, start_rows, spreads, previous = [], {}, [], [], None
    gtol_kw = {"gtol": args.map_gtol, "max_iter": args.map_max_iter}
    for m in args.tau_multiples:
        tau = m * args.tau_ref
        t0 = time.perf_counter()
        with stage(f"region {region_id} {mode}: MAP at tau = {m:g} tau_ref, "
                   f"starts {args.map_starts}") as step:
            fit, starts = fit_starts(engine, ref.theta, tau, args.map_starts, previous, **gtol_kw)
            laplace_evidence(engine, ref.theta, fit)
            values = [f.objective for f in starts]
            spreads.append({"n_starts": len(starts), "spread": max(values) - min(values),
                            "evaluations_all_starts": sum(f.evaluations for f in starts),
                            "seconds_all_starts": sum(f.seconds for f in starts)})
            step.detail = (f"MAP from {fit.start}; F by start "
                           + ", ".join(f"{f.start} {f.objective:.2f}" for f in starts)
                           + f"; spread {spreads[-1]['spread']:.2f}; F(0) {fit.objective_at_zero:.2f}"
                           f"; gradient {fit.grad_max:.1e}; log evidence {fit.log_evidence:.2f}")
        start_rows += [{**tag, "tau_multiple": m, **f.info(), "selected": f is fit}
                       for f in starts]
        times[f"map_t{m:g}"] = time.perf_counter() - t0
        if not fit.converged:
            logger.warning("region %s %s tau %g: MAP not converged (%s, gradient %.2e)",
                           region_id, mode, m, fit.message, fit.grad_max)
        map_fits.append(fit)
        previous = fit.U
        p_b = engine.probabilities(ref.theta, fit.U).T
        bases[m] = p_b
        model = tau_label(m)
        res = conditional(model, p_b)
        score(model, share[:, None] * res.r, True)
        score(f"{model}:p", share[:, None] * p_b, True)
    chosen = choose_tau(map_fits)
    m_star = args.tau_multiples[chosen]
    tau_rows = [{**tag, "tau_multiple": m, "map_starts": args.map_starts, **f.info(), **sp_,
                 "chosen": k == chosen}
                for k, (m, f, sp_) in enumerate(zip(args.tau_multiples, map_fits, spreads))]
    # `latent` is the evidence-chosen τ: a copy of its rows.
    for k in FRAMES:
        extra = []
        for frame in frames[k]:
            for src, dst in ((tau_label(m_star), "latent"), (f"{tau_label(m_star)}:p", "latent:p")):
                sel = frame[frame["model"] == src]
                if len(sel):
                    extra.append(sel.assign(model=dst))
        frames[k].extend(extra)
    res = conditional("latent_n", bases[m_star], kappa=n_A * share)
    score("latent_n", share[:, None] * res.r, True)

    for k, parts in frames.items():
        write_text(out / f"{k}.csv", pd.concat(parts, ignore_index=True).to_csv(index=False))
    write_text(out / "fits.csv", pd.DataFrame(fits).to_csv(index=False))
    write_text(out / "tau.csv", pd.DataFrame(tau_rows).to_csv(index=False))
    write_text(out / "starts.csv", pd.DataFrame(start_rows).to_csv(index=False))
    save_npz(out / "map.npz", theta=ref.theta, tau_multiples=np.array(args.tau_multiples),
             tau=np.array([f.tau for f in map_fits]), U=np.array([f.U for f in map_fits]),
             log_evidence=np.array([f.log_evidence for f in map_fits]), chosen=np.array(chosen),
             zones=np.array(region.pumas))
    save_npz(out / "realized.npz", **realized, zones=np.array(region.pumas),
             region_units=problem.units)
    seconds = time.perf_counter() - start
    write_text(out / "timings.json", json.dumps({
        **tag, "seconds": seconds, "parts": times, "peak_gb": peak_gb(), "Z": Z,
        "rows": int(area.n_rows), "threads": args.threads,
        "map_starts": args.map_starts,
        "map_evaluations": [r["evaluations_all_starts"] for r in spreads],
        "map_seconds_per_evaluation": [f.info()["seconds_per_evaluation"] for f in map_fits]},
        indent=1))
    write_text(out / "done.json", json.dumps({"seconds": seconds, "chosen_tau_multiple": m_star}))
    logger.info("region %s %s: done in %.0fs (chosen tau = %g tau_ref)", region_id, mode,
                seconds, m_star)
    return region_id, mode, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="stage 1 run directory (z)")
    parser.add_argument("--step1", required=True, type=Path, help="step 1 run root")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--region", nargs="+", default=None, help="subset of region ids")
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--tau-ref", type=float, default=TAU_REF)
    parser.add_argument("--tau-multiples", type=float, nargs="+", default=list(TAU_MULTIPLES))
    parser.add_argument("--sigma-alpha", type=float, default=0.01)
    parser.add_argument("--prior-sd", type=float, default=3.0, help="s in omega = N / (n s)")
    parser.add_argument("--map-starts", required=True, choices=list(START_SETS),
                        help="the MAP's set of starts (PLAN_latent_step3.md §4.3b), chosen from "
                             "the diagnostic (step3_diagnostic.py)")
    parser.add_argument("--map-gtol", type=float, default=1e-4,
                        help="L-BFGS stop: largest gradient component")
    parser.add_argument("--map-max-iter", type=int, default=20000)
    parser.add_argument("--cond-tol", type=float, default=1e-10,
                        help="conditional PMEDM stop: Newton decrement")
    parser.add_argument("--cond-max-iter", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1, help="processes (work units)")
    parser.add_argument("--threads", type=int, default=1,
                        help="threads per process for the engine's sparse products")
    args = parser.parse_args()
    os.environ.setdefault(OFFLINE_ENV, "1")
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)

    config = load_config(args.config)
    regions = {r.id: r for r in load_regions(args.regions)}
    chosen = args.region or list(regions)
    unknown = [r for r in chosen if r not in regions]
    if unknown:
        raise SystemExit(f"unknown regions {unknown}")
    step1 = load_step1(args.step1)
    for note in step1.notes:
        logger.warning("%s", note)
    with stage("load PUMS, tables and replicate weights"):
        data = load_check_data(config)
    with stage("atoms"):
        atoms = load_atoms(data, args.run)
    if not pd.Index(atoms.serial.astype(str)).equals(pd.Index(step1.serial)):
        raise SystemExit("the atoms are not step 1's (SERIALNO order differs)")
    with stage(f"basis {step1.h}") as step:
        H = load_basis(step1.root / "inputs" / "bases", step1.h, atoms, args.run)
        step.detail = f"K = {H.shape[1]}"
    del atoms
    sets = transfer_sets(data)
    T_all = region_contributions(data, sets["A"])
    B_all = region_contributions(data, sets["B"])
    with stage("pair operator over the S1 cells") as step:
        pairs = PairOperator(T_all, workers=args.threads)
        step.detail = json.dumps(pairs.summary())
    frame = pd.DataFrame({"SERIALNO": data.units.index.astype(str), "weight": data.weight,
                          "is_group_quarters": data.units["is_group_quarters"].to_numpy()})
    write_text(args.out / "settings.json", json.dumps({
        "h": step1.h, "gamma": step1.gamma, "sigma_theta": step1.sigma, "K": int(H.shape[1]),
        "M": int(H.shape[0]), "tau_ref": args.tau_ref, "tau_multiples": args.tau_multiples,
        "sigma_alpha": args.sigma_alpha, "taper": None, "variance_floor": None,
        "prior_sd_s": args.prior_sd, "modes": args.modes, "tables_A": list(sets["A"]),
        "tables_B": list(sets["B"]), "step1": str(step1.root), "run": str(args.run),
        "map_starts": args.map_starts, "map_gtol": args.map_gtol, "map_max_iter": args.map_max_iter,
        "cond_tol": args.cond_tol, "pair_operator": pairs.summary(),
        "step1_notes": step1.notes}, indent=1, default=str))
    _SHARED.update(data=data, args=args, regions=regions, chosen=chosen, step1=step1, H=H,
                   sets=sets, T_all=T_all, B_all=B_all, pairs=pairs, frame=frame)
    with stage(f"region references ({len(chosen)} regions)"):
        run_references(args)
    # Largest regions first, so the pool's last units are short.
    units = sorted(((r, m) for m in args.modes for r in chosen),
                   key=lambda u: -len(regions[u[0]].pumas))
    logger.info("%d work units (%d regions x %d modes), %d processes x %d threads", len(units),
                len(chosen), len(args.modes), args.workers, args.threads)
    if args.workers > 1 and len(units) > 1:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for region_id, mode, seconds in pool.map(run_unit, units):
                logger.info("unit %s %s finished (%.0fs)", region_id, mode, seconds)
    else:
        for unit in units:
            run_unit(unit)


if __name__ == "__main__":
    main()
