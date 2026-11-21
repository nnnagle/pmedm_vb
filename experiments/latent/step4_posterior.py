"""Step 4 of the latent-population PMEDM: the stage 2 posterior and its
calibration on the regional run with survey error (``PLAN_latent_step4.md``
version 1; model ``MODEL_latent.qmd`` version 3).

    python experiments/latent/step4_posterior.py --config experiments/embed/stage1_v2.json \
        --regions experiments/latent/regions_tn.json \
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d16_g0.5_r0.01 \
        --step1 $PMEDM_VB_RUNS/latent/step1_sigma --step3 $PMEDM_VB_RUNS/latent/step3 \
        --step3b $PMEDM_VB_RUNS/latent/step3b --out $PMEDM_VB_RUNS/latent/step4 \
        --workers 16 --threads 1
    python experiments/latent/collect_step4.py --root $PMEDM_VB_RUNS/latent/step4 \
        --step3b $PMEDM_VB_RUNS/latent/step3b

The model as fixed in step 3b: ``h`` = step 1's basis (``kern100``) with
``θ_A`` from step 3 (``<step3>/<region>/reference.npz``), ``g`` = ``kern30``
(``--basis``; ``τ^g_ref`` from step 3b's ``tau_ref.json``), ``γ`` and
``σ_θ`` from step 1, entropy weight ``N_b``, leave-zone-out support, the B1/B2
split of step 3b (its ``settings.json``; checked against the committed
``step3b_split.json``).

**Work units:** (simulation, region, replicate), simulations ``G`` and ``S``
(:mod:`pmedm_vb.latent.calibration`), ``--replicates`` (10) each, data seeds
fixed by (``--seed``, simulation, region, replicate). Per unit:

1. the replicate's published tables: the fitted S1 cells and the held-out
   B1 and B2 cells, drawn jointly (§3);
2. ``Σ`` for fitting: G, the region's ``Σ_survey`` plus ``ω² I``
   (:func:`~pmedm_vb.latent.region_engine.region_sigma`); S, ``Σ̂`` plus
   ``ω² I`` (:class:`~pmedm_vb.latent.calibration.BlockSigma`);
3. for each ``τ ∈ --tau-multiples × τ^g_ref`` (default
   ``{0.0625, 0.125, 0.25, 0.5}``), increasing: the MAP over the ``all4``
   starts with plain L-BFGS (the lowest exact objective), the conditional
   PMEDM at ``p_b(Û)``, and its misfit on B1 (published SEs of the replicate);
   ``τ`` chosen by B1 (§4.1);
4. at the chosen ``τ``: Laplace and ``--draws`` (200) posterior draws of
   every functional (:mod:`pmedm_vb.latent.posterior`), their interval
   statistics against truth, and for the fitted cells the published SE and the
   direct estimate's 90% interval ``Y ± 1.645 SE``;
5. the MAP's point accuracy: step 3's scoring of the realized MAP at the
   chosen ``τ`` (``latent_kern30``) and of ``p_b`` (``:p``), against truth,
   with transfer misfit on B1 and B2 (the replicate's published tables).

Each unit writes ``<out>/<simulation>/<region>/r<k>/``: ``posterior.csv.gz``
(one row per zone and functional), ``tau.csv``, ``starts.csv``, ``fits.csv``,
``scores``, ``rare``, ``misfit``, ``transfer`` CSVs, ``map.npz`` and
``done.json`` (with timings); a unit with ``done.json`` is skipped, so a
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
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.assemble.prior import prior_variance
from pmedm_vb.data.cache import OFFLINE_ENV
from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.puma_check import load_check_data
from pmedm_vb.latent.area import load_basis, load_step1
from pmedm_vb.latent.atoms import load_atoms
from pmedm_vb.latent.calibration import (
    SIMULATIONS,
    BlockSigma,
    draw_bernoulli,
    draw_gaussian,
    interval_stats,
    with_tables,
)
from pmedm_vb.latent.collapsed import PairOperator
from pmedm_vb.latent.conditional import solve_conditional
from pmedm_vb.latent.deviation_basis import DEVIATION_BASES, deviation_basis
from pmedm_vb.latent.map_fit import fit_starts
from pmedm_vb.latent.posterior import build_functionals, posterior_samples, zone_truth
from pmedm_vb.latent.reference import log_base
from pmedm_vb.latent.region_engine import (
    deviation_engine,
    region_area,
    region_contributions,
    region_sigma,
)
from pmedm_vb.latent.region_reference import region_distance, zone_log_bases, zone_of_atoms
from pmedm_vb.latent.regions import build_region, load_regions, transfer_sets
from pmedm_vb.latent.scoring import score_region, transfer_misfit
from pmedm_vb.progress import logger, record_run, save_npz, stage, write_text

_SHARED: dict = {}

MODE = "lzo"
TAU_MULTIPLES = (0.0625, 0.125, 0.25, 0.5)
SPLIT_FILE = Path(__file__).with_name("step3b_split.json")
SCORED = ("B1", "B2")


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def statewide_view(problem, X_all, frame, zone_all):
    inputs = replace(problem.inputs, X_B=X_all, units=frame)
    return replace(problem, inputs=inputs, units=np.arange(X_all.shape[0]),
                   zone_of_unit=zone_all)


def run_unit(unit: tuple[str, str, int]) -> tuple[str, str, int, float]:
    sim, region_id, k = unit
    s, args = _SHARED, _SHARED["args"]
    out = args.out / sim / region_id / f"r{k}"
    if (out / "done.json").exists():
        return sim, region_id, k, 0.0
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    times = {}
    data, sets, X = s["data"], s["sets"], s["X"]
    region, ref = s["regions"][region_id], s["refs"][region_id]
    index = list(s["regions"]).index(region_id)
    rng = np.random.default_rng([args.seed, SIMULATIONS.index(sim), index, k])

    # 1-2. The replicate's tables and Σ.
    problems = {key: build_region(data, region, tables=sets[key]) for key in ("A", *SCORED)}
    joint = build_region(data, region, tables=(*sets["A"], *sets["B1"], *sets["B2"]))
    draw = draw_gaussian(joint, rng) if sim == "G" else draw_bernoulli(joint, rng)
    sizes = [len(problems[key].cells) for key in ("A", *SCORED)]
    cuts = np.cumsum([0, *sizes])
    reps = {key: with_tables(problems[key], np.arange(cuts[j], cuts[j + 1]), draw)
            for j, key in enumerate(("A", *SCORED))}
    pA = reps["A"]
    inputs = pA.inputs
    Z = inputs.n_zones
    N_b = np.asarray(pA.zone_weight, float)
    share = N_b / float(inputs.N)
    if sim == "G":
        sigma = region_sigma(pA, args.sigma_alpha, args.prior_sd)
    else:
        blocks = [B[:sizes[0], :sizes[0]] for B in draw.blocks]
        sigma = BlockSigma(blocks, draw.floor[:, :sizes[0]], Z,
                           prior_variance(args.prior_sd, inputs.n, inputs.N))
    sigma_dense = sigma.to_dense()
    area = region_area(pA, sigma)
    views = {key: statewide_view(reps[key], X[key], s["frame"], ref["zone"])
             for key in ("A", *SCORED)}
    engine = deviation_engine(area, s["H"], ref["theta"], s["G"],
                              zone_log_bases(ref["logb"], ref["zone"], Z, MODE), s["pairs"])
    theta0 = np.zeros(s["G"].shape[1])
    se_cells = np.sqrt(inputs.sigma_v)
    tag = {"simulation": sim, "region": region_id, "replicate": k}

    # 3. MAP per τ, the conditional PMEDM, B1 misfit.
    kw = {"gtol": args.map_gtol, "max_iter": args.map_max_iter, "optimizer": "m10"}
    previous, rows, start_rows, fit_rows, maps = None, [], [], [], {}
    for m in args.tau_multiples:
        t0 = time.perf_counter()
        tau = m * s["tau_ref_g"]
        fit, starts = fit_starts(engine, theta0, tau, "all4", previous, **kw)
        p_b = engine.probabilities(theta0, fit.U).T
        res = solve_conditional(X["A"], p_b, N_b, inputs.targets(), sigma_dense,
                                tol=args.cond_tol, max_iter=args.cond_max_iter)
        b1 = transfer_misfit(views["B1"], share[:, None] * res.r, "x", "A", "B1")["rms_z"].mean()
        values = [f.objective for f in starts]
        rows.append({**tag, "tau_multiple": m, "tau": tau, **fit.info(),
                     "n_starts": len(starts), "spread": max(values) - min(values),
                     "evaluations_all_starts": sum(f.evaluations for f in starts),
                     "b1_rms_z": float(b1), "seconds_tau": time.perf_counter() - t0})
        start_rows += [{**tag, "tau_multiple": m, **f.info(), "selected": f is fit}
                       for f in starts]
        fit_rows.append({**tag, "tau_multiple": m, "model": "conditional", **res.info()})
        maps[m] = (fit.U, p_b, res)
        previous = fit.U
        logger.info("%s %s r%d tau %g: MAP from %s (spread %.2f), B1 %.3f, %.0fs", sim,
                    region_id, k, m, fit.start, rows[-1]["spread"], b1, rows[-1]["seconds_tau"])
    chosen = int(np.argmin([r["b1_rms_z"] for r in rows]))
    m_star = args.tau_multiples[chosen]
    for j, r in enumerate(rows):
        r["chosen_b1"] = j == chosen
    U_hat, p_b, res = maps[m_star]
    times["map"] = float(sum(r["seconds_tau"] for r in rows))

    # 4. Posterior at the chosen τ.
    t0 = time.perf_counter()
    post_rng = np.random.default_rng([args.seed, SIMULATIONS.index(sim), index, k, 1])
    sample = posterior_samples(engine, theta0, U_hat, m_star * s["tau_ref_g"], s["fun"],
                               X["A"], args.draws, post_rng)
    truth = zone_truth(s["fun"], ref["zone"], data.weight, Z)
    st = interval_stats(sample.samples, truth)
    times["posterior"] = time.perf_counter() - t0
    fun = s["fun"]
    frame = pd.DataFrame({
        "zone": np.repeat(np.array(region.pumas), fun.F),
        "cls": np.tile(fun.cls, Z), "functional": np.tile(fun.names, Z),
        "truth": truth.ravel(), **{key: np.asarray(v).ravel() for key, v in st.items()}})
    fitted = fun.of("fitted")
    se = np.sqrt(draw.var[:, :sizes[0]])                  # (Z, C_A) published SE
    Yf = draw.Y[:, :sizes[0]]
    extra = {"se": np.full((Z, fun.F), np.nan), "published": np.full((Z, fun.F), np.nan)}
    extra["se"][:, fitted] = se
    extra["published"][:, fitted] = Yf
    frame["se"] = extra["se"].ravel()
    frame["published"] = extra["published"].ravel()
    frame["direct_cover90"] = np.where(frame["se"].notna(),
                                       np.abs(frame["published"] - frame["truth"])
                                       <= 1.645 * frame["se"], np.nan)
    frame["width90_over_se"] = frame["width90"] / (2 * frame["se"])
    frame["width90_over_truth"] = np.where(frame["truth"] > 0,
                                           frame["width90"] / frame["truth"], np.nan)
    frame = frame.assign(**tag, tau_multiple=m_star)
    frame.to_csv(out / "posterior.csv.gz", index=False)

    # 5. The MAP's point accuracy.
    parts = {key: [] for key in ("scores", "rare", "misfit", "transfer")}
    for name, pm in (("latent_kern30", share[:, None] * res.r),
                     ("latent_kern30:p", share[:, None] * p_b)):
        scored = score_region(data, views["A"], pm, name, se_cells)
        scored["transfer"] = pd.concat([transfer_misfit(views[key], pm, name, "A", key)
                                        for key in SCORED], ignore_index=True)
        for key in parts:
            parts[key].append(scored[key].assign(**tag, tau_multiple=m_star))
    for key, p in parts.items():
        write_text(out / f"{key}.csv", pd.concat(p, ignore_index=True).to_csv(index=False))
    write_text(out / "tau.csv", pd.DataFrame(rows).to_csv(index=False))
    write_text(out / "starts.csv", pd.DataFrame(start_rows).to_csv(index=False))
    write_text(out / "fits.csv", pd.DataFrame(fit_rows).to_csv(index=False))
    save_npz(out / "map.npz", U=U_hat, tau_multiple=np.array(m_star),
             Y=draw.Y, var=draw.var, zones=np.array(region.pumas))
    seconds = time.perf_counter() - start
    write_text(out / "done.json", json.dumps({
        **tag, "seconds": seconds, "parts": times, "chosen_tau_multiple": m_star,
        "draws": args.draws, "negative_variances": sample.negative_variances,
        "seconds_per_draw": sample.seconds_per_draw, "Z": Z, "peak_gb": peak_gb(),
        "sigma_rank": int(getattr(sigma, "rank", 0) or 0)}, default=float))
    logger.info("%s %s r%d: done in %.0fs (tau %g, posterior %.0fs)", sim, region_id, k,
                seconds, m_star, times["posterior"])
    return sim, region_id, k, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="stage 1 run directory (z)")
    parser.add_argument("--step1", required=True, type=Path, help="step 1 run root")
    parser.add_argument("--step3", required=True, type=Path, help="step 3 output root")
    parser.add_argument("--step3b", required=True, type=Path, help="step 3b output root")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--region", nargs="+", default=None, help="subset of region ids")
    parser.add_argument("--simulations", nargs="+", default=list(SIMULATIONS),
                        choices=list(SIMULATIONS))
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--basis", default="kern30", choices=list(DEVIATION_BASES))
    parser.add_argument("--tau-multiples", type=float, nargs="+", default=list(TAU_MULTIPLES))
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
    step1 = load_step1(args.step1)
    s3b = json.loads((args.step3b / "settings.json").read_text())
    tau_refs = json.loads((args.step3b / "tau_ref.json").read_text())
    if args.basis not in tau_refs:
        raise SystemExit(f"step 3b has no tau_ref for {args.basis}")
    tau_ref_g = float(tau_refs[args.basis]["tau_ref"])
    with stage("load PUMS, tables and replicate weights"):
        data = load_check_data(config)
    with stage("atoms"):
        atoms = load_atoms(data, args.run)
    if not pd.Index(atoms.serial.astype(str)).equals(pd.Index(step1.serial)):
        raise SystemExit("the atoms are not step 1's (SERIALNO order differs)")
    bases_dir = step1.root / "inputs" / "bases"
    H = load_basis(bases_dir, step1.h, atoms, args.run)
    G, basis_info = deviation_basis(args.basis, atoms, bases_dir, args.run)
    del atoms
    split = json.loads(SPLIT_FILE.read_text())
    used = s3b.get("split", {"B1": split["B1"], "B2": split["B2"]})
    if list(used["B1"]) != list(split["B1"]) or list(used["B2"]) != list(split["B2"]):
        raise SystemExit("step 3b's B1/B2 split differs from step3b_split.json")
    sets = {"A": transfer_sets(data)["A"], "B1": tuple(split["B1"]), "B2": tuple(split["B2"])}
    X = {key: region_contributions(data, t) for key, t in sets.items()}
    with stage("pair operator over the S1 cells"):
        pairs = PairOperator(X["A"], workers=args.threads)
    fun = build_functionals(data, sets)
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
        "mode": MODE, "h": step1.h, "basis": basis_info, "tau_ref_g": tau_ref_g,
        "tau_multiples": args.tau_multiples, "gamma": step1.gamma, "sigma_theta": step1.sigma,
        "simulations": args.simulations, "replicates": args.replicates, "draws": args.draws,
        "seed": args.seed, "split": {key: list(v) for key, v in sets.items()},
        "functionals": {c: int((fun.cls == c).sum()) for c in np.unique(fun.cls)},
        "map_starts": "all4", "optimizer": "m10", "sigma_alpha": args.sigma_alpha,
        "prior_sd": args.prior_sd, "step1": str(step1.root), "step3": str(args.step3),
        "step3b": str(args.step3b), "run": str(args.run)}, indent=1, default=str))
    _SHARED.update(data=data, args=args, regions=regions, refs=refs, H=H, G=G, sets=sets, X=X,
                   pairs=pairs, fun=fun, frame=frame, tau_ref_g=tau_ref_g)
    units = sorted(((sim, r, k) for sim in args.simulations for r in chosen
                    for k in range(args.replicates)),
                   key=lambda u: (-len(regions[u[1]].pumas), u[2], u[0]))
    logger.info("%d work units (%d simulations x %d regions x %d replicates), %d processes",
                len(units), len(args.simulations), len(chosen), args.replicates, args.workers)
    if args.workers > 1 and len(units) > 1:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for sim, rid, k, seconds in pool.map(run_unit, units):
                logger.info("unit %s %s r%d finished (%.0fs)", sim, rid, k, seconds)
    else:
        for unit in units:
            run_unit(unit)


if __name__ == "__main__":
    main()
