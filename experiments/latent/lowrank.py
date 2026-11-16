"""Stage 2 regional run, low-rank models: θ_b = W u_b with ``W`` shared by every
zone of every region, so all regions are fitted jointly
(:mod:`pmedm_vb.latent.lowrank`).

    python experiments/latent/lowrank.py --config experiments/embed/stage1_v2.json \
        --regions experiments/latent/regions_tn.json \
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d16_g0.5_r0.01 \
        --out $PMEDM_VB_RUNS/latent/lowrank \
        --theta-from $PMEDM_VB_RUNS/latent/regional --workers 7

One configuration per (features, rank, fitted table set): by default the
7-attribute and all-attribute (``h37``) category-probability features at
r = 4, 8, 16 fitted to table set A, plus the 7-attribute features at r = 8
fitted to B for the transfer test; α = 0.1 throughout. The configurations run
in parallel, ``--workers`` at a time. Each is named
``lowrank_<h>_r<r>_a<α>[@B]`` and writes ``<out>/<name>/<region>/`` with the
same CSVs as ``regional.py`` (``scores``, ``rare``, ``misfit``, ``spread``,
``summary``, ``transfer``), plus ``<out>/<name>/starts.csv`` (both starts) and
``W.npz``. A configuration with a ``done.json`` is skipped.

Starts: ``pca`` always; ``svd`` for the 7-attribute features when
``--theta-from`` holds the per-region latent-only fits
(``theta_lonly_a<α>[@B].npz``, written by ``regional.py``), so that run should
finish first. The svd start is the full-rank fits truncated to rank r
(:func:`~pmedm_vb.latent.lowrank.svd_start`) and then shrunk towards zero by
the factor with the lowest objective. Each start runs L-BFGS for up to
``--max-iter`` iterations; the fits are slow to converge, so ``starts.csv``
and ``fits.csv`` record convergence and the final largest gradient component. Full-rank latent-only fits with all-attribute features are not
run (about 370 coefficients per zone against 169 cells); those features are
used with low rank only.

Pool with ``experiments/latent/collect_regional.py --root <regional> <lowrank>``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.puma_check import DECODER_ATTRIBUTES, decoder_features, load_check_data, load_phi
from pmedm_vb.latent.lowrank import (
    AreaTerm, LowRankModel, centring_shift, pca_start, statewide_features, svd_start, to_statewide)
from pmedm_vb.latent.model import area_sigma
from pmedm_vb.latent.regions import build_region, load_regions, transfer_sets
from pmedm_vb.latent.scoring import score_region, transfer_misfit
from pmedm_vb.progress import logger, record_run, stage, write_text

_SHARED: dict = {}


def config_name(h: str, rank: int, alpha: float, fit_set: str) -> str:
    return f"lowrank_{h}_r{rank}_a{alpha:g}" + ("" if fit_set == "A" else f"@{fit_set}")


def full_rank_start(theta_dir: Path, problems, fit_set: str, alpha: float, Hg, names_g, spread_g,
                    binary_g):
    """Per-region latent-only fits moved onto the statewide standardisation,
    or None if any region's is missing."""
    suffix = "" if fit_set == "A" else f"@{fit_set}"
    full = {}
    for rid, problem in problems.items():
        path = theta_dir / rid / f"theta_lonly_a{alpha:g}{suffix}.npz"
        if not path.exists():
            return None
        z = np.load(path, allow_pickle=True)
        names = [str(n) for n in z["features"]][:-1]           # last is the intercept
        spread = np.asarray(z["spread"], dtype=np.float64)
        theta = np.asarray(z["theta"], dtype=np.float64)
        theta_g = to_statewide(theta[:, :-1], names, spread, names_g, spread_g)
        weight = problem.inputs.units["weight"].to_numpy(dtype=np.float64)
        full[rid] = {"theta": theta_g,
                     "c": theta[:, -1] + centring_shift(theta_g, Hg[problem.units], binary_g, weight),
                     "beta": to_statewide(np.asarray(z["lam"], dtype=np.float64), names, spread,
                                          names_g, spread_g)}
    return full


def run_config(spec: tuple[str, int, str]) -> str:
    h, rank, fit_set = spec
    s = _SHARED
    args, data = s["args"], s["data"]
    name = config_name(h, rank, args.alpha, fit_set)
    out = args.out / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        logger.info("%s: done, skipped", name)
        return name
    other = "B" if fit_set == "A" else "A"
    Hg, names_g, spread_g, binary_g, cov_g = s["features"][h]
    terms = {}
    for rid, pr in s["problems"][fit_set].items():
        terms[rid] = AreaTerm(pr.inputs, s["sigmas"][fit_set][rid], Hg[pr.units], pr.zone_weight,
                              args.alpha)
    model = LowRankModel(terms, rank, tau_beta=args.tau_beta, tau_intercept=args.tau_intercept,
                         tau_u=args.tau_u, tau_w=args.tau_w)
    starts = {"pca": pca_start(model, Hg, data.weight)}
    if h == "h7" and args.theta_from is not None:
        full = full_rank_start(args.theta_from, s["problems"][fit_set], fit_set, args.alpha, Hg,
                               names_g, spread_g, binary_g)
        if full is not None:
            starts["svd"], t = model.shrink(svd_start(model, full, cov_g))
            logger.info("%s: svd start shrunk by %g", name, t)
        else:
            logger.info("%s: no full-rank fits under %s, svd start skipped", name, args.theta_from)
    fits = {k: model.fit(x0, max_iter=args.max_iter, start=k, label=name)
            for k, x0 in starts.items()}
    best = min(fits, key=lambda k: fits[k].objective)
    fit = fits[best]
    rows = [{"model": name, "start": k, "objective": f.objective, "prior": f.prior,
             "iterations": f.iterations, "evaluations": f.evaluations, "converged": f.converged,
             "message": f.message, "grad_max": f.grad_max, "seconds": f.seconds, "kept": k == best,
             "n_params": model.n_params} for k, f in fits.items()]
    write_text(out / "starts.csv", pd.DataFrame(rows).to_csv(index=False))
    np.savez_compressed(out / "W.npz", W=fit.W, features=np.array(names_g, dtype=object))

    for rid in model.ids:
        problem = s["problems"][fit_set][rid]
        p = fit.areas[rid]["p"]
        se_cells = np.sqrt(problem.inputs.sigma_v)
        scored = score_region(data, problem, p, name, se_cells)
        scored["transfer"] = transfer_misfit(s["problems"][other][rid], p, name, fit_set, other)
        scored["fits"] = pd.DataFrame([{
            "region": rid, "model": name, "fit_set": fit_set, "converged": fit.converged,
            "grad_max": fit.grad_max,
            "iterations": fit.iterations, "objective": fit.objective, "start": best,
            "seconds": fit.seconds / len(model.ids),
            **{f"objective_from_{k}": f.objective for k, f in fits.items()},
            **{f"part_{k}": v for k, v in fit.areas[rid]["parts"].items()}}])
        region_out = out / rid
        region_out.mkdir(parents=True, exist_ok=True)
        for k, v in scored.items():
            if "fit_set" not in v:
                v["fit_set"] = fit_set
            write_text(region_out / f"{k}.csv", v.to_csv(index=False))
        np.savez_compressed(region_out / "coefficients.npz", theta=fit.areas[rid]["theta"],
                            c=fit.areas[rid]["c"], beta=fit.areas[rid]["beta"])
    write_text(out / "done.json", json.dumps({"start": best, "objective": fit.objective}))
    logger.info("%s finished: start %s, f %.6g", name, best, fit.objective)
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="stage 1 run directory (φ)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--theta-from", type=Path, default=None,
                        help="regional.py output with the full-rank latent-only fits (svd start)")
    parser.add_argument("--h", nargs="+", default=["h7", "h37"], choices=["h7", "h37"])
    parser.add_argument("--rank", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--transfer-rank", type=int, nargs="*", default=[8],
                        help="ranks also fitted to table set B (7-attribute features)")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--region", nargs="+", default=None, help="subset of region ids")
    parser.add_argument("--sigma-alpha", type=float, default=0.01)
    parser.add_argument("--prior-sd", type=float, default=3.0)
    parser.add_argument("--tau-beta", type=float, default=3.0)
    parser.add_argument("--tau-intercept", type=float, default=3.0)
    parser.add_argument("--tau-u", type=float, default=3.0)
    parser.add_argument("--tau-w", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=30000,
                        help="L-BFGS iterations per start (evaluations capped at twice this)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    config = load_config(args.config)
    regions = load_regions(args.regions)
    if args.region:
        regions = [r for r in regions if r.id in set(args.region)]
    with stage("load PUMS, tables and replicate weights"):
        data = load_check_data(config)
    sets = transfer_sets(data)
    specs = [(h, r, "A") for h in args.h for r in args.rank]
    specs += [("h7", r, "B") for r in args.transfer_rank]
    with stage("positions and category-probability features"):
        phi, columns, binary = load_phi(args.run, data.units.index)
        features = {}
        for h, attributes in (("h7", tuple(DECODER_ATTRIBUTES)), ("h37", "all")):
            if h not in {sp[0] for sp in specs}:
                continue
            fm = decoder_features(args.run, phi, columns, binary, data.weight, attributes)
            Hg, names_g, _, spread_g, binary_g = statewide_features(fm.F, fm.binary, data.weight,
                                                                     fm.columns)
            w = data.weight / data.weight.sum()
            centred = Hg - w @ Hg
            features[h] = (Hg, names_g, spread_g, binary_g, centred.T @ (centred * w[:, None]))
            logger.info("features %s: %d columns", h, Hg.shape[1])
    with stage("region problems for table sets A and B"):
        needed = {sp[2] for sp in specs} | {"A", "B"}
        problems = {k: {r.id: build_region(data, r, tables=sets[k]) for r in regions}
                    for k in needed}
        sigmas = {k: {rid: area_sigma(pr.inputs, sigma_alpha=args.sigma_alpha,
                                      prior_sd=args.prior_sd)
                      for rid, pr in problems[k].items()} for k in needed}
    _SHARED.update(args=args, data=data, features=features, problems=problems, sigmas=sigmas)
    if args.workers > 1 and len(specs) > 1:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for name in pool.map(run_config, specs):
                logger.info("%s returned", name)
    else:
        for spec in specs:
            run_config(spec)


if __name__ == "__main__":
    main()
