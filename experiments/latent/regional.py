"""Stage 2 regional run: regions of PUMAs as areas, their PUMAs as zones, the
PUMS as the truth.

    python experiments/latent/regional.py --config experiments/embed/stage1_v2.json \
        --regions experiments/latent/regions_tn.json \
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d16_g0.5_r0.01 \
        --out $PMEDM_VB_RUNS/latent/regional --workers 8

For every region (:mod:`pmedm_vb.latent.regions`), fits and scores:

``none``                      the design distribution q (every unit spread evenly);
``pmedm``                     today's PMEDM on the same inputs: one multiplier per
                              cell per zone, the existing dual solver
                              (:func:`~pmedm_vb.solvers.map_dual.solve_map`), Σ shrunk by
                              ``--sigma-alpha`` with no taper, and the paper
                              baseline's multiplier prior (hierarchy
                              ``none+prior<s>``; the baseline's ``puma``
                              hierarchy needs tract rows, which a region does
                              not have);
``intercepts_a<α>``           the latent model with no features of φ: shared x
                              multipliers and zone intercepts only (what φ adds
                              is measured against this);
``latent_a<α>``               the latent model, h = the decoder features of φ
                              (``HANDOFF_stage2.md``);
``latent_race_a<α>``          the same with each household's race/ethnicity
                              member shares appended to h.

Every latent model is fitted from two starts, ``η = 0`` and the projection of
the PMEDM's solution onto the family
(:meth:`~pmedm_vb.latent.model.LatentModel.project`), and the lower objective
kept, since the objective is not convex; both objectives are in ``fits.csv``.
The PMEDM is solved for every region, also under ``--no-pmedm`` (which then
only leaves it unscored).

Results per region go to ``<out>/<region>/`` (``scores``, ``rare``,
``misfit``, ``spread``, ``summary``, ``fits`` CSVs); a region whose models are
all recorded in its ``done.json`` is skipped, so a timed-out job can be
resubmitted as it was. Pool with ``experiments/latent/collect_regional.py``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.puma_check import DECODER_ATTRIBUTES, decoder_features, load_check_data, load_phi
from pmedm_vb.latent.model import LatentModel, area_sigma
from pmedm_vb.latent.regions import build_region, load_regions, race_composition
from pmedm_vb.latent.scoring import score_region
from pmedm_vb.progress import logger, record_run, stage, write_text
from pmedm_vb.solvers.map_dual import solve_map

#: Filled before the worker pool forks, so workers share it without pickling.
_SHARED: dict = {}


def model_names(alphas, variants, pmedm: bool) -> list[str]:
    names = ["none"] + (["pmedm"] if pmedm else [])
    names += [f"{v}_a{a:g}" for v in variants for a in alphas]
    return names


def run_region(region_id: str) -> str:
    s = _SHARED
    data, args, region = s["data"], s["args"], s["regions"][region_id]
    out = args.out / region_id
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    wanted = model_names(args.alpha, args.variants, not args.no_pmedm)
    todo = [m for m in wanted if m not in done]
    if not todo:
        logger.info("region %s: all models done, skipped", region_id)
        return region_id

    problem = build_region(data, region, args.set)
    inputs = problem.inputs
    sigma = area_sigma(inputs, sigma_alpha=args.sigma_alpha, prior_sd=args.prior_sd)
    se_cells = np.sqrt(inputs.sigma_v)
    H_dec = s["features"].F[problem.units]
    names_dec, binary_dec = s["features"].columns, s["features"].binary
    shares, share_names = race_composition(data, problem.units)
    logger.info("region %s: %d zones, %s units, %d cells", region_id, inputs.n_zones,
                f"{inputs.n_units:,}", len(problem.cells))

    start = time.perf_counter()
    pmedm = solve_map(inputs, alpha=args.sigma_alpha, taper=None,
                      hierarchy=f"none+prior{args.prior_sd:g}", max_iter=200)
    pmedm_seconds = time.perf_counter() - start

    frames = {k: [] for k in ("scores", "rare", "misfit", "spread", "summary", "fits")}
    for model in todo:
        start = time.perf_counter()
        info = {"region": region_id, "model": model}
        if model == "none":
            p = inputs.q
            info.update(converged=True, iterations=0)
        elif model == "pmedm":
            p = pmedm.W / inputs.N
            info.update(converged=bool(pmedm.converged), iterations=int(pmedm.n_iter),
                        objective=float(pmedm.objective))
            start -= pmedm_seconds
        else:
            variant, alpha = model.rsplit("_a", 1)
            alpha = float(alpha)
            if variant == "intercepts":
                H, binary, names = np.zeros((inputs.n_units, 0)), np.zeros(0, bool), []
            elif variant == "latent":
                H, binary, names = H_dec, binary_dec, names_dec
            elif variant == "latent_race":
                H = np.column_stack([H_dec, shares])
                binary = np.concatenate([binary_dec, np.zeros(shares.shape[1], bool)])
                names = [*names_dec, *share_names]
            else:
                raise ValueError(f"unknown variant {variant!r}")
            m = LatentModel(inputs, sigma, H, binary, zone_weight=problem.zone_weight,
                            alpha=alpha, tau_theta=args.tau_theta,
                            tau_intercept=args.tau_intercept, names=names)
            # Two starts, the better kept: the objective is not convex.
            fits = {
                "zero": m.fit(solver=args.solver, label=f"region {region_id} {model} from 0"),
                "pmedm": m.fit(solver=args.solver, init=m.project(pmedm.W / inputs.N),
                               label=f"region {region_id} {model} from PMEDM"),
            }
            best = min(fits, key=lambda k: fits[k].objective)
            fit = fits[best]
            p = fit.p
            info.update(converged=fit.converged, iterations=fit.iterations,
                        objective=fit.objective, solver=fit.solver, message=fit.message,
                        n_params=m.n_params, start=best,
                        **{f"objective_from_{k}": f.objective for k, f in fits.items()},
                        **{f"converged_from_{k}": f.converged for k, f in fits.items()},
                        **{f"part_{k}": v for k, v in fit.parts.items()})
            np.savez_compressed(out / f"theta_{model}.npz", theta=fit.theta, lam=fit.lam,
                                features=np.array(fit.feature_names, dtype=object),
                                zones=np.array(region.pumas, dtype=object))
        info["seconds"] = time.perf_counter() - start
        scored = score_region(data, problem, p, model, se_cells)
        for k, v in scored.items():
            frames[k].append(v)
        frames["fits"].append(pd.DataFrame([info]))
        logger.info("region %s %s: allocation accuracy %.3f, %.1fs", region_id, model,
                    float(scored["summary"]["allocation_accuracy"].iloc[0]), info["seconds"])
        for k, parts in frames.items():
            if parts:
                path = out / f"{k}.csv"
                new = pd.concat(parts, ignore_index=True)
                if path.exists():
                    old = pd.read_csv(path, dtype={"zone": str})
                    old = old[~old["model"].isin(new["model"].unique())]
                    new = pd.concat([old, new], ignore_index=True)
                write_text(path, new.to_csv(index=False))
            frames[k] = []
        done.add(model)
        write_text(done_path, json.dumps(sorted(done)))
    return region_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--regions", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="stage 1 run directory (φ)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.03, 0.1, 0.3])
    parser.add_argument("--variants", nargs="+", default=["intercepts", "latent", "latent_race"],
                        choices=["intercepts", "latent", "latent_race"])
    parser.add_argument("--no-pmedm", action="store_true", help="skip today's PMEDM")
    parser.add_argument("--region", nargs="+", default=None, help="subset of region ids")
    parser.add_argument("--set", default="S1", help="constraint set of the check (default S1)")
    parser.add_argument("--sigma-alpha", type=float, default=0.01)
    parser.add_argument("--prior-sd", type=float, default=3.0)
    parser.add_argument("--tau-theta", type=float, default=3.0)
    parser.add_argument("--tau-intercept", type=float, default=3.0)
    parser.add_argument("--solver", default="auto", choices=["auto", "newton", "gn", "lbfgs"])
    parser.add_argument("--decoder-attributes", nargs="+", default=list(DECODER_ATTRIBUTES))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    config = load_config(args.config)
    regions = {r.id: r for r in load_regions(args.regions)}
    chosen = args.region or list(regions)
    unknown = [r for r in chosen if r not in regions]
    if unknown:
        raise SystemExit(f"unknown regions {unknown}")
    with stage("load PUMS, tables and replicate weights"):
        data = load_check_data(config)
    with stage("positions and decoder features"):
        phi, columns, binary = load_phi(args.run, data.units.index)
        features = decoder_features(args.run, phi, columns, binary, data.weight,
                                    tuple(args.decoder_attributes))
    _SHARED.update(data=data, args=args, regions=regions, features=features)
    if args.workers > 1 and len(chosen) > 1:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
            for region_id in pool.map(run_region, chosen):
                logger.info("region %s finished", region_id)
    else:
        for region_id in chosen:
            run_region(region_id)


if __name__ == "__main__":
    main()
