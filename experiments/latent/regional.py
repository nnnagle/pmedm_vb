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
                              member shares appended to h;
``lonly_a<α>``                the latent-only model: no x term, the area's
                              level β_A on the same features
                              (``LatentModel(shared="features")``); the
                              primary model;
``lonly_race_a<α>``           latent-only with the race/ethnicity shares;
``pmedm_latentview_k<κ>``     the latent-view PMEDM (:mod:`pmedm_vb.latent.views`):
                              the PMEDM above, same solver and settings, with
                              each unit's cell counts replaced by their
                              least-squares views on the all-attribute
                              decoder features, ``x̂ = [1, h(φ)] G`` (G fitted
                              statewide per table set), and ``κ`` times the
                              approximation-error variance added to the cells'
                              variances.

A suffix ``@B`` fits the model to table set B instead of A
(:func:`~pmedm_vb.latent.regions.transfer_sets`; A is the check's S1 set).
Every model, whichever set it is fitted to, is scored against the truth and
for **transfer**: its misfit to the other set's cells (``transfer.csv``).

The default plan (``--models``) is: on A, ``none``, ``pmedm``,
``intercepts_a0.1``, ``latent_a0.1`` (x-anchored), ``lonly_a0.03``,
``lonly_a0.1``, ``lonly_a0.3`` and ``lonly_race_a0.1``; on B, ``none@B``,
``pmedm@B``, ``latent_a0.1@B`` and ``lonly_a0.1@B``; and the latent-view PMEDM
at κ = 0 and 1 on each set. Earlier names (any variant and α) remain valid.
The views' R² per table go to ``<out>/views_r2.csv``.

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
from pmedm_vb.latent.regions import build_region, load_regions, race_composition, transfer_sets
from pmedm_vb.latent.scoring import score_region, transfer_misfit
from pmedm_vb.latent.views import fit_views, latent_view_inputs
from pmedm_vb.progress import logger, record_run, stage, write_text
from pmedm_vb.solvers.map_dual import solve_map

#: Filled before the worker pool forks, so workers share it without pickling.
_SHARED: dict = {}


VARIANTS = ("intercepts", "latent", "latent_race", "lonly", "lonly_race")

DEFAULT_MODELS = ("none", "pmedm", "intercepts_a0.1", "latent_a0.1", "lonly_a0.03", "lonly_a0.1",
                  "lonly_a0.3", "lonly_race_a0.1",
                  "pmedm_latentview_k0", "pmedm_latentview_k1",
                  "none@B", "pmedm@B", "latent_a0.1@B", "lonly_a0.1@B",
                  "pmedm_latentview_k0@B", "pmedm_latentview_k1@B")


def model_names(alphas, variants, pmedm: bool) -> list[str]:
    names = ["none"] + (["pmedm"] if pmedm else [])
    names += [f"{v}_a{a:g}" for v in variants for a in alphas]
    return names


def parse_model(name: str) -> tuple[str, str, float | None]:
    """``"lonly_a0.1@B"`` -> ``("lonly", "B", 0.1)``; ``"none"`` -> ``("none", "A", None)``;
    ``"pmedm_latentview_k1"`` -> ``("latentview", "A", 1.0)`` (κ in α's place)."""
    base, _, fit_set = name.partition("@")
    fit_set = fit_set or "A"
    if fit_set not in ("A", "B"):
        raise ValueError(f"unknown table set in {name!r}")
    if base in ("none", "pmedm"):
        return base, fit_set, None
    if base.startswith("pmedm_latentview_k"):
        return "latentview", fit_set, float(base[len("pmedm_latentview_k"):])
    variant, alpha = base.rsplit("_a", 1)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant in {name!r}")
    return variant, fit_set, float(alpha)


def run_region(region_id: str) -> str:
    s = _SHARED
    data, args, region = s["data"], s["args"], s["regions"][region_id]
    out = args.out / region_id
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "done.json"
    done = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    wanted = args.models
    todo = [m for m in wanted if m not in done]
    if not todo:
        logger.info("region %s: all models done, skipped", region_id)
        return region_id

    sets = s["sets"]
    problems = {k: build_region(data, region, tables=sets[k]) for k in ("A", "B")}
    inputs_a = problems["A"].inputs
    sigmas = {k: area_sigma(pr.inputs, sigma_alpha=args.sigma_alpha, prior_sd=args.prior_sd)
              for k, pr in problems.items()}
    units = problems["A"].units
    H_dec = s["features"].F[units]
    names_dec, binary_dec = s["features"].columns, s["features"].binary
    shares, share_names = race_composition(data, units)
    logger.info("region %s: %d zones, %s units, %d cells in A, %d in B", region_id,
                inputs_a.n_zones, f"{inputs_a.n_units:,}", len(problems["A"].cells),
                len(problems["B"].cells))

    fit_sets = sorted({parse_model(m)[1] for m in todo})
    pmedm, pmedm_seconds = {}, {}
    for k in fit_sets:
        t0 = time.perf_counter()
        pmedm[k] = solve_map(problems[k].inputs, alpha=args.sigma_alpha, taper=None,
                             hierarchy=f"none+prior{args.prior_sd:g}", max_iter=200)
        pmedm_seconds[k] = time.perf_counter() - t0

    frames = {k: [] for k in ("scores", "rare", "misfit", "spread", "summary", "fits", "transfer")}
    for model in todo:
        variant, fit_set, alpha = parse_model(model)
        problem, sigma = problems[fit_set], sigmas[fit_set]
        inputs = problem.inputs
        other = "B" if fit_set == "A" else "A"
        start = time.perf_counter()
        info = {"region": region_id, "model": model, "fit_set": fit_set}
        if variant == "none":
            p = inputs.q
            info.update(converged=True, iterations=0)
        elif variant == "pmedm":
            p = pmedm[fit_set].W / inputs.N
            info.update(converged=bool(pmedm[fit_set].converged),
                        iterations=int(pmedm[fit_set].n_iter),
                        objective=float(pmedm[fit_set].objective))
            start -= pmedm_seconds[fit_set]
        elif variant == "latentview":
            view_inputs = latent_view_inputs(problem, s["views"][fit_set], s["features_all"][units],
                                             kappa=alpha)
            res = solve_map(view_inputs, alpha=args.sigma_alpha, taper=None,
                            hierarchy=f"none+prior{args.prior_sd:g}", max_iter=200)
            p = res.W / inputs.N
            info.update(converged=bool(res.converged), iterations=int(res.n_iter),
                        objective=float(res.objective), kappa=alpha)
        else:
            shared = "features" if variant.startswith("lonly") else "x"
            if variant == "intercepts":
                H, binary, names = np.zeros((inputs.n_units, 0)), np.zeros(0, bool), []
            elif variant in ("latent", "lonly"):
                H, binary, names = H_dec, binary_dec, names_dec
            else:
                H = np.column_stack([H_dec, shares])
                binary = np.concatenate([binary_dec, np.zeros(shares.shape[1], bool)])
                names = [*names_dec, *share_names]
            m = LatentModel(inputs, sigma, H, binary, zone_weight=problem.zone_weight,
                            alpha=alpha, tau_theta=args.tau_theta,
                            tau_intercept=args.tau_intercept, names=names,
                            shared=shared, tau_shared=args.tau_shared)
            # Two starts, the better kept: the objective is not convex.
            fits = {
                "zero": m.fit(solver=args.solver, label=f"region {region_id} {model} from 0"),
                "pmedm": m.fit(solver=args.solver, init=m.project(pmedm[fit_set].W / inputs.N),
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
                                spread=m.feature_spread, shared=np.array(m.shared),
                                zones=np.array(region.pumas, dtype=object))
        info["seconds"] = time.perf_counter() - start
        se_cells = np.sqrt(inputs.sigma_v)
        scored = score_region(data, problem, p, model, se_cells)
        scored["transfer"] = transfer_misfit(problems[other], p, model, fit_set, other)
        for k, v in scored.items():
            if "fit_set" not in v:
                v["fit_set"] = fit_set
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
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                        help="models to fit (see the module docstring)")
    parser.add_argument("--region", nargs="+", default=None, help="subset of region ids")
    parser.add_argument("--sigma-alpha", type=float, default=0.01)
    parser.add_argument("--prior-sd", type=float, default=3.0)
    parser.add_argument("--tau-theta", type=float, default=3.0)
    parser.add_argument("--tau-intercept", type=float, default=3.0)
    parser.add_argument("--tau-shared", type=float, default=3.0,
                        help="prior sd of the latent-only model's area-level coefficients")
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
    for m in args.models:
        parse_model(m)
    sets = transfer_sets(data)
    logger.info("table set A: %s", ", ".join(sets["A"]))
    logger.info("table set B: %s", ", ".join(sets["B"]))
    views, features_all = {}, None
    if any(parse_model(m)[0] == "latentview" for m in args.models):
        with stage("latent views of table sets A and B"):
            features_all = decoder_features(args.run, phi, columns, binary, data.weight, "all").F
            r2 = []
            for k in ("A", "B"):
                views[k] = fit_views(data, sets[k], features_all)
                r2.append(views[k].r2.assign(table_set=k))
                logger.info("views of set %s: R² by table %s", k, ", ".join(
                    f"{t} {v:.2f}" for t, v in zip(views[k].r2["table"], views[k].r2["r2"])))
            write_text(args.out / "views_r2.csv",
                       pd.concat(r2, ignore_index=True).to_csv(index=False))
    _SHARED.update(data=data, args=args, regions=regions, features=features, sets=sets,
                   views=views, features_all=features_all)
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
