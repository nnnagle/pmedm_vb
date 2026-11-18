"""Step 2 of the latent-population PMEDM: the collapsed-likelihood engine and the
information diagnostic (``PLAN_latent_step2.md`` version 2; model
``MODEL_latent.qmd`` version 2).

    python experiments/latent/step2_engine.py --config experiments/embed/stage1_v2.json \
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d16_g0.5_r0.01 \
        --step1 $PMEDM_VB_RUNS/latent/step1_sigma --out $PMEDM_VB_RUNS/latent/step2
    python experiments/latent/collect_step2.py --root $PMEDM_VB_RUNS/latent/step2

**Inputs** (from step 1's run root, ``--step1``): ``final/settings.json`` (h,
γ, σ_θ of the final fits; checked against ``decisions.json``),
``final/reference.npz`` (``θ̂_P`` and ``q^P``), ``inputs/T/``,
``inputs/bases/<h>.*`` and ``inputs/distances.csv``; the constraint table set
is step 1's (``tables`` in its grid file, ``all`` by default). The atoms are
rebuilt as in step 1 (:func:`~pmedm_vb.latent.atoms.load_atoms`, positions
from ``--run``) and must be step 1's, in order.

**Per PUMA** (``--puma``; default 4701502, then the four test PUMAs), written
under ``<out>/<puma>/``:

* ``area.json``: the §4.1 checks (column mapping one to one, ``T`` against
  ``X`` on the PUMA's own records, the tracts' layouts partitioning the rows,
  ``Σ_b N_b`` against the PUMA's ``N``), ``ω`` with the ``N`` and ``n`` it
  uses, and the reference check (``p_b`` at ``u = 0`` against step 1's
  stored ``q^P``);
* ``timings.csv``: ``--evals`` evaluations of the log-likelihood and its
  gradient, the first at the reference (``θ_P = θ̂_P``, ``u = 0``), the
  others at ``u_b ~ N(0, (0.1 τ_ref)² I)``; seconds for building ``p``, ``m``
  and ``S``, factorization, log-likelihood, gradient without the ``log det``
  term, the ``log det`` term and in total, and the process's peak memory;
* ``blockgroups.csv``: per block group, ``N_b`` (households and
  group-quarters persons), the trace and five leading eigenvalues of
  ``I_bb``, and at each ``τ ∈ {0.5, 1, 2} × τ_ref`` the df and the number of
  eigenvalues above ``τ⁻²``;
* ``tracts.csv``: per tract, the joint df of its block of ``I`` divided by
  its number of block groups, beside the mean of its block groups' df;
* ``puma.csv``: the joint df of the full ``I`` divided by the number of block
  groups, beside the mean of the block groups' df (what the cross-tract
  coupling adds);
* ``functionals.csv``: per block group, functional and ``τ``, the share of
  prior variance removed by the tables (:data:`FUNCTIONALS`);
* ``s_split.csv``: ``diag S`` split into survey (``Σ_survey``), model-gap
  (``ω²``) and realization variance, by level and table, with the PUMA's mean
  weight ``N / n``;
* ``done.json`` (a PUMA with it is skipped, so a job can be resubmitted).

``<out>/settings.json`` holds ``τ_ref`` (step 1's spread of ``θ̂_P``), the
settings and the pair operator's size and build time.

Runs offline: ``PMEDM_VB_OFFLINE=1`` is set if the caller has not set it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.data.cache import OFFLINE_ENV
from pmedm_vb.data.geography import puma_crosswalk_whole
from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.puma_check import B17010_RACE_ALONE, load_check_data
from pmedm_vb.latent.area import area_inputs, load_basis, load_step1
from pmedm_vb.latent.atoms import Contributions, load_atoms
from pmedm_vb.latent.collapsed import CollapsedLikelihood, PairOperator, default_workers
from pmedm_vb.latent.geography import atom_distance
from pmedm_vb.latent.information import (
    block_covariances,
    columns_of,
    degrees_of_freedom,
    eigenvalues,
    joint_eigenvalues,
    puma_information,
    share_gradient,
    variance_removed,
)
from pmedm_vb.latent.reference import log_base
from pmedm_vb.progress import logger, record_run, stage, write_text

PUMAS = ("4701502", "4703000", "4702300", "4702900", "4701900")

#: Timed parts of one evaluation (:class:`~pmedm_vb.latent.collapsed.Evaluation`).
TIMINGS = ("build", "factor", "loglik", "gradient", "logdet_gradient", "total")

#: τ grid as multiples of τ_ref (§3).
TAU_MULTIPLES = (0.5, 1.0, 2.0)

#: Functionals of §4.3: (name, kind, table, numerator categories); the
#: denominator is the table's whole universe (every category summed).
#: ``B17010A-G`` stacks the families by the householder's race alone, so its
#: universe is all families; "households below poverty" is read as families.
FUNCTIONALS = (
    ("nh_aian_share_of_persons", "rare", "B03002", ("nh_aian",)),
    ("nh_nhpi_share_of_persons", "rare", "B03002", ("nh_nhpi",)),
    ("nh_asian_share_of_persons", "rare", "B03002", ("nh_asian",)),
    ("institutional_gq_share_of_units", "rare", "unit_type", ("institutional",)),
    ("noninstitutional_gq_share_of_units", "rare", "unit_type", ("noninstitutional",)),
    ("owner_share_of_households", "common", "B25003", ("owner",)),
    ("age_65_plus_share_of_persons", "common", "B01001", "65+"),
    ("families_below_poverty_share", "common", "B17010A-G", "below_poverty"),
)


def lower_age(category: str) -> int:
    """Lower age bound of a ``B01001`` category name."""
    if re.search(r"_lt_\d+$", category):
        return 0
    found = re.search(r"_(?:ge_)?(\d+)(?:_to_\d+)?$", category)
    if not found:
        raise ValueError(f"no age in {category!r}")
    return int(found.group(1))


def functional_columns(data) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    """``{name: (kind, g, n)}``: numerator and denominator counts per atom."""
    out = {}
    for name, kind, table, chosen in FUNCTIONALS:
        counts = data.counts[table].toarray().astype(float)
        if table == "B17010A-G":
            categories = [c.name for t in B17010_RACE_ALONE for c in data.tables[t].categories]
        else:
            categories = [c.name for c in data.tables[table].categories]
        if chosen == "65+":
            pick = [k for k, c in enumerate(categories) if lower_age(c) >= 65]
        elif isinstance(chosen, str):
            pick = [k for k, c in enumerate(categories) if c == chosen]
        else:
            pick = [categories.index(c) for c in chosen]
        if not pick:
            raise ValueError(f"{name}: no categories chosen in {table}")
        out[name] = (kind, counts[:, pick].sum(axis=1), counts.sum(axis=1))
    return out


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def s_split(A, ev) -> pd.DataFrame:
    """``diag S`` = survey + model gap (``ω²``) + realization variance, by
    level and table."""
    omega2 = A.sigma.prior_var
    rows = A.row_labels().assign(survey=A.sigma.diagonal() - omega2,
                                 realization=ev.realization_diag)
    out = []
    for keys, frame in [*rows.groupby(["level", "table"], sort=False),
                        *((k, f.assign(table="(all)")) for k, f in
                          rows.groupby("level", sort=False))]:
        level, table = (keys, "(all)") if isinstance(keys, str) else keys
        positive = frame["realization"] > 0
        ratio = frame.loc[positive, "survey"] / frame.loc[positive, "realization"]
        out.append({
            "level": level, "table": table, "cells": len(frame),
            "cells_realization_zero": int((~positive).sum()),
            "survey_mean": frame["survey"].mean(), "model_gap": omega2,
            "realization_mean": frame["realization"].mean(),
            "ratio_of_means": frame["survey"].mean() / frame["realization"].mean()
            if frame["realization"].mean() > 0 else np.nan,
            "ratio_median": ratio.median() if len(ratio) else np.nan,
            "ratio_q10": ratio.quantile(0.1) if len(ratio) else np.nan,
            "ratio_q90": ratio.quantile(0.9) if len(ratio) else np.nan,
            "share_survey_larger": float((frame["survey"] > frame["realization"]).mean()),
            "share_model_gap_larger_than_realization":
                float((omega2 > frame["realization"]).mean()),
        })
    return pd.DataFrame(out)


def run_puma(puma: str, s: dict, args) -> None:
    out = args.out / puma
    if (out / "done.json").exists():
        logger.info("PUMA %s already done", puma)
        return
    out.mkdir(parents=True, exist_ok=True)
    step1, atoms, H = s["step1"], s["atoms"], s["H"]
    start = time.perf_counter()
    with stage(f"PUMA {puma}: area inputs"):
        A = area_inputs(s["area"], puma, s["tables"], s["contributions"], alpha=args.alpha,
                        variance_floor=args.variance_floor, prior_sd=args.prior_sd,
                        zones=s["zones"])
    if s.get("pairs") is None or not np.array_equal(s["base_columns"], A.columns.base_columns):
        with stage("pair operator") as step:
            s["pairs"] = PairOperator(s["contributions"].T[:, A.columns.base_columns],
                                      workers=args.workers)
            s["base_columns"] = A.columns.base_columns
            step.detail = json.dumps(s["pairs"].summary())
    pairs = s["pairs"]
    logb = log_base(atoms.q, atom_distance(step1.distances, puma, atoms.puma), step1.gamma)
    theta = step1.theta_of(puma)
    engine = CollapsedLikelihood(A, H, logb, pairs)
    tau_ref = s["tau_ref"]
    rng = np.random.default_rng(args.seed)
    timing_rows, ev0 = [], None
    for k in range(args.evals):
        label = "reference" if k == 0 else f"perturbed_{k}"
        U = None if k == 0 else rng.normal(0.0, 0.1 * tau_ref, size=(A.B, engine.K))
        with stage(f"PUMA {puma}: evaluation {label}") as step:
            ev = engine.evaluate(theta, U)
            step.detail = f"{ev.timings['total']:.1f}s"
        timing_rows.append({
            "puma": puma, "evaluation": label, "B": A.B, "n_tracts": A.n_tracts,
            "n_rows": A.n_rows, "max_tract_rows": max(t["rows"].size for t in engine.tracts),
            **{f"{key}_s": ev.timings[key] for key in TIMINGS},
            "peak_gb": peak_gb(), "loglik": ev.loglik, "quad": ev.quad, "logdet": ev.logdet,
            "grad_theta_norm": float(np.linalg.norm(ev.grad_theta)),
            "grad_U_norm": float(np.linalg.norm(ev.grad_U)), "workers": args.workers,
            "threads": os.environ.get("OMP_NUM_THREADS")})
        if k == 0:
            ev0 = ev
    write_text(out / "timings.csv", pd.DataFrame(timing_rows).to_csv(index=False))

    q_ref = step1.q_of(puma)
    p0 = ev0.p[:, 0]
    checks = {**A.checks, "max_abs_p_minus_stored_qP": float(np.abs(p0 - q_ref).max()),
              "max_stored_qP": float(q_ref.max()), "tau_ref": tau_ref,
              "mean_weight_N_over_n": A.N_puma / A.n_puma if A.n_puma else float("nan")}
    write_text(out / "area.json", json.dumps(checks, indent=1, default=str))

    with stage(f"PUMA {puma}: information diagnostic"):
        C = block_covariances(engine, ev0, engine.thetas(theta, None))
        I_full = puma_information(engine, ev0, C)
        K = engine.K
        info = []
        for t in range(A.n_tracts):
            cols = columns_of(A.tract_members(t), K)
            info.append(I_full[np.ix_(cols, cols)])
        grads = {name: (kind, *share_gradient(g, n, p0, H))
                 for name, (kind, g, n) in s["functionals"].items()}
        taus = {f"{m:g}x": m * tau_ref for m in TAU_MULTIPLES}
        bg_rows, tract_rows, fun_rows = [], [], []
        for t, I_t in enumerate(info):
            members = A.tract_members(t)
            dfs = {label: [] for label in taus}
            for j, b in enumerate(members):
                I_bb = I_t[j * K:(j + 1) * K, j * K:(j + 1) * K]
                values, vectors = eigenvalues(I_bb)
                row = {"puma": puma, "block_group": A.block_groups[b], "tract": A.tracts[t],
                       "N_b": A.N_b[b], "households": A.households[b], "gq": A.gq[b],
                       "trace_I": float(values.sum()),
                       **{f"eig{e + 1}": float(values[e]) for e in range(min(5, K))}}
                for label, tau in taus.items():
                    df = degrees_of_freedom(values, tau)
                    row[f"df_{label}"] = df
                    row[f"n_above_{label}"] = int((values > tau ** -2).sum())
                    if A.N_b[b] > 0:
                        dfs[label].append(df)
                    for name, (kind, a, f) in grads.items():
                        fun_rows.append({"puma": puma, "block_group": A.block_groups[b],
                                         "N_b": A.N_b[b], "functional": name, "kind": kind,
                                         "share_at_reference": f, "tau_multiple": label,
                                         "tau": tau,
                                         "variance_removed": variance_removed(values, vectors,
                                                                              a, tau)})
                bg_rows.append(row)
            joint, _ = eigenvalues(I_t)
            row = {"puma": puma, "tract": A.tracts[t], "n_bg": members.size,
                   "n_bg_populated": int((A.N_b[members] > 0).sum())}
            for label, tau in taus.items():
                row[f"joint_df_per_bg_{label}"] = degrees_of_freedom(joint, tau) / members.size
                row[f"mean_df_{label}"] = float(np.mean(dfs[label])) if dfs[label] else np.nan
            tract_rows.append(row)
        joint = joint_eigenvalues(I_full)
        del I_full
        populated = A.N_b > 0
        puma_row = {"puma": puma, "B": A.B, "n_bg_populated": int(populated.sum()),
                    "BK": A.B * K}
        for label, tau in taus.items():
            df_b = np.array([r[f"df_{label}"] for r in bg_rows])
            puma_row[f"joint_df_per_bg_{label}"] = degrees_of_freedom(joint, tau) / A.B
            puma_row[f"mean_df_{label}"] = float(df_b[populated].mean()) if populated.any() \
                else np.nan
    write_text(out / "puma.csv", pd.DataFrame([puma_row]).to_csv(index=False))
    write_text(out / "blockgroups.csv", pd.DataFrame(bg_rows).to_csv(index=False))
    write_text(out / "tracts.csv", pd.DataFrame(tract_rows).to_csv(index=False))
    write_text(out / "functionals.csv", pd.DataFrame(fun_rows).to_csv(index=False))
    write_text(out / "s_split.csv", s_split(A, ev0).assign(
        puma=puma, mean_weight=checks["mean_weight_N_over_n"]).to_csv(index=False))
    write_text(out / "done.json", json.dumps({
        "seconds": time.perf_counter() - start, "peak_gb": peak_gb(),
        "B": A.B, "n_tracts": A.n_tracts}))
    logger.info("PUMA %s done in %.0fs", puma, time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="stage 1 run directory (z)")
    parser.add_argument("--step1", required=True, type=Path, help="step 1 run root")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--puma", nargs="+", default=list(PUMAS))
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Sigma_survey shrinkage (the standard PMEDM's)")
    parser.add_argument("--variance-floor", default="zero",
                        help="PMEDMInputs.sigma's variance_floor ('zero', 'none' or a number)")
    parser.add_argument("--prior-sd", type=float, default=3.0,
                        help="s in omega = N / (n s), the model-gap term")
    parser.add_argument("--evals", type=int, default=2,
                        help="engine evaluations per PUMA (the first at u = 0)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help="threads for the sparse products")
    args = parser.parse_args()
    if args.variance_floor not in (None, "zero"):
        args.variance_floor = None if args.variance_floor == "none" else float(args.variance_floor)
    os.environ.setdefault(OFFLINE_ENV, "1")
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)

    config = load_config(args.config)
    area = config.area
    step1 = load_step1(args.step1)
    for note in step1.notes:
        logger.warning("%s", note)
    with stage("load PUMS, tables and replicate weights"):
        data = load_check_data(config)
    with stage("atoms"):
        atoms = load_atoms(data, args.run)
    if not pd.Index(atoms.serial.astype(str)).equals(pd.Index(step1.serial)):
        raise SystemExit("the atoms are not step 1's (SERIALNO order differs)")
    contributions = Contributions.load(step1.root / "inputs" / "T")
    if contributions.serial is None or not contributions.serial.equals(step1.serial):
        raise SystemExit("step 1's T rows are not the atoms")
    with stage(f"basis {step1.h}") as step:
        H = load_basis(step1.root / "inputs" / "bases", step1.h, atoms, args.run)
        step.detail = f"K = {H.shape[1]}"
    if H.shape[1] != len(step1.columns):
        raise SystemExit(f"basis has {H.shape[1]} columns, the reference {len(step1.columns)}")
    from pmedm_vb.assemble.allcells import constraint_tables

    s = {"area": area, "step1": step1, "atoms": atoms, "H": H, "contributions": contributions,
         "tables": constraint_tables(area, step1.tables), "tau_ref": step1.tau_ref(),
         "functionals": functional_columns(data)}
    del data
    with stage("PUMA crosswalk"):
        s["zones"] = puma_crosswalk_whole(area)
    logger.info("tau_ref = %.4g from %d PUMAs' theta", s["tau_ref"], len(step1.pumas))
    unknown = sorted(set(args.puma) - set(step1.pumas))
    if unknown:
        raise SystemExit(f"PUMAs {unknown} have no step 1 reference")
    for puma in args.puma:
        run_puma(str(puma), s, args)
        if s.get("pairs") is not None:
            write_text(args.out / "settings.json", json.dumps({
                "h": step1.h, "gamma": step1.gamma, "sigma_theta": step1.sigma,
                "tables": step1.tables, "alpha": args.alpha, "taper": None,
                "variance_floor": args.variance_floor, "prior_sd_s": args.prior_sd,
                "tau_ref": s["tau_ref"],
                "tau_ref_pumas": len(step1.pumas), "tau_multiples": list(TAU_MULTIPLES),
                "K": int(H.shape[1]), "M": int(H.shape[0]), "step1": str(step1.root),
                "run": str(args.run), "step1_notes": step1.notes,
                "pair_operator": s["pairs"].summary()}, indent=1, default=str))


if __name__ == "__main__":
    main()
