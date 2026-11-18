"""Pool step 2's per-PUMA outputs and apply the §6 decision rules
(``PLAN_latent_step2.md`` version 2).

    python experiments/latent/collect_step2.py --root $PMEDM_VB_RUNS/latent/step2

Reads ``<root>/settings.json`` and ``<root>/<puma>/`` (finished PUMAs only:
those with ``done.json``) as ``step2_engine.py`` writes them. Writes under
``<root>``:

* ``summary.txt``: settings and ``τ_ref``; the §4.1 checks per PUMA; the
  engine's timings and peak memory; the information diagnostic per PUMA
  (quantiles of ``df_b`` at each ``τ``, eigenvalues above ``τ⁻²``, what tract
  rows and the cross-tract coupling add) and per block group; the variance
  removed for the §4.3 functionals; the survey, model-gap (``ω²``) and
  realization split of ``diag S``; and the §6 decisions;
* ``blockgroups.csv``, ``tracts.csv``, ``puma.csv``, ``functionals.csv``,
  ``timings.csv``, ``s_split.csv`` (all PUMAs) and ``decisions.json``.

**Decision rules** (§6, thresholds fixed at approval):

* computability: the slowest full evaluation (log-likelihood and gradient,
  ``log det`` term included) over PUMAs and evaluations; at most
  :data:`SECONDS` -> exact collapsed likelihood in step 3, otherwise ``S`` held
  fixed within each outer iteration;
* identification: the median of ``df_b(τ_ref)`` over the block groups of all
  PUMAs pooled (block groups with ``N_b = 0`` left out: they have no
  population, so no ``u_b`` to identify); at least 5 well identified, from 1
  to 5 weakly identified (step 3 adds the ``u_b = 0`` variant), below 1 not
  identified (stop and review the model).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.progress import write_text

SECONDS = 60.0
DF_WELL, DF_WEAK = 5.0, 1.0
EXPECTED_PUMAS = 5


def read(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(p.parent / f"{name}.csv" for p in root.glob("*/done.json"))
    frames = [pd.read_csv(p, dtype={"puma": str, "block_group": str, "tract": str})
              for p in paths if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fmt(frame: pd.DataFrame, digits: int = 3) -> str:
    return frame.to_string(float_format=lambda x: f"{x:.{digits}g}")


def decide(timings: pd.DataFrame, bgs: pd.DataFrame, n_pumas: int) -> tuple[dict, list[str]]:
    slowest = timings.loc[timings["total_s"].idxmax()]
    exact = bool(slowest["total_s"] <= SECONDS)
    populated = bgs[bgs["N_b"] > 0]
    median = float(populated["df_1x"].median())
    if median >= DF_WELL:
        category = "well identified"
        action = "step 3 as planned"
    elif median >= DF_WEAK:
        category = "weakly identified"
        action = "step 3 proceeds and adds the u_b = 0 variant (blocks share theta_P) as a comparison"
    else:
        category = "not identified"
        action = "blocks collapse to theta_P; stop and review the model before step 3"
    decision = {
        "collapsed_likelihood": "exact" if exact else "fixed_S",
        "slowest_evaluation_seconds": float(slowest["total_s"]),
        "slowest_evaluation_puma": str(slowest["puma"]),
        "threshold_seconds": SECONDS,
        "identification": category,
        "median_df_tau_ref": median,
        "n_block_groups": len(populated),
        "n_block_groups_without_population": int((bgs["N_b"] <= 0).sum()),
        "n_pumas": n_pumas,
        "complete": n_pumas >= EXPECTED_PUMAS,
    }
    lines = [
        f"Computability: slowest full evaluation {decision['slowest_evaluation_seconds']:.1f} s "
        f"(PUMA {decision['slowest_evaluation_puma']}) against {SECONDS:g} s -> "
        + ("EXACT collapsed likelihood in step 3." if exact else
           "S FIXED within each outer iteration of step 3, refreshed between iterations "
           "(an approximation, to be recorded)."),
        f"Identification of u_b: median df_b(tau_ref) = {median:.3g} over "
        f"{len(populated)} populated block groups of {n_pumas} PUMA(s) -> "
        f"{category.upper()}: {action}.",
    ]
    if not decision["complete"]:
        lines.append(f"NOTE: only {n_pumas} of the {EXPECTED_PUMAS} PUMAs are finished; the "
                     "decisions are provisional.")
    return decision, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    settings = json.loads((root / "settings.json").read_text())
    timings, bgs = read(root, "timings"), read(root, "blockgroups")
    tracts, funs, split = read(root, "tracts"), read(root, "functionals"), read(root, "s_split")
    puma_df = read(root, "puma")
    if bgs.empty:
        raise SystemExit(f"no finished PUMAs under {root}")
    pumas = sorted(bgs["puma"].unique())
    areas = {p: json.loads((root / p / "area.json").read_text()) for p in pumas}
    decision, decision_lines = decide(timings, bgs, len(pumas))
    labels = [c[len("df_"):] for c in bgs.columns if c.startswith("df_")]

    out = ["Step 2: collapsed-likelihood engine and information diagnostic",
           "(PLAN_latent_step2.md version 2; MODEL_latent.qmd version 2)", ""]
    out += ["== Settings ==",
            *(f"{k}: {settings[k]}" for k in ("h", "gamma", "sigma_theta", "tables", "alpha",
                                               "taper", "variance_floor", "prior_sd_s", "K",
                                               "M")),
            f"tau_ref = {settings['tau_ref']:.4g} (from {settings['tau_ref_pumas']} PUMAs' "
            f"theta_P); tau grid {', '.join(f'{m:g} x' for m in settings['tau_multiples'])}",
            f"pair operator: {json.dumps(settings['pair_operator'])}",
            *(f"step 1 note: {n}" for n in settings.get("step1_notes", [])), ""]

    area_rows = pd.DataFrame([{
        "puma": p, "tracts": a["n_tracts"], "block_groups": a["n_block_groups"],
        "rows": a["n_rows"], "J_T": a["J_T"], "J_B": a["J_B"], "J0": a["J0"],
        "one_to_one": a["one_to_one"],
        "T_vs_X_own": a.get("own_rows_max_abs_T_minus_X", np.nan),
        "sum_N_b": a["N_b_sum"], "N_puma": a["N_puma"], "ratio": a["N_b_sum_over_N_puma"],
        "gq": a["gq_total"], "bg_N0": a["n_block_groups_N_zero"],
        "N": a["N_puma"], "n": a["n_puma"], "omega": a["omega"],
        "p_vs_qP": a["max_abs_p_minus_stored_qP"]} for p, a in areas.items()]).set_index("puma")
    out += ["== Area inputs (4.1 checks) ==",
            "one_to_one: T's constraint cells map one to one onto build_puma's columns; "
            "T_vs_X_own: max |T - X| on the PUMA's own records; ratio: sum_b N_b / PUMA N "
            "(PUMS weights); p_vs_qP: max |p_b(u=0) - step 1's q^P|; omega = N / (n s), "
            "with N and n the PUMA's units and own records (build_puma).",
            fmt(area_rows, 4), ""]

    tcols = ["build_s", "factor_s", "loglik_s", "gradient_s", "logdet_gradient_s", "total_s",
             "peak_gb"]
    out += ["== Engine timings (seconds; peak memory of the process in GB) ==",
            "gradient = without the log det term; logdet_gradient = the log det term "
            "(including the tract blocks of S^-1).",
            fmt(timings.set_index(["puma", "evaluation"])[
                ["B", "n_tracts", "n_rows", "max_tract_rows", *tcols, "workers"]]), ""]

    q = bgs[bgs["N_b"] > 0].groupby("puma")
    rows = []
    for label in labels:
        stats = q[f"df_{label}"].describe(percentiles=[0.1, 0.5, 0.9])[
            ["count", "min", "10%", "50%", "90%", "max"]]
        stats.columns = [f"df_{c}" for c in stats.columns]
        above = q[f"n_above_{label}"].median().rename("median_n_eig_above_tau^-2")
        rows.append(pd.concat([stats, above], axis=1).assign(tau=label))
    df_table = pd.concat(rows).reset_index().set_index(["tau", "puma"]).sort_index()
    pooled = bgs[bgs["N_b"] > 0]
    pooled_rows = pd.DataFrame([{
        "tau": label, "median_df": pooled[f"df_{label}"].median(),
        "q10": pooled[f"df_{label}"].quantile(0.1), "q90": pooled[f"df_{label}"].quantile(0.9),
        "median_n_eig_above": pooled[f"n_above_{label}"].median()} for label in labels])
    out += ["== Information diagnostic: df_b(tau) = tr[I_bb (I_bb + tau^-2 I)^-1] in [0, K] ==",
            f"K = {settings['K']}; block groups with N_b = 0 left out of the quantiles.",
            fmt(df_table), "", "Pooled over PUMAs:", fmt(pooled_rows.set_index("tau")), ""]
    eig = pooled[[c for c in pooled.columns if c.startswith("eig")] + ["trace_I"]].describe(
        percentiles=[0.1, 0.5, 0.9]).T[["min", "10%", "50%", "90%", "max"]]
    out += ["Leading eigenvalues of I_bb (pooled), against tau^-2 = "
            + ", ".join(f"{label}: {(float(label[:-1]) * settings['tau_ref']) ** -2:.4g}"
                        for label in labels) + ":", fmt(eig), ""]
    tract_cols = [c for c in tracts.columns if c.startswith(("joint_df_per_bg_", "mean_df_"))]
    out += ["What tract rows add: per tract, the joint df over its block groups divided by "
            "their number, beside the mean of the block groups' own df (medians over tracts):",
            fmt(tracts.groupby("puma")[tract_cols].median()), "",
            "What the cross-tract coupling adds: the joint df of the PUMA's full I divided by "
            "its number of block groups, beside the mean of the block groups' own df:",
            fmt(puma_df.set_index("puma")), ""]

    vr = funs[funs["N_b"] > 0].groupby(["kind", "functional", "tau_multiple"])[
        "variance_removed"].describe(percentiles=[0.1, 0.5, 0.9])[["10%", "50%", "90%"]]
    vr = vr.unstack("tau_multiple")
    share = funs.groupby(["kind", "functional"])["share_at_reference"].agg(["min", "max"])
    out += ["== Share of prior variance removed by the tables (quantiles over populated block "
            "groups, all PUMAs) ==",
            "1 - a'(I_bb + tau^-2 I)^-1 a / (tau^2 a'a), a = d(share)/d(u_b) at q^P.",
            fmt(vr), "", "The shares at the reference q^P (range over PUMAs):", fmt(share, 4), ""]

    per_puma = funs[(funs["N_b"] > 0) & (funs["tau_multiple"] == "1x")].groupby(
        ["functional", "puma"])["variance_removed"].median().unstack("puma")
    out += ["Median share removed at tau_ref, by PUMA:", fmt(per_puma), ""]

    allrows = split[split["table"] == "(all)"].set_index(["puma", "level"])
    out += ["== diag S: survey (Sigma_survey) + model gap (omega^2) + realization variance ==",
            "ratio = survey / realization per cell (cells with realization > 0); mean_weight = "
            "the PUMA's N / n.",
            fmt(allrows[["cells", "cells_realization_zero", "survey_mean", "model_gap",
                         "realization_mean", "ratio_of_means", "ratio_median", "ratio_q10",
                         "ratio_q90", "share_survey_larger",
                         "share_model_gap_larger_than_realization", "mean_weight"]]), "",
            "By level and table (median over PUMAs):",
            fmt(split[split["table"] != "(all)"].groupby(["level", "table"])[
                ["survey_mean", "model_gap", "realization_mean", "ratio_median",
                 "share_survey_larger"]].median()), ""]

    out += ["== Decisions (section 6) ==", *decision_lines, ""]
    bg_cols = ["puma", "block_group", "N_b", "households", "gq", "trace_I", "eig1",
               *(f"df_{label}" for label in labels), *(f"n_above_{label}" for label in labels)]
    out += ["== Per block group ==", bgs[bg_cols].to_string(
        index=False, float_format=lambda x: f"{x:.3g}"), ""]

    write_text(root / "summary.txt", "\n".join(out))
    for name, frame in (("blockgroups", bgs), ("tracts", tracts), ("puma", puma_df),
                        ("functionals", funs),
                        ("timings", timings), ("s_split", split)):
        write_text(root / f"{name}.csv", frame.to_csv(index=False))
    write_text(root / "decisions.json", json.dumps(decision, indent=1))
    print("\n".join(decision_lines))
    print(f"written {root / 'summary.txt'}")


if __name__ == "__main__":
    main()
