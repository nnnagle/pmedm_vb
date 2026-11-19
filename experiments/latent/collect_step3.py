"""Pool step 3 across regions and support modes, apply the §6 decision rules,
and write a summary the user can copy (``PLAN_latent_step3.md`` version 1,
§6 and §8).

    python experiments/latent/collect_step3.py --root $PMEDM_VB_RUNS/latent/step3

Reads every ``<root>/<region>/<mode>/`` unit of ``step3_regional.py`` and each
region's ``reference.json`` and ``reference_check.csv``, and writes under
``--out`` (default: the root) ``summary.txt``, ``decision.json``,
``pooled_ladder.csv``, ``pooled_regions.csv``, ``pooled_rare.csv`` and
``pooled_tau.csv``.

**Metrics**, per model and mode, over every (region, zone):

* ``tvd``, ``lor_mae``: means over (region, zone, scoring table) of the
  scoring tables' TVD and log-odds-ratio error (mean absolute; tables
  without two-way structure have no ``lor_mae``);
* ``rare_rms``: the rare groups' spread, the RMS of ``log(pred / truth)``
  over every (region, zone, rare group) of
  :data:`~pmedm_vb.embed.puma_check.RARE` where both are positive (pooled
  over the rare groups; ``rare_rms_step1`` the same over step 1's groups,
  reported beside it);
* ``misfit_rms_z``: misfit to the fitted S1 cells in SE units (mean over
  zones of the per-zone RMS z);
* ``transfer_rms_z``: misfit to table set B's cells, fitted on S1 (mean over
  zones);
* ``allocation``: allocation accuracy (``own`` mode only; mean over regions).

**Models.** As written by ``step3_regional.py``, plus ``latent_best``: per
region and mode, the ``latent_t<m>`` with the lowest mean log-odds-ratio
error over its zones (the best ``τ`` by truth, reported beside the
evidence-chosen ``latent`` as a check on the rule; ``latent_best:p`` the
same ``τ``'s ``p_b``).

**Decision (§6).** Primary: ``latent`` against ``pmedm_own`` in ``lzo``
mode. Go to step 4 if ``latent`` is at most 5% worse on ``lor_mae`` and on
``tvd`` and at most 10% worse on ``rare_rms``. An improvement is reported for
any of those, or ``transfer_rms_z``, where ``latent`` is at least 5% better.
Otherwise: stop and review, with the ladder's steps (support, weight, zone
deviations) shown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.latent.reference_check import RARE_STEP1
from pmedm_vb.progress import write_text

LADDER = ("pmedm_own", "pmedm_ref", "latent_u0", "latent", "latent_n", "latent_best")
METRICS = ("tvd", "lor_mae", "rare_rms", "rare_rms_step1", "misfit_rms_z", "transfer_rms_z",
           "allocation")
#: §6 thresholds: relative excess allowed (worse), and the improvement margin.
GO = {"lor_mae": 0.05, "tvd": 0.05, "rare_rms": 0.10}
IMPROVEMENT = 0.05
IMPROVEMENT_METRICS = ("lor_mae", "tvd", "rare_rms", "transfer_rms_z")


def read(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"*/*/{name}.csv"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p, dtype={"zone": str}) for p in paths], ignore_index=True)


def add_best(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add ``latent_best`` (and ``:p``) rows: per (region, mode), the τ with the
    lowest mean log-odds-ratio error. Returns the choice per (region, mode)."""
    scores = frames["scores"]
    tau_rows = scores[scores["model"].str.fullmatch(r"latent_t[0-9.]+")]
    by = tau_rows.groupby(["region", "mode", "model"])["lor_mae"].mean().reset_index()
    best = by.loc[by.groupby(["region", "mode"])["lor_mae"].idxmin()]
    best = best.rename(columns={"model": "best_model"})[["region", "mode", "best_model"]]
    for name, frame in frames.items():
        if frame.empty or "model" not in frame:
            continue
        extra = []
        for suffix in ("", ":p"):
            merged = frame.merge(best, on=["region", "mode"])
            sel = merged[merged["model"] == merged["best_model"] + suffix]
            extra.append(sel.drop(columns="best_model").assign(model="latent_best" + suffix))
        frames[name] = pd.concat([frame, *extra], ignore_index=True)
    best["best_tau_multiple"] = best["best_model"].str.replace("latent_t", "").astype(float)
    return best


def ladder(frames: dict[str, pd.DataFrame], keys=("mode", "model")) -> pd.DataFrame:
    keys = list(keys)
    scores, misfit = frames["scores"], frames["misfit"]
    out = scores.groupby(keys)[["tvd", "lor_mae"]].mean()
    out = out.join(misfit.groupby(keys)["rms_z"].mean().rename("misfit_rms_z"))
    out = out.join(frames["transfer"].groupby(keys)["rms_z"].mean().rename("transfer_rms_z"))
    out = out.join(frames["summary"].groupby(keys)["allocation_accuracy"].mean()
                   .rename("allocation"))
    rare = frames["rare"]
    for name, groups in (("rare_rms", None), ("rare_rms_step1", RARE_STEP1)):
        frame = rare
        if groups is not None:
            frame = rare[rare.set_index(["table", "category"]).index.isin(list(groups))]
        ok = frame[(frame["truth"] > 0) & (frame["pred"] > 0)]
        lr2 = np.log(ok["pred"] / ok["truth"]) ** 2
        out = out.join(np.sqrt(lr2.groupby([ok[k] for k in keys]).mean()).rename(name))
    return out[list(METRICS)]


def order_models(models) -> list[str]:
    models = list(dict.fromkeys(models))
    first = [m for m in LADDER if m in models]
    taus = sorted((m for m in models if m.startswith("latent_t") and ":" not in m),
                  key=lambda m: float(m[len("latent_t"):]))
    rest_p = [m for m in models if m.endswith(":p")]
    rest_p = sorted(rest_p, key=lambda m: (m.startswith("latent_t"), m))
    return first + taus + rest_p + [m for m in models if m not in first + taus + rest_p]


def decide(lad: pd.DataFrame) -> tuple[dict, list[str]]:
    lines = []
    try:
        ref = lad.loc[("lzo", "pmedm_own")]
        lat = lad.loc[("lzo", "latent")]
    except KeyError:
        return {"decision": "incomplete: no lzo latent and pmedm_own"}, \
            ["Decision: incomplete (lzo mode's latent or pmedm_own missing)"]
    rel = {m: float(lat[m] / ref[m] - 1.0) for m in (*GO, "transfer_rms_z")
           if np.isfinite(ref[m]) and ref[m] != 0}
    passes = {m: rel.get(m, np.inf) <= limit for m, limit in GO.items()}
    improved = [m for m in IMPROVEMENT_METRICS if rel.get(m, 0.0) <= -IMPROVEMENT]
    go = all(passes.values())
    lines.append("Primary comparison (section 6): latent (evidence-chosen tau) against pmedm_own, "
                 "leave-zone-out, all zones of all regions")
    for m in (*GO, "transfer_rms_z"):
        if m in rel:
            limit = f"limit +{GO[m]:.0%}" if m in GO else "reported"
            verdict = ("passes" if passes[m] else "FAILS") if m in GO else ""
            lines.append(f"  {m:15s} pmedm_own {ref[m]:.4f}  latent {lat[m]:.4f}  "
                         f"relative {rel[m]:+.1%}  ({limit}) {verdict}")
    lines.append("  improvement (>= 5% better): " + (", ".join(improved) or "none"))
    if go:
        lines.append("DECISION: go to step 4.")
    else:
        lines.append("DECISION: stop and review (section 6). The ladder locates the loss:")
        steps = (("support", "pmedm_own", "pmedm_ref"), ("weight", "pmedm_ref", "latent_u0"),
                 ("zone deviations", "latent_u0", "latent"))
        for name, a, b in steps:
            if ("lzo", a) in lad.index and ("lzo", b) in lad.index:
                parts = [f"{m} {lad.loc[('lzo', b), m] / lad.loc[('lzo', a), m] - 1:+.1%}"
                         for m in GO if lad.loc[("lzo", a), m]]
                lines.append(f"  {name:16s} {a} -> {b}: " + ", ".join(parts))
    decision = {"decision": "go to step 4" if go else "stop and review",
                "relative": rel, "passes": passes, "improvement": improved,
                "pmedm_own": {m: float(ref[m]) for m in METRICS},
                "latent": {m: float(lat[m]) for m in METRICS}}
    return decision, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root, out = args.root, args.out or args.root
    frames = {k: read(root, k) for k in ("scores", "rare", "misfit", "spread", "summary",
                                         "transfer", "fits", "tau")}
    if frames["scores"].empty:
        raise SystemExit(f"no results under {root}")
    best = add_best(frames)
    lad = ladder(frames)
    models = order_models(lad.index.get_level_values("model"))
    modes = [m for m in ("lzo", "own") if m in lad.index.get_level_values("mode")]
    decision, decision_lines = decide(lad)

    by_region = ladder(frames, keys=("mode", "region", "model"))
    tau = frames["tau"].merge(best, on=["region", "mode"], how="left")
    tau_view = tau[["mode", "region", "tau_multiple", "log_evidence", "chosen", "objective",
                    "objective_at_zero", "grad_max", "iterations", "evaluations",
                    "seconds_per_evaluation", "seconds", "converged"]]
    choice = tau[tau["chosen"]][["mode", "region", "tau_multiple"]].rename(
        columns={"tau_multiple": "chosen_by_evidence"}).merge(
        best[["mode", "region", "best_tau_multiple"]], on=["mode", "region"])
    refs = [json.loads(p.read_text()) for p in sorted(root.glob("*/reference.json"))]
    checks = pd.concat([pd.read_csv(p) for p in sorted(root.glob("*/reference_check.csv"))],
                       ignore_index=True) if refs else pd.DataFrame()
    fits = frames["fits"]
    timings = [json.loads(p.read_text()) for p in sorted(root.glob("*/*/timings.json"))]

    rare = frames["rare"]
    ok = rare[(rare["truth"] > 0) & (rare["pred"] > 0)].copy()
    ok["lr2"] = np.log(ok["pred"] / ok["truth"]) ** 2
    rare_tab = ok.groupby(["mode", "table", "category", "model"])["lr2"].mean().pipe(np.sqrt)
    pooled = rare.groupby(["mode", "table", "category", "model"])[["truth", "pred"]].sum()
    pooled["ratio"] = pooled["pred"] / pooled["truth"]

    with pd.option_context("display.width", 250, "display.max_columns", 60,
                           "display.max_rows", 1000):
        n_regions = frames["scores"]["region"].nunique()
        n_zones = frames["scores"].groupby("region")["zone"].nunique().sum()
        text = [f"Step 3 regional run: {n_regions} regions, {n_zones} zones; modes "
                f"{', '.join(modes)}", ""]
        text += decision_lines + [""]
        for mode in modes:
            sub = lad.loc[mode].reindex([m for m in models if m in lad.loc[mode].index])
            text += [f"1{'ab'[modes.index(mode)]}. Ladder, mode {mode} (means over zones; "
                     "rare_rms pooled over zones and rare groups)", sub.round(4).to_string(), ""]
        main_models = [m for m in LADDER if m in models]
        for metric in ("lor_mae", "tvd", "rare_rms", "transfer_rms_z"):
            for mode in modes:
                tab = by_region.loc[mode][metric].unstack("model")
                tab = tab[[m for m in main_models if m in tab]]
                text += [f"2. By region: {metric}, mode {mode}", tab.round(4).to_string(), ""]
        text += ["3. tau: chosen by Laplace evidence against the best by truth (lowest mean "
                 "lor_mae)", choice.to_string(index=False), "",
                 "   agreement: " + (f"{(choice['chosen_by_evidence'] == choice['best_tau_multiple']).mean():.0%}"
                                     if len(choice) else "n/a"), "",
                 "   per tau: MAP objective F against F(U = 0), largest gradient component, "
                 "L-BFGS iterations and evaluations, seconds",
                 tau_view.round(4).to_string(index=False), ""]
        if refs:
            text += ["4. Region references (theta_A on all the region's records)",
                     pd.DataFrame(refs)[["region", "n_A", "N_A", "iterations", "converged",
                                         "grad_max", "ess", "share_outside",
                                         "max_moment_error"]].round(4).to_string(index=False),
                     "", "   reproduction check: q^A against the region's own records, pooled "
                     "over regions (pred / truth); AIAN and NH/PI within [0.8, 1.25] per region",
                     ]
            pc = checks.groupby(["table", "category"])[["truth", "pred"]].sum()
            pc["pooled_ratio"] = pc["pred"] / pc["truth"]
            pc = pc.join(checks.groupby(["table", "category"])["ratio"].agg(["min", "max"]))
            text += [pc.round(3).to_string(), ""]
            bad = checks[checks["stop_group"] & (checks["in_stop_range"] == 0)]
            text += ["   regions with AIAN or NH/PI outside [0.8, 1.25]: " +
                     (", ".join(f"{r.region} {r.category} {r.ratio:.2f}" for r in bad.itertuples())
                      or "none"), ""]
        cond = fits[fits["kind"] == "conditional"]
        text += ["5. Solver checks",
                 f"   conditional PMEDM: {len(cond)} solves, converged {cond['converged'].mean():.0%}, "
                 f"max iterations {cond['iterations'].max()}, max relative duality gap "
                 f"{cond['relative_gap'].max():.2e}, max KKT residual (relative to max |e|) "
                 f"{cond['kkt_relative'].max():.2e}",
                 f"   standard PMEDM: {int((fits['kind'] == 'pmedm').sum())} solves, converged "
                 f"{fits.loc[fits['kind'] == 'pmedm', 'converged'].mean():.0%}",
                 f"   MAP: {len(frames['tau'])} fits, converged {frames['tau']['converged'].mean():.0%}, "
                 f"F <= F(0) in {frames['tau']['objective_not_above_zero'].mean():.0%}, largest "
                 f"gradient component {frames['tau']['grad_max'].max():.2e}", ""]
        if timings:
            t = pd.DataFrame(timings)
            evals = frames["tau"]["evaluations"]
            text += ["6. Timings",
                     f"   units {len(t)}; seconds per unit: median {t['seconds'].median():.0f}, "
                     f"max {t['seconds'].max():.0f}, total {t['seconds'].sum():.0f}; "
                     f"peak memory per process {t['peak_gb'].max():.1f} GB",
                     f"   MAP evaluations per tau: median {evals.median():.0f}, max {evals.max()}; "
                     f"seconds per evaluation: median "
                     f"{frames['tau']['seconds_per_evaluation'].median():.2f}",
                     t[["region", "mode", "Z", "rows", "seconds", "peak_gb", "threads"]]
                     .round(1).to_string(index=False), ""]
        text += ["7. Rare groups: RMS log(pred / truth) over zones, by group",
                 rare_tab.unstack("model")[[m for m in models if m in rare_tab.index
                                            .get_level_values("model")]]
                 .round(3).to_string(), "",
                 "8. Rare groups: pooled prediction / truth",
                 pooled["ratio"].unstack("model")[[m for m in models if m in pooled.index
                                                   .get_level_values("model")]]
                 .round(3).to_string(), ""]
    write_text(out / "summary.txt", "\n".join(text) + "\n")
    write_text(out / "decision.json", json.dumps(decision, indent=1, default=float))
    write_text(out / "pooled_ladder.csv", lad.reset_index().to_csv(index=False))
    write_text(out / "pooled_regions.csv", by_region.reset_index().to_csv(index=False))
    write_text(out / "pooled_rare.csv", pooled.join(rare_tab.rename("rms_log_ratio"))
               .reset_index().to_csv(index=False))
    write_text(out / "pooled_tau.csv", tau.to_csv(index=False))
    print("\n".join(decision_lines))
    print(f"\nfull summary in {out / 'summary.txt'}")


if __name__ == "__main__":
    main()
