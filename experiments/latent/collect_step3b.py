"""Pool step 3b and apply its decision rule (``PLAN_latent_step3b.md`` version 1,
§4-§6).

    python experiments/latent/collect_step3b.py --root $PMEDM_VB_RUNS/latent/step3b

Reads ``<root>/<region>/baselines/`` and ``<root>/<region>/<g>/`` written by
``step3b_basis.py`` and writes, under the root, ``summary.txt``,
``decision.json``, ``pooled_models.csv`` and ``pooled_tau.csv``.

**Metrics** (leave-zone-out, means over every (region, zone)): ``tvd``,
``lor_mae`` (scoring tables), ``rare_rms`` (RMS log(pred / truth) over every
(region, zone, rare group)), ``misfit_rms_z`` (fitted S1 cells),
``b1_rms_z`` and ``b2_rms_z`` (transfer misfit on B1, which chooses ``τ``,
and on B2, the transfer score).

**Models:** the baselines ``pmedm_own``, ``pmedm_ref``, ``latent_u0`` (and
``latent_u0:p``) from step 3, transfer re-scored on B1 and B2; per basis
``g``, ``latent_<g>_t<m>`` at each ``τ`` and ``latent_<g>`` at the ``τ``
chosen by B1 per region; ``latent_<g>_ev`` (evidence-chosen) and
``latent_<g>_best`` (truth-best: lowest mean log-odds-ratio error over the
region's zones) are added here, reported only.

**Decision (§6).** A basis qualifies when ``latent_<g>`` is no more than 2%
worse than ``latent_u0`` on ``lor_mae`` and on ``tvd`` and at least 5% better
on ``b2_rms_z``. If any qualifies, the deviations stay with the best such
basis, here the qualifying basis with the lowest ``b2_rms_z`` (a choice made
here: the plan does not define "best"; the transfer gain is what the rule
asks of the deviations); otherwise they are dropped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.progress import write_text

BASELINES = ("pmedm_own", "pmedm_ref", "latent_u0")
WORSE = 0.02     # lor_mae, tvd: at most 2% worse than latent_u0
BETTER = 0.05    # b2_rms_z: at least 5% better
METRICS = ("tvd", "lor_mae", "rare_rms", "misfit_rms_z", "b1_rms_z", "b2_rms_z")


def read(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(p for p in root.glob(f"*/*/{name}.csv"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p, dtype={"zone": str}) for p in paths], ignore_index=True)


def copy_choice(frames: dict, tau: pd.DataFrame, flag: str, suffix: str) -> None:
    """Add ``latent_<g><suffix>`` rows: per region and basis, the τ in ``flag``."""
    pick = tau[tau[flag]][["region", "basis", "tau_multiple"]]
    for name, frame in frames.items():
        if frame.empty or "tau_multiple" not in frame:
            continue
        sel = frame.merge(pick, on=["region", "basis", "tau_multiple"])
        sel = sel[sel["model"].str.fullmatch(r"latent_.+_t[0-9.]+")]
        frames[name] = pd.concat(
            [frame, sel.assign(model="latent_" + sel["basis"] + suffix)], ignore_index=True)


def pool(frames: dict) -> pd.DataFrame:
    keys = "model"
    out = frames["scores"].groupby(keys)[["tvd", "lor_mae"]].mean()
    ok = frames["rare"][(frames["rare"]["truth"] > 0) & (frames["rare"]["pred"] > 0)]
    out = out.join(np.sqrt((np.log(ok["pred"] / ok["truth"]) ** 2).groupby(ok[keys]).mean())
                   .rename("rare_rms"))
    out = out.join(frames["misfit"].groupby(keys)["rms_z"].mean().rename("misfit_rms_z"))
    t = frames["transfer"]
    for k in ("B1", "B2"):
        out = out.join(t[t["scored_set"] == k].groupby(keys)["rms_z"].mean()
                       .rename(f"{k.lower()}_rms_z"))
    return out[list(METRICS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root
    frames = {k: read(root, k) for k in ("scores", "rare", "misfit", "transfer")}
    tau = read(root, "tau")
    if frames["scores"].empty or tau.empty:
        raise SystemExit(f"no results under {root}")
    # truth-best τ per region and basis: lowest mean lor_mae of the realized MAP
    s = frames["scores"]
    s = s[s["model"].str.fullmatch(r"latent_.+_t[0-9.]+")]
    best = s.groupby(["region", "basis", "tau_multiple"])["lor_mae"].mean().reset_index()
    best = best.loc[best.groupby(["region", "basis"])["lor_mae"].idxmin()]
    tau = tau.merge(best.assign(chosen_truth=True)[["region", "basis", "tau_multiple",
                                                    "chosen_truth"]],
                    on=["region", "basis", "tau_multiple"], how="left")
    tau["chosen_truth"] = tau["chosen_truth"].fillna(False).astype(bool)
    copy_choice(frames, tau, "chosen_evidence", "_ev")
    copy_choice(frames, tau, "chosen_truth", "_best")
    table = pool(frames)
    bases = sorted(tau["basis"].unique(), key=lambda b: int(b.replace("kern", "")))

    lines = []
    ref = table.loc["latent_u0"] if "latent_u0" in table.index else None
    qualifying, decision_rows = [], []
    if ref is None:
        lines.append("Decision: incomplete (latent_u0 missing)")
    else:
        lines.append("Decision (section 6): each basis at its B1-chosen tau against latent_u0 "
                     f"(lor_mae and tvd at most +{WORSE:.0%}; B2 transfer at least "
                     f"-{BETTER:.0%})")
        for g in bases:
            name = f"latent_{g}"
            if name not in table.index:
                continue
            row = table.loc[name]
            rel = {m: float(row[m] / ref[m] - 1) for m in ("lor_mae", "tvd", "b2_rms_z")}
            ok = rel["lor_mae"] <= WORSE and rel["tvd"] <= WORSE and rel["b2_rms_z"] <= -BETTER
            if ok:
                qualifying.append((float(row["b2_rms_z"]), g))
            decision_rows.append({"basis": g, **{f"rel_{k}": v for k, v in rel.items()},
                                  "qualifies": ok})
            lines.append(f"  {g:8s} lor_mae {rel['lor_mae']:+.1%}  tvd {rel['tvd']:+.1%}  "
                         f"B2 transfer {rel['b2_rms_z']:+.1%}  -> "
                         f"{'qualifies' if ok else 'does not qualify'}")
        if qualifying:
            chosen = min(qualifying)[1]
            lines.append(f"DECISION: the zone deviations stay, with g = {chosen} (lowest B2 "
                         "transfer misfit among the qualifying bases).")
        else:
            chosen = None
            lines.append("DECISION: the zone deviations are dropped (no basis qualifies); the "
                         "model keeps the type-reference support and the realized-population "
                         "level, without u_b, and step 4 proceeds on that.")
        if "pmedm_ref" in table.index:
            lines.append("  against pmedm_ref (reported): " + "; ".join(
                f"{g} lor_mae {table.loc[f'latent_{g}', 'lor_mae'] / table.loc['pmedm_ref', 'lor_mae'] - 1:+.1%}, "
                f"tvd {table.loc[f'latent_{g}', 'tvd'] / table.loc['pmedm_ref', 'tvd'] - 1:+.1%}, "
                f"B2 {table.loc[f'latent_{g}', 'b2_rms_z'] / table.loc['pmedm_ref', 'b2_rms_z'] - 1:+.1%}"
                for g in bases if f"latent_{g}" in table.index))

    order = [*BASELINES, "latent_u0:p"]
    for g in bases:
        order += [f"latent_{g}", f"latent_{g}_ev", f"latent_{g}_best"]
        order += sorted((m for m in table.index if m.startswith(f"latent_{g}_t") and ":" not in m),
                        key=lambda m: float(m.rsplit("_t", 1)[1]))
    order += sorted(m for m in table.index if m.endswith(":p") and m not in order)
    order += [m for m in table.index if m not in order]
    view = table.reindex([m for m in order if m in table.index])
    choice = pd.DataFrame({
        label: tau[tau[flag]].set_index(["basis", "region"])["tau_multiple"]
        for label, flag in (("b1", "chosen_b1"), ("evidence", "chosen_evidence"),
                            ("truth", "chosen_truth"))})
    agree = pd.DataFrame({
        "b1_eq_truth": (choice["b1"] == choice["truth"]).groupby(level="basis").mean(),
        "evidence_eq_truth": (choice["evidence"] == choice["truth"]).groupby(level="basis").mean()})
    tau_cols = [c for c in ("basis", "region", "tau_multiple", "tau", "tau_ref_g", "K_g",
                            "b1_rms_z", "log_evidence", "chosen_b1", "chosen_evidence",
                            "chosen_truth", "start", "n_starts", "spread", "objective",
                            "objective_at_zero", "grad_max", "evaluations_all_starts",
                            "seconds_per_evaluation", "seconds_tau", "converged") if c in tau]
    tau_refs = json.loads((root / "tau_ref.json").read_text()) if (root / "tau_ref.json").exists() \
        else {}
    settings = json.loads((root / "settings.json").read_text()) \
        if (root / "settings.json").exists() else {}
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 1000):
        text = [f"Step 3b: deviation basis g; {tau['region'].nunique()} regions, bases "
                f"{', '.join(bases)}; leave-zone-out", "", *lines, "",
                "1. Models (means over zones; rare_rms pooled over zones and rare groups)",
                view.round(4).to_string(), "",
                "2. tau per region and basis: chosen by B1, by evidence, by truth (multiples of "
                "tau_ref^g)", choice.to_string(), "",
                "   agreement with truth", agree.round(2).to_string(), "",
                "3. tau_ref^g (spread of the PUMA tilts on g)",
                "\n".join(f"   {g}: {v['tau_ref']:.4f} ({v['converged']}/{v['n_pumas']} "
                          f"converged)" for g, v in tau_refs.items()), "",
                "   B1 / B2 split (by table): " + json.dumps(settings.get("split", {})), "",
                "4. Per tau: B1 misfit, evidence, MAP (start, spread over starts, objective vs "
                "U = 0, gradient, evaluations, seconds)",
                tau[tau_cols].round(4).to_string(index=False), ""]
        done = [json.loads(p.read_text()) for p in sorted(root.glob("*/kern*/done.json"))]
        if done:
            d = pd.DataFrame(done)
            text += ["5. Timings (seconds over all tau per unit)",
                     d.groupby("basis")["seconds_all_taus"].agg(["count", "median", "max", "sum"])
                     .round(0).to_string(),
                     f"   total {d['seconds_all_taus'].sum() / 3600:.2f} CPU-hours (x threads); "
                     f"peak memory {d['peak_gb'].max():.1f} GB", ""]
    write_text(root / "summary.txt", "\n".join(text) + "\n")
    write_text(root / "decision.json", json.dumps({
        "decision": "deviations stay" if qualifying else "deviations dropped",
        "basis": min(qualifying)[1] if qualifying else None, "per_basis": decision_rows},
        indent=1, default=float))
    write_text(root / "pooled_models.csv", view.reset_index().to_csv(index=False))
    write_text(root / "pooled_tau.csv", tau.to_csv(index=False))
    print("\n".join(lines))
    print(f"\nfull summary in {root / 'summary.txt'}")


if __name__ == "__main__":
    main()
