"""Pool step 4's calibration and apply its decision rule (``PLAN_latent_step4.md``
version 1, §4.4, §6 and §8).

    python experiments/latent/collect_step4.py --root $PMEDM_VB_RUNS/latent/step4 \
        --step3b $PMEDM_VB_RUNS/latent/step3b

Reads every ``<root>/<simulation>/<region>/r<k>/`` unit of
``step4_posterior.py`` and writes, under the root, ``summary.txt``,
``decision.json``, ``calibration.csv`` (by simulation and functional
class), ``calibration_tables.csv`` (also by table), ``pit.csv`` and
``map_accuracy.csv``.

* **Calibration** (:func:`~pmedm_vb.latent.calibration.summarize`), pooled
  over replicates, zones and cells, by simulation and functional class:
  coverage of the 50% and 90% intervals, RMS of the standardized errors,
  median 90% width (over twice the published SE for fitted cells, over the
  truth for the others), the share of 90% intervals with a lower end below
  zero, the direct estimate's coverage (fitted cells), and the spread of
  coverage over replicates; PIT histograms.
* **Decision (§6)** (:func:`~pmedm_vb.latent.calibration.decide`): S
  primary, G diagnostic; bands [0.85, 0.95] at 90%, [0.42, 0.58] at 50%;
  rare groups judged separately.
* **MAP accuracy under noise:** the realized MAP at the B1-chosen ``τ``
  (``latent_kern30``), means over zones of LOR error, TVD, rare-group spread,
  fitted-cell misfit and transfer misfit on B1 and B2, by simulation, beside
  step 3b's noise-free ``latent_kern30`` (``--step3b``'s ``pooled_models.csv``).
* ``τ``: the distribution of the B1-chosen multiple, and the share at the
  grid's lower edge (reported, not acted on).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.latent.calibration import decide, pit_histogram, summarize
from pmedm_vb.progress import write_text


def read(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"*/*/r*/{name}"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p, dtype={"zone": str}) for p in paths], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--step3b", type=Path, default=None)
    args = parser.parse_args()
    root = args.root
    post = read(root, "posterior.csv.gz")
    if post.empty:
        raise SystemExit(f"no results under {root}")
    for col in ("cover50", "cover90", "below_zero"):
        post[col] = post[col].astype(bool)
    table = summarize(post)
    post["table"] = post["functional"].str.split(":").str[0]
    by_table = summarize(post, keys=("simulation", "cls", "table"))
    pit = pit_histogram(post)
    decision, lines = decide(table)

    tau = read(root, "tau.csv")
    chosen = tau[tau["chosen_b1"].astype(bool)]
    edge = min(tau["tau_multiple"]) if len(tau) else np.nan
    tau_dist = chosen.groupby(["simulation", "tau_multiple"]).size().unstack(fill_value=0)
    scores, rare = read(root, "scores.csv"), read(root, "rare.csv")
    misfit, transfer = read(root, "misfit.csv"), read(root, "transfer.csv")
    acc = scores.groupby(["simulation", "model"])[["lor_mae", "tvd"]].mean()
    ok = rare[(rare["truth"] > 0) & (rare["pred"] > 0)]
    acc = acc.join(np.sqrt((np.log(ok["pred"] / ok["truth"]) ** 2)
                           .groupby([ok["simulation"], ok["model"]]).mean()).rename("rare_rms"))
    acc = acc.join(misfit.groupby(["simulation", "model"])["rms_z"].mean().rename("misfit_rms_z"))
    for k in ("B1", "B2"):
        t = transfer[transfer["scored_set"] == k]
        acc = acc.join(t.groupby(["simulation", "model"])["rms_z"].mean()
                       .rename(f"{k.lower()}_rms_z"))
    noise_free = None
    if args.step3b is not None and (args.step3b / "pooled_models.csv").exists():
        pm = pd.read_csv(args.step3b / "pooled_models.csv").set_index("model")
        if "latent_kern30" in pm.index:
            noise_free = pm.loc[["latent_kern30"]]
    done = [json.loads(p.read_text()) for p in sorted(root.glob("*/*/r*/done.json"))]
    timing = pd.DataFrame(done)

    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 1000):
        text = [f"Step 4: stage 2 posterior calibration; {post['region'].nunique()} regions, "
                f"simulations {', '.join(sorted(post['simulation'].unique()))}, "
                f"{post.groupby('simulation')['replicate'].nunique().to_dict()} replicates", "",
                *lines, "",
                "1. Calibration by simulation and functional class (pooled over replicates, "
                "zones and cells)", table.round(3).to_string(), "",
                "2. PIT histograms (share per decile; 0.1 each if calibrated)",
                pit.round(3).to_string(), "",
                "3. Coverage by table (90% / 50%)",
                by_table[["n", "cover90", "cover50", "rms_z"]].round(3).to_string(), "",
                "4. MAP point accuracy under noise (realized MAP and p_b at the B1-chosen tau)",
                acc.round(4).to_string(), ""]
        if noise_free is not None:
            text += ["   step 3b, noise-free (latent_kern30):",
                     noise_free.round(4).to_string(), ""]
        text += [f"5. tau chosen by B1 (multiples of tau_ref^g); grid's lower edge {edge:g}",
                 tau_dist.to_string(),
                 "   share at the lower edge: " + ", ".join(
                     f"{sim} {float((g['tau_multiple'] == edge).mean()):.0%}"
                     for sim, g in chosen.groupby("simulation")), ""]
        if len(timing):
            text += ["6. Timings",
                     f"   units {len(timing)}; seconds per unit median "
                     f"{timing['seconds'].median():.0f}, max {timing['seconds'].max():.0f}; "
                     f"total {timing['seconds'].sum() / 3600:.1f} CPU-hours; peak memory "
                     f"{timing['peak_gb'].max():.1f} GB; posterior seconds per draw median "
                     f"{timing['seconds_per_draw'].median():.2f}; negative variances "
                     f"{int(timing['negative_variances'].sum())}", ""]
    write_text(root / "summary.txt", "\n".join(text) + "\n")
    write_text(root / "decision.json", json.dumps(decision, indent=1, default=str))
    write_text(root / "calibration.csv", table.reset_index().to_csv(index=False))
    write_text(root / "calibration_tables.csv", by_table.reset_index().to_csv(index=False))
    write_text(root / "pit.csv", pit.reset_index().to_csv(index=False))
    write_text(root / "map_accuracy.csv", acc.reset_index().to_csv(index=False))
    print("\n".join(lines))
    print(f"\nfull summary in {root / 'summary.txt'}")


if __name__ == "__main__":
    main()
