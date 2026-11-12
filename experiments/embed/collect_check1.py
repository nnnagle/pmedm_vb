"""Collect check 1 across fits into one table: one row per fit, its settings,
the mean held-out deviance explained, each target's and each rare category's.

    python experiments/embed/collect_check1.py --root $PMEDM_VB_RUNS/embed/baseline

Writes ``check1_summary.csv`` under ``--root`` and prints the settings with the
mean and the rare categories.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for result in sorted(args.root.glob("*/result.json")):
        directory = result.parent
        info = json.loads(result.read_text())
        row = {"fit": directory.name}
        row.update({k: info[k] for k in ("d", "gamma", "ridge", "weighted", "iterations",
                                         "seconds", "loss", "heldout_mean_deviance_explained")})
        targets = pd.read_csv(directory / "check1_targets.csv")
        for _, t in targets.iterrows():
            row[f"target:{t['target']}"] = t["heldout_deviance_explained"]
        rare = pd.read_csv(directory / "check1_rare.csv")
        for _, r in rare.iterrows():
            row[f"rare:{r['category']}"] = r["heldout_deviance_explained"]
        rows.append(row)
    if not rows:
        raise SystemExit(f"no completed fits under {args.root}")
    summary = pd.DataFrame(rows).sort_values(["weighted", "d", "gamma", "ridge"],
                                             ascending=[False, True, True, True])
    out = args.root / "check1_summary.csv"
    summary.to_csv(out, index=False)
    shown = ["fit", "iterations", "heldout_mean_deviance_explained",
             *[c for c in summary.columns if c.startswith("rare:")]]
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.float_format", "{:.3f}".format):
        print(summary[shown].rename(columns=lambda c: c.replace("rare:", "")).to_string(index=False))
    print(f"\n{len(summary)} fits; full table in {out}")


if __name__ == "__main__":
    main()
