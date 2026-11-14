"""Pool the leave-one-PUMA-out results under one output root.

    python experiments/embed/collect_puma_check.py --root $PMEDM_VB_RUNS/embed/puma_check

Reads every ``<root>/<label>/<puma>/{scores,misfit,rare}.csv`` and writes, under
``--root``, ``summary_overall.csv`` (means over PUMAs by source, set and tilt:
TVD, KL, log-odds-ratio errors, constraint misfit), ``summary_tables.csv`` (the
same per scoring table) and ``summary_rare.csv`` (rare groups summed over
PUMAs against truth and, where published PUMA tables were available, the z of
the pooled prediction and the RMS of per-PUMA z). Prints the overall table,
then, for one constraint set (``--set``, default S1), every tilt side by side:
TVD and log-odds-ratio error overall and per scoring table, and the rare
groups' pooled prediction over truth. Tilts of different feature maps h carry
their suffix (``phi_quad``, ``oracle_dec``...), so all h line up as columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pmedm_vb.embed.puma_check import FEATURE_MAPS, H_TILTS, TILTS, summarise
from pmedm_vb.progress import write_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--set", default="S1", help="constraint set for the side-by-side tables")
    args = parser.parse_args()

    frames = {}
    for name in ("scores", "misfit", "rare"):
        paths = sorted(args.root.glob(f"*/*/{name}.csv"))
        if not paths:
            raise SystemExit(f"no {name}.csv under {args.root}")
        frames[name] = pd.concat([pd.read_csv(p, dtype={"puma": str}) for p in paths],
                                 ignore_index=True)
    summary = summarise(frames["scores"], frames["misfit"], frames["rare"])
    for name, frame in summary.items():
        write_text(args.root / f"summary_{name}.csv", frame.to_csv(index=False))
    n = frames["misfit"].groupby("source")["puma"].nunique()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary["overall"].round(4).to_string(index=False))
    order = [*(t for t in TILTS if t not in H_TILTS),
             *(t + suffix for suffix in FEATURE_MAPS.values() for t in H_TILTS)]

    def side_by_side(frame, index, value):
        frame = frame[frame["set"] == args.set]
        if frame.empty or value not in frame:
            return None
        wide = frame.pivot_table(index=index, columns="tilt", values=value)
        return wide[[t for t in order if t in wide.columns]].round(3)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for label, frame, index in (("overall", summary["overall"], "source"),
                                    ("per table", summary["tables"], ["source", "table"])):
            for value in ("tvd", "lor_mae"):
                wide = side_by_side(frame, index, value)
                if wide is not None:
                    print(f"\n{args.set} {label}: {value}\n{wide.to_string()}")
        rare = summary["rare"]
        wide = side_by_side(rare, ["source", "table", "category"], "pred_over_truth")
        if wide is not None:
            print(f"\n{args.set} rare groups: pooled prediction / truth\n{wide.to_string()}")
    print(f"\nPUMAs per source: {n.to_dict()}; tables in {args.root}/summary_*.csv")


if __name__ == "__main__":
    main()
