"""Pool the leave-one-PUMA-out results under one output root.

    python experiments/embed/collect_puma_check.py --root $PMEDM_VB_RUNS/embed/puma_check

Reads every ``<root>/<label>/<puma>/{scores,misfit,rare}.csv`` and writes, under
``--root``, ``summary_overall.csv`` (means over PUMAs by source, set and tilt:
TVD, KL, log-odds-ratio errors, constraint misfit), ``summary_tables.csv`` (the
same per scoring table) and ``summary_rare.csv`` (rare groups summed over
PUMAs against truth and, where published PUMA tables were available, the z of
the pooled prediction and the RMS of per-PUMA z). Prints the overall table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pmedm_vb.embed.puma_check import summarise
from pmedm_vb.progress import write_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
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
    print(f"\nPUMAs per source: {n.to_dict()}; tables in {args.root}/summary_*.csv")


if __name__ == "__main__":
    main()
