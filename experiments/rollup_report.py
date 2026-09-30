"""Cells before and after the structure-aware roll-up, per PUMA, level and table.

For each PUMA and each roll-up threshold, builds the roll-up
(:mod:`pmedm_vb.assemble.rollup`) at every level -- the PUMA, each tract, each
block group -- and counts, per table:

- ``categories``: category x area rows before (summed over the level's areas);
- ``cells``: cells after, of which ``zero_cells`` are zero cells;
- ``zero_categories``: rows that were published zeros;
- ``parallel_merges``: outer groups merged in a two-way block (e.g. age
  groups, to keep two poverty classes);
- ``collapsed_nodes``: child nodes below the threshold made one cell;
- ``crossings``: cells merged into an adjacent sub-universe's cell;
- ``cells_below_threshold``: positive cells still below it (a table whose
  whole count in the area is below it).

Writes ``<out>/rollup_report.csv`` (one row per PUMA, threshold, level and
table) and ``<out>/rollup_report.txt`` (summed over the PUMAs). Needs the
category trees in the inputs (``table_trees.py --write-inputs``)::

    $CONDA_PREFIX/bin/python experiments/rollup_report.py \\
        --puma 4701501 4701502 4701503 4701504 --rollup 15 15h10 --out $RUNS/rollup_report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pmedm_vb.assemble.collapse import CollapsedHierarchy
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.rollup import RollupSpec
from pmedm_vb.config import processed_dir

COUNTS = ["areas", "categories", "cells", "zero_cells", "zero_categories", "parallel_merges",
          "collapsed_nodes", "crossings", "cells_below_threshold"]
LEVEL_ORDER = {"puma": 0, "tract": 1, "block group": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--puma", nargs="+", required=True)
    parser.add_argument("--area", default="knox-2024-5yr")
    parser.add_argument("--rollup", nargs="+", required=True,
                        help="thresholds, PERSONS or PERSONShHOUSEHOLDS (e.g. 15 15h10)")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for puma in args.puma:
        inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / puma)
        for text in args.rollup:
            spec = RollupSpec.parse(text)
            h = CollapsedHierarchy.build(inputs, "puma", spec)
            rows += [dict(puma=puma, rollup=str(spec), **r) for r in h.rollup_report]
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "rollup_report.csv", index=False)

    lines = [f"Structure-aware roll-up, {args.area}, PUMAs {' '.join(args.puma)} (summed)", ""]
    total = frame.groupby(["rollup", "level", "table", "universe", "threshold"], sort=False)[COUNTS].sum()
    total = total.reset_index()
    total["order"] = total["level"].map(LEVEL_ORDER)
    for spec, part in total.groupby("rollup", sort=False):
        lines.append(f"== rollup {spec}")
        part = part.sort_values(["order", "table"])
        shown = part[["level", "table", "universe", "threshold", *COUNTS]].copy()
        shown["cells/categories"] = (shown["cells"] / shown["categories"]).round(3)
        lines.append(shown.to_string(index=False))
        by_level = part.groupby("level", sort=False)[["categories", "cells", "zero_cells",
                                                     "cells_below_threshold"]].sum()
        lines.append("")
        lines.append(by_level.to_string())
        lines.append("")
    text = "\n".join(lines) + "\n"
    (args.out / "rollup_report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
