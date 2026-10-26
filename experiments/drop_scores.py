"""Remove some methods' scores from experiments' scorings, so that the driver
scores those methods again (``score_experiment.py --resume`` scores only the
calls without scores; ``compare_methods.py`` appends).

For each experiment folder and the ``--tag``'s scoring, drops the ``--methods``
rows from every ``per_cell/<cell>.csv`` and ``<cell>_tables.csv`` and deletes
their ``detail/<cell>/<method>_cells.parquet`` and ``_draws.parquet``. The
combined ``all_scores.csv`` is then older than the cell CSVs, so the driver
combines again. ``--dry-run`` reports and changes nothing. After IPF began
raking to the tract-only tables (``pmedm_vb.rake``)::

    $CONDA_PREFIX/bin/python experiments/drop_scores.py --tag paper2 \\
        --methods ipf ipf_multinomial \\
        --experiments $R/exp01_baseline $R/exp02_nullspace $R/exp03_rollup
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", type=Path, nargs="+", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = set(args.methods)
    for exp in args.experiments:
        out = exp / "scores" / args.tag
        if not out.exists():
            print(f"{out}: no scoring; skipped")
            continue
        rows = files = 0
        for path in sorted((out / "per_cell").glob("*.csv")):
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            drop = frame["method"].isin(methods)
            if drop.any():
                rows += int(drop.sum())
                if not args.dry_run:
                    frame[~drop].to_csv(path, index=False)
        for method in sorted(methods):
            for path in sorted([*(out / "detail").glob(f"*/{method}_cells.parquet"),
                                *(out / "detail").glob(f"*/{method}_draws.parquet")]):
                files += 1
                if not args.dry_run:
                    path.unlink()
        verb = "would drop" if args.dry_run else "dropped"
        print(f"{exp.name}: {verb} {rows:,} CSV row(s) and {files} detail file(s) of "
              f"{', '.join(sorted(methods))}")


if __name__ == "__main__":
    main()
