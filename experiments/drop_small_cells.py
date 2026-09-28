"""Drop the constraint categories that are small at the PUMA level.

For each PUMA, a category's PUMA total is the sum of its tract estimates (block
group estimates nest in tracts, so the block group sum is the same number). A
category whose PUMA total is below the threshold for its table is removed at
every level: its tract rows, its block group rows, and so its PUMA row under
``--hierarchy puma``. Thresholds are in the table's own unit: ``--person-min``
people for person tables, ``--household-min`` households for household tables
(:data:`HOUSEHOLD_TABLES`). B26001 (group quarters) counts people.

Each reduced problem is written as a new study area, ``processed/inputs/
<out-area>/<puma>``, with the held-out tables copied unchanged, so the fitting
and scoring scripts run on it as they are: ``run_map.py --name`` with the name
part of ``<out-area>`` (``knox-min`` for ``knox-min-2024-5yr``), and
``compare_methods.py --area <out-area>``.

The ``zero`` variance floor is recomputed from the kept cells. Within an area
the zero-cell variance is one number, so it changes only where an area loses
all its published zeros and falls back to the median over areas; how many rows
that affects is reported. Writes ``<out>/dropped.csv`` (one row per dropped
category) and prints a per-PUMA summary::

    $CONDA_PREFIX/bin/python experiments/drop_small_cells.py \\
        --puma 4701501 4701502 4701503 4701504 --out $RUNS/drop_small
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.assemble.heldout import HELDOUT_DIR
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir

#: Constrained tables whose unit is the household; every other table counts people.
HOUSEHOLD_TABLES = frozenset({"B19001", "B25003"})
#: Every table the default constraint set uses, so an unknown one is an error.
PERSON_TABLES = frozenset({"B01001", "B03002", "B08301", "C24010", "B26001", "B17024", "B23001"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--puma", nargs="+", required=True)
    parser.add_argument("--area", default="knox-2024-5yr")
    parser.add_argument("--out-area", default="knox-min-2024-5yr")
    parser.add_argument("--person-min", type=float, default=50.0)
    parser.add_argument("--household-min", type=float, default=20.0)
    parser.add_argument("--out", type=Path, required=True, help="directory for dropped.csv")
    return parser.parse_args()


def threshold(name: str, person_min: float, household_min: float) -> float:
    table = name.split(".")[0]
    if table in HOUSEHOLD_TABLES:
        return household_min
    if table in PERSON_TABLES:
        return person_min
    raise ValueError(f"{name}: table {table} is not classed as person or household")


def drop_small(inputs: PMEDMInputs, person_min: float, household_min: float
               ) -> tuple[PMEDMInputs, pd.DataFrame]:
    """The problem without its small categories, and a frame of what was dropped."""
    if inputs.support is not None:
        raise ValueError("statewide-support problems are not supported")
    totals = inputs.Y_T.sum(axis=0)
    limits = np.array([threshold(c, person_min, household_min) for c in inputs.tract_constraints])
    small = totals < limits
    dropped_names = {c for c, s in zip(inputs.tract_constraints, small) if s}
    missing = set(inputs.bg_constraints) - set(inputs.tract_constraints)
    if missing:
        raise ValueError(f"block group categories with no tract row: {sorted(missing)}")
    keep_t = ~small
    keep_b = np.array([c not in dropped_names for c in inputs.bg_constraints])

    n_tracts, n_bg = inputs.Y_T.shape[0], inputs.Y_B.shape[0]
    rows = np.concatenate([
        np.repeat(keep_t, n_tracts),  # stacked column-major: category k's areas together
        np.repeat(keep_b, n_bg),
    ])
    reduced = dataclasses.replace(
        inputs,
        X_T=inputs.X_T.tocsc()[:, np.flatnonzero(keep_t)],
        X_B=inputs.X_B.tocsc()[:, np.flatnonzero(keep_b)],
        Y_T=inputs.Y_T[:, keep_t],
        Y_B=inputs.Y_B[:, keep_b],
        sigma_v=inputs.sigma_v[rows],
        sigma_l=None if inputs.sigma_l is None else inputs.sigma_l[rows],
        tract_constraints=[c for c, k in zip(inputs.tract_constraints, keep_t) if k],
        bg_constraints=[c for c, k in zip(inputs.bg_constraints, keep_b) if k],
    )
    reduced.validate()

    dropped = pd.DataFrame({
        "puma": inputs.puma,
        "category": np.array(inputs.tract_constraints)[small],
        "puma_total": totals[small],
        "threshold": limits[small],
        "at_block_group": [c in set(inputs.bg_constraints) for c in np.array(inputs.tract_constraints)[small]],
    })
    return reduced, dropped


def floor_changes(before: PMEDMInputs, after: PMEDMInputs, rows: np.ndarray) -> int:
    """Kept rows whose zero-cell floor differs after the drop."""
    old = before.zero_cell_variances()[rows]
    new = after.zero_cell_variances()
    return int((~np.isclose(old, new)).sum())


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames, lines = [], []
    for puma in args.puma:
        source = processed_dir() / "inputs" / args.area / puma
        target = processed_dir() / "inputs" / args.out_area / puma
        inputs = PMEDMInputs.load(source)
        reduced, dropped = drop_small(inputs, args.person_min, args.household_min)
        kept = np.concatenate([
            np.repeat(np.isin(inputs.tract_constraints, reduced.tract_constraints), inputs.Y_T.shape[0]),
            np.repeat(np.isin(inputs.bg_constraints, reduced.bg_constraints), inputs.Y_B.shape[0]),
        ])
        changed = floor_changes(inputs, reduced, kept)
        if target.exists():
            shutil.rmtree(target)
        reduced.save(target)
        if (source / HELDOUT_DIR).exists():
            shutil.copytree(source / HELDOUT_DIR, target / HELDOUT_DIR)
        frames.append(dropped)
        by_table = dropped.category.str.split(".").str[0].value_counts().sort_index()
        lines.append(
            f"{puma}: dropped {len(dropped)} of {len(inputs.tract_constraints)} tract categories "
            f"({int(dropped.at_block_group.sum())} also at block group); constraint rows "
            f"{inputs.n_constraints:,} -> {reduced.n_constraints:,}; zero-cell floor changed on "
            f"{changed} kept rows\n    by table: "
            + (", ".join(f"{t} {n}" for t, n in by_table.items()) or "none"))
    pd.concat(frames).to_csv(args.out / "dropped.csv", index=False)
    text = "\n".join(lines) + "\n"
    (args.out / "drop_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
