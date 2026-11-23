"""Side check: which diagonal elements of Sigma are exactly zero (zero MOE)?

For the tables the latent model uses (``contribution_tables(area, "all")``:
constraint and held-out sets, at block group and tract), each table's
constraint cells are built exactly as ``build_targets`` builds them for
``Sigma`` (published cells collapsed into categories, replicate deviations
summed), and the raw SDR variance ``(4/80) * sum_r L_r^2`` is taken *before*
any zero-cell policy. A zero there is a zero diagonal element of Sigma's survey
part. Two kinds are reported apart:

* ``zero count``: estimate 0 and variance 0 (the case the floor or the
  ``w * k`` model already handles);
* ``nonzero estimate``: estimate > 0 and variance 0 -- an estimate with no
  sampling error, such as a controlled total.

PUMA level has no replicate files, so it is checked on the published cells
from the Census API (not collapsed): an MOE of 0, split into the "controlled"
jam code ``-555555555`` and a literal 0, again by estimate zero or not.

Tables are checked one at a time, so a table published for a different set of
geographies than another does not stop the run. Statewide (the area of the
stage-1 configuration).

Run on a login node (it reads the cache; PUMA values not cached are downloaded
only if ``CENSUS_API_KEY`` is set and ``PMEDM_VB_OFFLINE`` is not; otherwise
those tables are listed as skipped)::

    python experiments/latent/zero_moe.py --out $R/latent/zero_moe
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.assemble.targets import build_targets
from pmedm_vb.data.acs_api import MOE_CONTROLLED, download_values, values_path
from pmedm_vb.data.variance import SDR_FACTOR
from pmedm_vb.embed.config import load_config
from pmedm_vb.latent.atoms import contribution_tables

LEVELS = ("block group", "tract")


def check_level(area, tables, level: str, kind: str):
    """Per-table summary rows and the zero-variance cells, for one (level, set)."""
    rows, cells = [], []
    for table in tables:
        block = build_targets(area, [table], geography=level, policy="floor", floor=1.0)
        variance = SDR_FACTOR * np.square(block.L).sum(axis=1)
        estimate = block.vec()
        # column-major: entry (i, j) of Y sits at i + n_areas * j
        geoid = np.tile(block.geoids.to_numpy(), block.n_categories)
        name = np.repeat(np.array(block.names), block.n_areas)
        zero = variance == 0.0
        zc = zero & (estimate == 0)
        nz = zero & (estimate > 0)
        geo_any_nz = pd.Series(nz).groupby(geoid).any()
        geo_any_zero = pd.Series(zero).groupby(geoid).any()
        rows.append({
            "level": level, "set": kind, "table": table.table,
            "n_geos": block.n_areas, "n_categories": block.n_categories,
            "n_cells": variance.size,
            "zero_count": int(zc.sum()), "zero_var_nonzero_est": int(nz.sum()),
            "frac_cells_zero_count": zc.mean(), "frac_cells_nonzero_est": nz.mean(),
            "frac_geos_any_zero": geo_any_zero.mean(),
            "frac_geos_any_nonzero_est": geo_any_nz.mean(),
        })
        if nz.any():
            cells.append(pd.DataFrame({"level": level, "set": kind, "table": table.table,
                                       "category": name[nz], "geoid": geoid[nz],
                                       "estimate": estimate[nz]}))
    return rows, cells


def check_puma(area, table_ids):
    """Per-table summary of published PUMA MOEs equal to 0, from the API JSON."""
    rows, cells, skipped = [], [], []
    for table in table_ids:
        try:
            path = download_values(area, table, geography="puma")
        except (FileNotFoundError, RuntimeError) as error:
            skipped.append({"table": table, "reason": str(error).splitlines()[0]})
            continue
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        frame = pd.DataFrame(data[1:], columns=data[0], dtype=str)
        geoid = frame["GEO_ID"].str.split("US").str[-1].to_numpy()
        moe_cols = [c for c in frame.columns if c.startswith(table + "_") and c.endswith("M")]
        n = controlled = literal = controlled_nz = literal_nz = missing = 0
        any_nz = np.zeros(len(frame), dtype=bool)
        for m in moe_cols:
            e = m[:-1] + "E"
            moe = pd.to_numeric(frame[m], errors="coerce").to_numpy()
            est = pd.to_numeric(frame[e], errors="coerce").to_numpy()
            is_c = moe == MOE_CONTROLLED
            is_0 = moe == 0
            nz_est = est > 0
            n += moe.size
            controlled += int(is_c.sum())
            literal += int(is_0.sum())
            controlled_nz += int((is_c & nz_est).sum())
            literal_nz += int((is_0 & nz_est).sum())
            missing += int(((moe < 0) & ~is_c).sum() + np.isnan(moe).sum())
            hit = (is_c | is_0) & nz_est
            any_nz |= hit
            if hit.any():
                cells.append(pd.DataFrame({"level": "puma", "set": "published",
                                           "table": table, "category": m[:-1],
                                           "geoid": geoid[hit], "estimate": est[hit],
                                           "moe_code": np.where(is_c[hit], "controlled", "0")}))
        rows.append({
            "level": "puma", "set": "published", "table": table,
            "n_geos": len(frame), "n_categories": len(moe_cols), "n_cells": n,
            "moe_controlled_code": controlled, "moe_literal_zero": literal,
            "zero_moe_nonzero_est": controlled_nz + literal_nz,
            "moe_missing_jam": missing,
            "frac_cells_nonzero_est": (controlled_nz + literal_nz) / n if n else np.nan,
            "frac_geos_any_nonzero_est": any_nz.mean(),
        })
    return rows, cells, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="experiments/embed/stage1_v2.json",
                        help="embedding configuration whose area is used (statewide)")
    parser.add_argument("--tables", default="all", help="table set, as run_map.py --tables")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--no-puma", action="store_true", help="skip the PUMA level")
    args = parser.parse_args()

    area = replace(load_config(args.config).area, counties=())
    by_block = contribution_tables(area, args.tables)

    rows, cells = [], []
    for (level, kind), tables in by_block.items():
        r, c = check_level(area, tables, level, kind)
        rows += r
        cells += c

    skipped = []
    if not args.no_puma:
        ids = sorted({t.table for tables in by_block.values() for t in tables})
        r, c, skipped = check_puma(area, ids)
        rows += r
        cells += c

    args.out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "by_table.csv", index=False)
    detail = pd.concat(cells, ignore_index=True) if cells else pd.DataFrame()
    detail.to_csv(args.out / "zero_var_nonzero_est_cells.csv", index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(args.out / "puma_skipped.csv", index=False)

    # Level summary: how many tables have any zero-variance cell, by kind.
    def column(name):
        return summary[name].fillna(0) if name in summary else pd.Series(0, index=summary.index)

    summary["any_nonzero_est"] = (column("zero_var_nonzero_est")
                                  + column("zero_moe_nonzero_est")) > 0
    summary["any_zero_count"] = column("zero_count") > 0
    level = summary.groupby(["level", "set"]).agg(
        n_tables=("table", "size"),
        tables_any_nonzero_est=("any_nonzero_est", "sum"),
        tables_any_zero_count=("any_zero_count", "sum"),
        cells=("n_cells", "sum"),
    )
    level["frac_tables_any_nonzero_est"] = level["tables_any_nonzero_est"] / level["n_tables"]
    level["frac_tables_any_zero_count"] = level["tables_any_zero_count"] / level["n_tables"]
    level.to_csv(args.out / "by_level.csv")

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
    print("\n== by level (block group, tract: Sigma diagonal; puma: published MOE) ==")
    print(level.to_string())
    print("\n== by table ==")
    print(summary.drop(columns=["any_nonzero_est", "any_zero_count"]).to_string(index=False))
    if not detail.empty:
        print("\n== zero variance with nonzero estimate: cells per (level, table, category) ==")
        print(detail.groupby(["level", "table", "category"]).size()
              .rename("n_geos").to_string())
    if skipped:
        print(f"\nPUMA tables skipped (not cached, no download): "
              f"{', '.join(s['table'] for s in skipped)}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
