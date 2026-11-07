"""How many sampled cases each constraint cell rests on, over every assembled PUMA of an area.

If a published estimate is the sum of ``k`` sampled cases of about the same
weight ``w``, it is about ``k w`` and its sampling standard error about
``w sqrt(k)``, so

.. math::

    \\hat k = (\\text{estimate} / \\text{SE})^2 = 1 / \\text{CV}^2

is roughly the number of sampled cases behind the cell. A tract estimate of 10
with an SE of 8.5 has ``k_hat`` about 1.4: one sampled household. Under a
Poisson count model, one sampled case and none are nearly indistinguishable
(their likelihood ratio is ``f T``, about 1 for a true count near one
sampling weight), yet the Gaussian error treats "10 +- 9" as a symmetric,
continuous measurement. This script counts such cells.

``k_hat`` uses each cell's published variance (``sigma_v``, the replicate
variance, or the modelled zero-count variance for a published zero), and also
the variance floored at the zero-count variance as the fits use it
(``variance_floor: "zero"``, :meth:`~pmedm_vb.assemble.inputs.PMEDMInputs.floored_variances`).
Both are approximations: ACS weights vary within a cell, and the design is not
simple random sampling.

With ``--exp`` (an experiment folder, e.g. ``$R/tn01_gpu``), each cell also
gets its multiplier at the MAP (``<exp>/<puma>/map/*.npz``, ``lam``: the
multipliers the data see), so the summary shows whether extreme multipliers
sit on cells resting on about one case.

Usage::

    python experiments/cell_cases.py --area tennessee-5t-2024-5yr --exp $R/tn01_gpu

Writes ``cells.csv`` (one row per PUMA, level, area and constraint) and
``summary.txt`` to ``--out`` (default ``<exp>/cell_cases``, or ``./cell_cases``
without ``--exp``).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir

#: ``k_hat`` bins for the summary; a published zero is its own bin.
BINS = [0.0, 1.0, 2.0, 3.0, 10.0, np.inf]
LABELS = ["(0,1]", "(1,2]", "(2,3]", "(3,10]", ">10"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", required=True, help="assembled-inputs folder name")
    parser.add_argument("--exp", type=Path, default=None,
                        help="experiment folder, for each PUMA's MAP multipliers")
    parser.add_argument("--puma", nargs="+", default=None, help="only these PUMAs")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def cell_frame(inputs: PMEDMInputs, puma: str) -> pd.DataFrame:
    """One row per constraint cell in the stacked order ``[vec(Y_T); vec(Y_B)]``."""
    blocks = []
    for level, Y, areas, names in (
        ("tract", inputs.Y_T, inputs.tracts.iloc[:, 0], inputs.tract_constraints),
        ("block group", inputs.Y_B, inputs.block_groups.iloc[:, 0], inputs.bg_constraints),
    ):
        n_areas, k = Y.shape
        blocks.append(pd.DataFrame({  # column-major: area varies fastest
            "level": level,
            "area": np.tile(areas.astype(str).to_numpy(), k),
            "constraint": np.repeat(list(names), n_areas),
        }))
    frame = pd.concat(blocks, ignore_index=True)
    frame.insert(0, "puma", puma)
    frame["table"] = frame["constraint"].str.split(".").str[0]
    frame["category"] = frame["constraint"].str.split(".", n=1).str[1]
    frame["estimate"] = inputs.targets()
    frame["se"] = np.sqrt(inputs.sigma_v)
    frame["se_floored"] = np.sqrt(inputs.floored_variances("zero"))
    for column, se in (("k_hat", "se"), ("k_hat_floored", "se_floored")):
        with np.errstate(divide="ignore", invalid="ignore"):
            frame[column] = np.where(frame["estimate"] > 0,
                                     (frame["estimate"] / frame[se]) ** 2, 0.0)
    return frame


def map_multipliers(exp: Path, puma: str, m: int) -> np.ndarray | None:
    """The PUMA's MAP multipliers (the ones the data see), if exactly one MAP fit is there."""
    paths = sorted(glob.glob(str(exp / puma / "map" / "*.npz")))
    if len(paths) != 1:
        return None
    with np.load(paths[0]) as saved:
        lam = np.asarray(saved["lam"], dtype=float)
    return lam if lam.size == m else None


def shares(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per group: cells, published zeros, and the share of nonzero cells under k_hat 1, 2, 3."""
    def one(g: pd.DataFrame) -> pd.Series:
        nonzero = g[g["estimate"] > 0]
        out = {"cells": len(g), "zero": int((g["estimate"] == 0).sum()), "nonzero": len(nonzero)}
        for t in (1, 2, 3):
            out[f"k<{t}"] = float((nonzero["k_hat"] < t).mean()) if len(nonzero) else np.nan
        out["k<2 floored"] = (float((nonzero["k_hat_floored"] < 2).mean())
                              if len(nonzero) else np.nan)
        return pd.Series(out)
    table = frame.groupby(by).apply(one).reset_index()
    return table.astype({"cells": int, "zero": int, "nonzero": int})


def main() -> None:
    args = parse_args()
    root = processed_dir() / "inputs" / args.area
    pumas = sorted(p.name for p in root.glob("*") if (p / "manifest.json").exists())
    if args.puma:
        pumas = [p for p in pumas if p in set(args.puma)]
    if not pumas:
        raise SystemExit(f"no assembled PUMAs under {root}")
    out = args.out or ((args.exp / "cell_cases") if args.exp else Path("cell_cases"))
    out.mkdir(parents=True, exist_ok=True)

    frames, with_lam = [], 0
    for puma in pumas:
        inputs = PMEDMInputs.load(root / puma)
        frame = cell_frame(inputs, puma)
        lam = map_multipliers(args.exp, puma, len(frame)) if args.exp else None
        frame["map_lam"] = lam if lam is not None else np.nan
        with_lam += lam is not None
        frames.append(frame)
    cells = pd.concat(frames, ignore_index=True)
    cells.to_csv(out / "cells.csv", index=False)

    nonzero = cells[cells["estimate"] > 0]
    lines = [
        f"{args.area}: {len(pumas)} PUMAs, {len(cells):,} constraint cells, "
        f"{(cells['estimate'] == 0).mean():.1%} published as zero",
        "k_hat = (estimate / SE)^2, about the number of sampled cases behind a cell;",
        "shares below are of the cells with a nonzero estimate ('floored': with the "
        "zero-count variance floor the fits use)",
        "",
        f"all nonzero cells ({len(nonzero):,}): k_hat < 1 {(nonzero['k_hat'] < 1).mean():.1%}, "
        f"< 2 {(nonzero['k_hat'] < 2).mean():.1%}, < 3 {(nonzero['k_hat'] < 3).mean():.1%}; "
        f"floored < 2 {(nonzero['k_hat_floored'] < 2).mean():.1%}",
        "",
        "By level and table:",
    ]
    lines += ["  " + s for s in shares(cells, ["level", "table"]).round(3)
              .to_string(index=False).splitlines()]
    race = cells[cells["table"] == "B03002"]
    if len(race):
        lines += ["", "B03002 (race and ethnicity) by level and category:"]
        lines += ["  " + s for s in shares(race, ["level", "category"]).round(3)
                  .to_string(index=False).splitlines()]
    if with_lam:
        withl = cells.dropna(subset=["map_lam"]).copy()
        withl["k bin"] = np.where(withl["estimate"] == 0, "zero",
                                  pd.cut(withl["k_hat"], BINS, labels=LABELS).astype(str))
        withl["abs_lam"] = withl["map_lam"].abs()
        lam_by = (withl.groupby("k bin")["abs_lam"]
                  .agg(cells="size", median="median", p95=lambda x: x.quantile(0.95), max="max")
                  .reindex(["zero", *LABELS]).dropna(how="all"))
        lines += ["", f"|MAP multiplier| by k_hat ({with_lam} PUMAs with one MAP fit):"]
        lines += ["  " + s for s in lam_by.round(3).to_string().splitlines()]
        top = withl.nlargest(15, "abs_lam")[
            ["puma", "level", "area", "constraint", "estimate", "se", "k_hat", "map_lam"]]
        lines += ["", "fifteen largest |MAP multiplier|:"]
        lines += ["  " + s for s in top.round(3).to_string(index=False).splitlines()]
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
