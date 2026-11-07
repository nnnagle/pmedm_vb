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

**Why the multipliers are large** (with ``--exp``). At the MAP, ``Sigma zeta =
(N / n) (allocation - estimate)`` (the dual's stationarity condition, up to each
sum-to-zero group's mean under a hierarchy), so a large multiplier can come from

1. a small miss in a direction ``Sigma`` pins: at ``alpha = 0.01`` a set of
   cells the replicates call perfectly correlated keeps 1% of its variance, so
   a miss of one SE there takes a multiplier 100 times larger;
2. a large miss on an ordinary cell, as a symmetric Gaussian error allows on a
   cell resting on one case; or
3. too few support records to reach a total (the PUMA multiplier ``lambda_P``).

Each cell therefore gets its MAP allocation (``A W X`` from the saved ``W``),
its residual in SE units, its own multiplier ``zeta`` (the split, for a
hierarchy's tract and block group rows; ``lambda_data`` adds ``lambda_P`` to
tract rows), the PUMA part, and a **pinning ratio**
``(Sigma zeta)_i / (diag(Sigma)_i zeta_i)``: about 1 where the cell's own
variance carries its multiplier, near 0 (or negative) where correlations with
other cells cancel it -- a pinned direction. The ``--top`` largest ``|lambda|``
cells get a heuristic cause (``pinned``: residual within 2 SE and pinning ratio
under 0.5; ``puma``: a tract row whose ``lambda_P`` part is the larger; ``miss``:
residual over 2 SE; else ``other``) and their **replicate-correlation
clusters**: cosines of their replicate deviations (``sigma_l``, the
correlations ``R`` in ``Sigma``) with every other row of their PUMA, the
largest, how many exceed 0.95, and those partners (``clusters.csv``). PUMA rows
(``puma_rows.csv``) get the same residual, multiplier and pinning columns, and
the number of support units carrying the category. For a fit with a Tweedie or
K0 term, the stationarity condition on those rows involves the term rather
than ``Sigma``, so their pinning ratio (computed from ``Sigma`` alone) is only
indicative.

**Cells a cell-by-cell Poisson model would pull out.** At each threshold in
``--thresholds``, every finest-level cell (block group; tract for a category
constrained at tract only) with ``k_hat`` below it, zeros included, would leave
the Gaussian hierarchy. Per (tract, category) -- per (PUMA, category) for
tract-only categories -- the cells are classed as all zero, all pulled, one
kept or two or more kept, and the effective dimension (one per finest cell)
compared with the rows left after merging each group's zeros into one row (one
row for a category zero across the whole PUMA).

**Zero cells with no support.** A published zero whose category no support unit
carries (``X`` column all zero) has an allocation of exactly zero whatever its
multiplier, so its row only adds a coordinate. Counted by level and table.

Writes ``cells.csv`` (one row per PUMA, level, area and constraint),
``summary.txt`` and, with ``--exp``, ``puma_rows.csv``, ``clusters.csv`` and
``pullout.csv`` to ``--out`` (default ``<exp>/cell_cases``, or ``./cell_cases``
without ``--exp``). ``pullout.csv`` is written without ``--exp`` too.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.hierarchy import Hierarchy
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.solvers.base import ConstraintOperator

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
    parser.add_argument("--top", type=int, default=50,
                        help="how many of the largest |MAP lambda| cells to explain (default 50)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[2.0, 3.0],
                        help="k_hat thresholds for the pull-out counts (default 2 3)")
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


def map_fit(exp: Path, puma: str) -> dict | None:
    """The PUMA's MAP fit, if exactly one is there: ``lam`` (the multipliers the
    data see), ``W`` (population weights), ``xi`` and the model options."""
    paths = sorted(glob.glob(str(exp / puma / "map" / "*.npz")))
    if len(paths) != 1:
        return None
    with np.load(paths[0]) as saved:
        fit = {"lam": np.asarray(saved["lam"], dtype=float),
               "W": np.asarray(saved["W"], dtype=float) if "W" in saved.files else None,
               "xi": np.asarray(saved["xi"], dtype=float) if "xi" in saved.files else None}
        for key, default in (("hierarchy", "none"), ("taper", "none"), ("alpha", "nan"),
                             ("variance_floor", "none")):
            fit[key] = str(saved[key]) if key in saved.files else default
    return fit


def floor_spec(text: str) -> str | float | None:
    return None if text == "none" else "zero" if text == "zero" else float(text)


def support_units(inputs: PMEDMInputs) -> np.ndarray:
    """Per stacked row, the number of support units carrying its category
    (``X`` column nonzero), PUMA-wide."""
    t = np.diff(sp.csc_matrix(inputs.X_T).indptr)
    b = np.diff(sp.csc_matrix(inputs.X_B).indptr)
    return np.concatenate([np.repeat(t, inputs.Y_T.shape[0]), np.repeat(b, inputs.Y_B.shape[0])])


def map_diagnostics(inputs: PMEDMInputs, fit: dict, frame: pd.DataFrame,
                    puma: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Add the MAP columns to ``frame`` (allocation, residual in SE, own
    multiplier, PUMA part, pinning ratio); return it and the PUMA rows' frame."""
    m = len(frame)
    frame["map_lam"] = fit["lam"]
    for column in ("allocation", "resid_se", "resid_se_floored", "map_zeta", "map_lam_puma",
                   "pinning"):
        frame[column] = np.nan
    if fit["W"] is None:
        return frame, None
    alloc = ConstraintOperator(inputs).forward(fit["W"])
    frame["allocation"] = alloc
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["resid_se"] = (alloc - frame["estimate"]) / frame["se"]
        frame["resid_se_floored"] = (alloc - frame["estimate"]) / frame["se_floored"]
    level = fit["hierarchy"]
    alpha = float(fit["alpha"])
    taper = None if fit["taper"] == "none" else fit["taper"]
    floor = floor_spec(fit["variance_floor"])
    try:
        h = Hierarchy.build(inputs, level, alpha=alpha, taper=taper, variance_floor=floor)
    except Exception as error:  # an exotic level: keep the residuals, skip the rest
        print(f"{puma}: no multiplier breakdown for hierarchy {level!r} ({error})")
        return frame, None
    if getattr(h, "is_collapsed", False) or (h.n_puma and fit["xi"] is None):
        return frame, None
    xi = fit["xi"] if fit["xi"] is not None else fit["lam"]
    zeta = h.zeta(xi) if not h.is_trivial else fit["lam"]
    if zeta.size != m + h.n_puma:
        return frame, None
    sigma = h.sigma(inputs, alpha, taper, floor)
    s_zeta = sigma.matvec(zeta)
    diag = sigma.diagonal()
    with np.errstate(divide="ignore", invalid="ignore"):
        pinning = np.where(np.abs(zeta) > 1e-9, s_zeta / (diag * zeta), np.nan)
    frame["map_zeta"] = zeta[:m]
    lam_p = np.zeros(m)
    if h.n_puma:
        rows = h.puma_of_row >= 0
        lam_p[rows] = xi[m:][h.puma_of_row[rows]]
    frame["map_lam_puma"] = lam_p
    frame["pinning"] = pinning[:m]
    if not h.n_puma:
        return frame, None
    tract_alloc = alloc[: inputs.Y_T.size].reshape(inputs.Y_T.shape, order="F").sum(axis=0)
    se_p = np.sqrt(h.v_P)
    t_support = np.diff(sp.csc_matrix(inputs.X_T).indptr)
    puma_rows = pd.DataFrame({
        "puma": puma, "constraint": list(inputs.tract_constraints), "estimate": h.Y_P,
        "se": se_p, "allocation": tract_alloc,
        "resid_se": (tract_alloc - h.Y_P) / np.where(se_p > 0, se_p, np.nan),
        "map_lam_puma": zeta[m:], "pinning": pinning[m:], "support_units": t_support})
    with np.errstate(divide="ignore", invalid="ignore"):
        puma_rows["k_hat"] = np.where(h.Y_P > 0, (h.Y_P / se_p) ** 2, 0.0)
    return frame, puma_rows


def cause(row: pd.Series) -> str:
    """A heuristic label for why a cell's multiplier is large (see the module docstring)."""
    if abs(row["map_lam_puma"]) > abs(row["map_zeta"]):
        return "puma"
    if abs(row["resid_se"]) <= 2 and row["pinning"] < 0.5:
        return "pinned"
    if abs(row["resid_se"]) > 2:
        return "miss"
    return "other"


def clusters(inputs: PMEDMInputs, frame: pd.DataFrame, positions: np.ndarray,
             partners: int = 20) -> tuple[list[dict], list[dict]]:
    """For each row position, its replicate-correlation cluster within the PUMA:
    the largest |cosine| of its replicate deviations with any other row, how
    many exceed 0.95, and (up to ``partners``) those rows."""
    if inputs.sigma_l is None:
        return [], []
    L = np.asarray(inputs.sigma_l, dtype=float)
    norms = np.linalg.norm(L, axis=1)
    unit = np.divide(L, norms[:, None], out=np.zeros_like(L), where=norms[:, None] > 0)
    summary, pairs = [], []
    for i in positions:
        if norms[i] == 0:
            summary.append({"position": i, "max_abs_r": np.nan, "n_r95": 0})
            continue
        r = unit @ unit[i]
        r[i] = 0.0
        strong = np.flatnonzero(np.abs(r) > 0.95)
        summary.append({"position": i, "max_abs_r": float(np.abs(r).max()), "n_r95": int(strong.size),
                        "r95_tables": ",".join(sorted(set(frame["table"].iloc[strong])))})
        for j in strong[np.argsort(-np.abs(r[strong]))][:partners]:
            pairs.append({"puma": frame["puma"].iloc[i], "cell": cell_label(frame.iloc[i]),
                          "partner": cell_label(frame.iloc[j]), "r": float(r[j]),
                          "partner_k_hat": float(frame["k_hat"].iloc[j]),
                          "partner_estimate": float(frame["estimate"].iloc[j])})
    return summary, pairs


def cell_label(row: pd.Series) -> str:
    return f"{row['level']}:{row['area']}:{row['constraint']}"


def pullout(inputs: PMEDMInputs, frame: pd.DataFrame, puma: str,
            thresholds: list[float]) -> pd.DataFrame:
    """Per (tract, category) -- (PUMA, category) for tract-only categories --
    and threshold: the finest-level cells pulled out (k_hat below it, zeros
    included), the class, and the effective dimension before and after merging
    each group's zeros into one row."""
    bg_names = set(inputs.bg_constraints)
    n_bg = inputs.Y_B.shape[0]
    bg_tract = inputs.zone_tracts() if inputs.A_B.shape[0] == inputs.n_zones else None
    split = inputs.Y_T.size
    out = []
    groups = []  # (category, table, group label, positions)
    for k, name in enumerate(inputs.bg_constraints):
        rows = split + k * n_bg + np.arange(n_bg)
        if bg_tract is None:
            groups.append((name, "puma", rows))
            continue
        for t in np.unique(bg_tract):
            groups.append((name, f"tract {inputs.tracts.iloc[t, 0]}", rows[bg_tract == t]))
    n_t = inputs.Y_T.shape[0]
    for k, name in enumerate(inputs.tract_constraints):
        if name not in bg_names:
            groups.append((name, "puma (tract-only)", k * n_t + np.arange(n_t)))
    estimate = frame["estimate"].to_numpy()
    for column in ("k_hat", "k_hat_floored"):
        khat = frame[column].to_numpy()
        for c in thresholds:
            for name, where, rows in groups:
                est, kh = estimate[rows], khat[rows]
                zero = est == 0
                low = (~zero) & (kh < c)
                kept = (~zero) & (kh >= c)
                if zero.all():
                    klass = "all zero"
                elif not kept.any():
                    klass = "all pulled"
                elif kept.sum() == 1:
                    klass = "one kept"
                else:
                    klass = "two or more kept"
                out.append({"puma": puma, "k_hat": column, "threshold": c, "constraint": name,
                            "table": name.split(".")[0], "group": where, "class": klass,
                            "cells": rows.size, "zero": int(zero.sum()), "low": int(low.sum()),
                            "kept": int(kept.sum()),
                            "rows_after": int(kept.sum() + low.sum() + (1 if zero.any() else 0))})
    table = pd.DataFrame(out)
    # A category zero across the whole PUMA is one row, not one per tract.
    for (kh, c, name), g in table.groupby(["k_hat", "threshold", "constraint"]):
        if (g["class"] == "all zero").all() and len(g) > 1:
            table.loc[g.index, "rows_after"] = 0
            table.loc[g.index[0], "rows_after"] = 1
    return table


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


MAP_COLUMNS = ("map_lam", "allocation", "resid_se", "resid_se_floored", "map_zeta",
               "map_lam_puma", "pinning")


def block(frame: pd.DataFrame, **kwargs) -> list[str]:
    return ["  " + line for line in frame.to_string(**kwargs).splitlines()]


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

    frames, puma_frames, pulls, with_lam = [], [], [], 0
    for puma in pumas:
        inputs = PMEDMInputs.load(root / puma)
        frame = cell_frame(inputs, puma)
        frame["support_units"] = support_units(inputs)
        fit = map_fit(args.exp, puma) if args.exp else None
        if fit is not None and fit["lam"].size == len(frame):
            frame, puma_rows = map_diagnostics(inputs, fit, frame, puma)
            with_lam += 1
            if puma_rows is not None:
                puma_frames.append(puma_rows)
        else:
            for column in MAP_COLUMNS:
                frame[column] = np.nan
        pulls.append(pullout(inputs, frame, puma, args.thresholds))
        frames.append(frame)
    cells = pd.concat(frames, ignore_index=True)
    pull = pd.concat(pulls, ignore_index=True)
    pull.to_csv(out / "pullout.csv", index=False)

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
        lines += explain(args, root, withl, puma_frames, out)
    cells.to_csv(out / "cells.csv", index=False)
    lines += pullout_summary(pull, args.thresholds)
    lines += no_support_summary(cells)
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"written to {out}")


def explain(args, root: Path, withl: pd.DataFrame, puma_frames: list, out: Path) -> list[str]:
    """The residual, cause and cluster sections; writes ``clusters.csv`` and ``puma_rows.csv``."""
    lines = []
    has = withl.dropna(subset=["resid_se"])
    if len(has):
        has = has.assign(abs_resid=has["resid_se"].abs())
        by = (has.groupby("k bin")
              .agg(cells=("abs_resid", "size"), median_abs_resid=("abs_resid", "median"),
                   over_2se=("abs_resid", lambda x: (x > 2).mean()),
                   median_pinning=("pinning", "median"))
              .reindex(["zero", *LABELS]).dropna(how="all"))
        lines += ["", "MAP residual (allocation - estimate) / SE, by k_hat (published SE; "
                      "pinning = (Sigma zeta)_i / (diag Sigma_i zeta_i)):"]
        lines += block(by.round(3))
    top = withl.nlargest(args.top, "abs_lam").copy()
    top["cause"] = [cause(r) for _, r in top.iterrows()]
    summary_rows, pairs = [], []
    for puma, g in top.groupby("puma"):
        inputs = PMEDMInputs.load(root / puma)
        frame = withl[withl["puma"] == puma]
        positions = g.index.to_numpy() - frame.index.min()
        s, p = clusters(inputs, frame.reset_index(drop=True), positions)
        for row, index in zip(s, g.index):
            summary_rows.append({"index": index, **{k: v for k, v in row.items() if k != "position"}})
        pairs += p
    if summary_rows:
        found = pd.DataFrame(summary_rows).set_index("index")
        top = top.join(found)
    else:
        top["max_abs_r"], top["n_r95"], top["r95_tables"] = np.nan, 0, ""
    pd.DataFrame(pairs).to_csv(out / "clusters.csv", index=False)
    columns = ["puma", "level", "area", "constraint", "estimate", "se", "k_hat", "allocation",
               "resid_se", "map_lam", "map_zeta", "map_lam_puma", "pinning", "support_units",
               "max_abs_r", "n_r95", "cause"]
    clustered = (top["n_r95"].fillna(0) > 0)
    lines += ["", f"the {len(top)} largest |MAP multiplier|, with a heuristic cause "
                  "(pinned: |resid| <= 2 SE and pinning < 0.5; puma: a tract row whose "
                  "lambda_P part is larger; miss: |resid| > 2 SE):",
              "  causes: " + ", ".join(f"{k} {v}" for k, v in top["cause"].value_counts().items()),
              f"  in a replicate-correlation cluster (|r| > 0.95 with another cell of the PUMA): "
              f"{int(clustered.sum())} of {len(top)}"]
    lines += block(top[columns].round(3), index=False)
    if puma_frames:
        prows = pd.concat(puma_frames, ignore_index=True)
        prows.to_csv(out / "puma_rows.csv", index=False)
        worst = prows.reindex(prows["map_lam_puma"].abs().sort_values(ascending=False).index).head(15)
        lines += ["", "PUMA rows, fifteen largest |lambda_P| (support_units: units carrying "
                      "the category):"]
        lines += block(worst.round(3), index=False)
    return lines


def pullout_summary(pull: pd.DataFrame, thresholds: list[float]) -> list[str]:
    lines = ["", "Cells a cell-by-cell Poisson model would pull out of the Gaussian hierarchy "
                 "(finest level: block group, or tract for tract-only categories):"]
    for (column, c), g in pull.groupby(["k_hat", "threshold"], sort=False):
        cells, pulled = g["cells"].sum(), g["zero"].sum() + g["low"].sum()
        lines += ["", f"  {column} < {c:g}: {pulled:,} of {cells:,} finest cells pulled "
                      f"({pulled / cells:.1%}); effective dimension {cells:,} -> "
                      f"{g['rows_after'].sum():,} after merging zeros"]
        by_table = g.groupby("table")[["cells", "zero", "low", "kept", "rows_after"]].sum()
        lines += block(by_table)
        classes = g.groupby(["table", "class"]).size().unstack(fill_value=0)
        lines += ["  groups (tract x category) by class:"] + block(classes)
    return lines


def no_support_summary(cells: pd.DataFrame) -> list[str]:
    zero = cells[cells["estimate"] == 0]
    if not len(zero):
        return []
    none = zero[zero["support_units"] == 0]
    lines = ["", f"Zero cells whose category no support unit carries (allocation exactly zero): "
                 f"{len(none):,} of {len(zero):,} zero cells"]
    if len(none):
        table = (zero.assign(no_support=zero["support_units"] == 0)
                 .groupby(["level", "table"])["no_support"].agg(["size", "sum"])
                 .rename(columns={"size": "zero cells", "sum": "no support"}))
        lines += block(table)
    return lines


if __name__ == "__main__":
    main()
