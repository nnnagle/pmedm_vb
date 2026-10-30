"""Score a simulation's tabulation (``sim_collect.py``) against the truth.

Per cell -- a published line in one block group or tract, of one truth -- and
method, in units of the cell's ``se`` (the square root of ``c`` times its
published variance):

- **error**: estimate - truth, the estimate being a point method's value or the
  mean of a method's draws;
- **CRPS**: the continuous ranked probability score of the draws as an
  ensemble, ``mean|X - T| - mean|X - X'| / 2`` (Gneiting & Raftery 2007,
  *JASA* 102); for a point method it is ``|estimate - truth|``, so point and
  sampled methods are on one scale; lower is better;
- **covered**: the truth between the smallest and largest draw, an interval
  that holds a new draw from the same distribution with probability
  ``(m - 1) / (m + 1)`` (15/17 for 16 draws);
- **interval**: the interval score of that interval at its level, ``alpha =
  2 / (m + 1)``: ``(max - min) + (2 / alpha) (min - T)+ + (2 / alpha) (T -
  max)+`` (Gneiting & Raftery 2007); a point method's interval has zero width,
  so its score is ``(2 / alpha) |estimate - truth|``; lower is better, and a
  method that knew the truth would score 0;
- **rank**: the truth's rank among the draws, 0 to ``m``, ties (common at zero)
  broken at random: uniform if the draws and the truth come from one
  distribution.

Point methods: ``survey`` (the direct estimate) and ``ipf``. Sampled:
``ipf_multinomial``, ``vb_gaussian``, ``vb_gaussian_trunc``, ``hmc_short``, and
the last three as integer populations, ``vb_gaussian_int``,
``vb_gaussian_trunc_int`` and ``hmc_short_int`` (``sim_collect.py``). Leaf cells only unless
``--subtotals``. Writes, under ``<output>/scores``:

``cells.parquet``
    Every cell's scores, one row per (kind, truth_id, level, geoid, table,
    line), with its role.
``summary.csv``, ``summary.md``
    By truth kind, level, role (``--by-table``: and ACS table) and whether the
    truth is zero (``truth_zero``: a sampled method's draws, N p, are never
    exactly zero, so a zero truth is never inside them): cells; per
    method the median and 90th percentile of |error|, mean CRPS, mean interval
    score, coverage and
    the share of ranks in the two outer bins (nominally ``2 / (m + 1)``).
``ranks.csv``
    The rank histograms: share of cells per rank, by kind, level, role,
    ``truth_zero`` and method, and over all cells (``truth_zero`` ``all``):
    selecting cells by their truth skews a rank histogram even for a calibrated
    method, so calibration is read from ``all``, with the integer draws.
``table_sim_<kind>.tex``, ``table_sim.csv``
    The supplement's tables: per truth kind, a bare booktabs ``tabular`` of the
    nonzero truths (block group, then tract; rows the roles), with median
    |error| and CRPS for IPF, IPF multinomial, VB Gaussian, VB Gaussian
    truncated and HMC short (the last three's integer draws), and interval
    score and coverage for the sampled four (IPF, a point, has no interval);
    and every value of those tables, zero truths and the survey too, in one
    long CSV.
``usability.csv``
    From ``draws.parquet``: per kind and method, over draws and truths, the
    median smallest D/H and smallest n_eff, each over the truth's own (median
    of the per-draw ratio), the share of draws whose smallest D/H is below
    ``--dh-floor``, and the median largest share::

    $CONDA_PREFIX/bin/python experiments/sim_score.py --sim experiments/sim/sim01.json --root $R
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))

from provenance import SIM_SCORE_SCRIPTS, write_stamp  # noqa: E402

POINT = ("survey", "ipf")
SAMPLED = ("ipf_multinomial", "vb_gaussian", "vb_gaussian_trunc", "hmc_short", "vb_gaussian_int",
           "vb_gaussian_trunc_int", "hmc_short_int")
KEYS = ["level", "geoid", "table", "line"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, help="experiments/sim/<name>.json (with --root)")
    parser.add_argument("--root", type=Path, help="the experiments' folder")
    parser.add_argument("--output", type=Path, default=None,
                        help="sim_collect.py's output folder (default from --sim and --root)")
    parser.add_argument("--subtotals", action="store_true", help="score subtotals too")
    parser.add_argument("--by-table", action="store_true", help="summaries by ACS table too")
    parser.add_argument("--dh-floor", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0, help="for breaking rank ties")
    args = parser.parse_args()
    if args.output is None:
        if args.sim is None or args.root is None:
            parser.error("give --output, or --sim and --root")
        from sim_common import Simulation

        args.output = Simulation(args.sim, args.root).folder / "output"
    return args


def ensemble_scores(X: np.ndarray, T: np.ndarray, rng) -> dict[str, np.ndarray]:
    """``X`` (cells, m) draws, ``T`` (cells,) truths: mean, CRPS, covered, rank, width,
    and the interval score of the draws' range."""
    m = X.shape[1]
    alpha = 2 / (m + 1)
    Xs = np.sort(X, axis=1)
    spread = (2 * np.arange(1, m + 1) - m - 1) @ Xs.T * 2 / m**2  # mean |X - X'|
    below = (X < T[:, None]).sum(axis=1)
    ties = (X == T[:, None]).sum(axis=1)
    return {"mean": X.mean(axis=1),
            "crps": np.abs(X - T[:, None]).mean(axis=1) - spread / 2,
            "covered": (Xs[:, 0] <= T) & (T <= Xs[:, -1]),
            "rank": below + np.floor(rng.random(len(T)) * (ties + 1)).astype(int),
            "width": Xs[:, -1] - Xs[:, 0],
            "interval": (Xs[:, -1] - Xs[:, 0]) + 2 / alpha * (np.maximum(Xs[:, 0] - T, 0)
                                                              + np.maximum(T - Xs[:, -1], 0))}


def truth_scores(frame: pd.DataFrame, rng) -> tuple[pd.DataFrame, int]:
    """One truth's cells, scored; the number of draws."""
    m = int(frame["draw"].max())
    frame = frame.sort_values(KEYS + ["draw"], kind="stable")
    counts = frame.groupby(KEYS, observed=True).size()
    if not (counts == m).all():
        raise SystemExit(f"cells without {m} draws: {counts[counts != m].head()}")
    first = frame[frame["draw"] == 1].reset_index(drop=True)
    T = first["truth"].to_numpy(float)
    se = first["se"].to_numpy(float)
    out = first[KEYS + ["truth", "se"]].copy()
    for method in POINT:
        value = first[method].to_numpy(float)
        out[f"{method}_error"] = (value - T) / se
        out[f"{method}_crps"] = np.abs(value - T) / se
        out[f"{method}_interval"] = (m + 1) * np.abs(value - T) / se  # 2 / alpha
    for method in SAMPLED:
        X = frame[method].to_numpy(float).reshape(-1, m)
        if np.isnan(X).all():
            continue
        s = ensemble_scores(X, T, rng)
        out[f"{method}_error"] = (s["mean"] - T) / se
        out[f"{method}_crps"] = s["crps"] / se
        out[f"{method}_interval"] = s["interval"] / se
        out[f"{method}_covered"] = s["covered"]
        out[f"{method}_rank"] = s["rank"]
        out[f"{method}_width"] = s["width"] / se
    return out, m


def summarise(cells: pd.DataFrame, by: list[str], m: int) -> pd.DataFrame:
    rows = []
    for key, group in cells.groupby(by, observed=True):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)), cells=len(group))
        for method in POINT + SAMPLED:
            if f"{method}_error" not in group:
                continue
            error = group[f"{method}_error"].abs()
            row[f"{method}_abs_error_p50"] = error.median()
            row[f"{method}_abs_error_p90"] = error.quantile(0.9)
            row[f"{method}_crps"] = group[f"{method}_crps"].mean()
            row[f"{method}_interval"] = group[f"{method}_interval"].mean()
            if f"{method}_rank" in group:
                rank = group[f"{method}_rank"]
                row[f"{method}_covered"] = group[f"{method}_covered"].mean()
                row[f"{method}_outer_ranks"] = ((rank == 0) | (rank == m)).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def rank_histograms(cells: pd.DataFrame, m: int) -> pd.DataFrame:
    frames = []
    for method in SAMPLED:
        if f"{method}_rank" not in cells:
            continue
        for split in (True, False):
            by = ["kind", "level", "role"] + (["truth_zero"] if split else [])
            share = (cells.groupby(by, observed=True)[f"{method}_rank"]
                     .value_counts(normalize=True).rename("share").reset_index()
                     .rename(columns={f"{method}_rank": "rank"}))
            if not split:
                share["truth_zero"] = "all"
            frames.append(share.assign(method=method))
    out = pd.concat(frames, ignore_index=True)
    out["truth_zero"] = out["truth_zero"].astype(str)
    out["nominal"] = 1 / (m + 1)
    return out.sort_values(["kind", "level", "role", "truth_zero", "method", "rank"])


def usability(draws: pd.DataFrame, floor: float) -> pd.DataFrame:
    truth = draws[draws.method == "truth"].set_index(["kind", "truth_id"])
    rows = []
    for (kind, method), group in draws.groupby(["kind", "method"]):
        own = truth.loc[list(zip(group.kind, group.truth_id))]
        rows.append(dict(
            kind=kind, method=method, draws=len(group),
            dh_min_p50=group.dh_min.median(),
            dh_min_over_truth_p50=float(np.median(group.dh_min.to_numpy()
                                                  / own.dh_min.to_numpy())),
            neff_min_p50=group.neff_min.median(),
            neff_min_over_truth_p50=float(np.median(group.neff_min.to_numpy()
                                                    / own.neff_min.to_numpy())),
            dh_below_floor=float((group.dh_min < floor).mean()),
            max_share_p50=group.max_share.median()))
    order = {"truth": 0, "ipf": 1, **{m: 2 + i for i, m in enumerate(SAMPLED)}}
    return pd.DataFrame(rows).sort_values(["kind", "method"], key=lambda s: s.map(
        lambda v: order.get(v, 99)) if s.name == "method" else s)


#: The supplement tables' rows, columns and labels.
ROLES = {"constraint": "Fitted", "dropped_line": "Unfitted lines (C24010 detail)",
         "other_table": "Unfitted tables (B19001, B23001)", "heldout": "Held-out tables"}
MEASURES = {"abs_error_p50": "Median $|$error$|$", "crps": "CRPS", "interval": "Interval score",
            "covered": "Coverage"}
TABLE_METHODS = {"ipf": "IPF", "ipf_multinomial": "IPF m.", "vb_gaussian_int": "VB G",
                 "vb_gaussian_trunc_int": "VB G tr.", "hmc_short_int": "HMC s."}
#: Each measure's methods: IPF is a point, so it has no interval and no coverage.
MEASURE_METHODS = {measure: [m for m in TABLE_METHODS
                             if m != "ipf" or measure in ("abs_error_p50", "crps")]
                   for measure in MEASURES}


def long_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Every (kind, level, role, truth_zero, method, measure) value of ``summary``."""
    keys = ["kind", "level", "role", "truth_zero", "cells"]
    frame = summary.melt(id_vars=keys, var_name="column", value_name="value").dropna()
    for measure in MEASURES:
        hit = frame.column.str.endswith("_" + measure)
        frame.loc[hit, "measure"] = measure
        frame.loc[hit, "method"] = frame.loc[hit, "column"].str[:-len(measure) - 1]
    return frame.dropna(subset=["measure"]).drop(columns="column")[
        keys + ["method", "measure", "value"]]


def tabular(summary: pd.DataFrame, kind: str) -> str:
    """The nonzero truths of ``kind`` as a bare booktabs ``tabular``."""
    part = summary[(summary.kind == kind) & ~summary.truth_zero.astype(bool)]
    widths = [len(MEASURE_METHODS[measure]) for measure in MEASURES]
    total = sum(widths)
    starts = [2 + sum(widths[:i]) for i in range(len(widths))]
    lines = ["\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{l" + "r" * total + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{label}}}"
                                for n, label in zip(widths, MEASURES.values())) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{a}-{a + n - 1}}}" for a, n in zip(starts, widths)),
             "Cells & " + " & ".join(TABLE_METHODS[m] for measure in MEASURES
                                     for m in MEASURE_METHODS[measure]) + " \\\\",
             "\\midrule"]
    for i, level in enumerate(("block group", "tract")):
        if i:
            lines.append("\\addlinespace")
        lines.append(f"\\multicolumn{{{1 + total}}}{{l}}"
                     f"{{\\textit{{{level.capitalize()}}}}} \\\\")
        for role, label in ROLES.items():
            row = part[(part.level == level) & (part.role == role)]
            if row.empty:
                continue
            row = row.iloc[0]
            values = [f"{row[f'{m}_{measure}']:.2f}" if pd.notna(row.get(f"{m}_{measure}"))
                      else "--" for measure in MEASURES for m in MEASURE_METHODS[measure]]
            lines.append(f"{label} & " + " & ".join(values) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def markdown(frame: pd.DataFrame, title: str) -> str:
    cols = list(frame.columns)
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(f"{v:.3g}" if isinstance(v, float) else str(v)
                                       for v in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out = args.output / "scores"
    out.mkdir(parents=True, exist_ok=True)
    roles = pd.read_parquet(args.output / "cells.parquet")
    dataset = ds.dataset(args.output / "estimates", partitioning="hive")
    parts = sorted({(f.split("kind=")[1].split("/")[0], int(f.split("truth_id=")[1].split("/")[0]))
                    for f in dataset.files})
    rng = np.random.default_rng(args.seed)
    scored, m = [], None
    for kind, truth_id in parts:
        frame = dataset.to_table(filter=(ds.field("kind") == kind)
                                 & (ds.field("truth_id") == truth_id)).to_pandas()
        frame = frame.drop(columns=["kind", "truth_id"])
        for column in KEYS:
            frame[column] = frame[column].astype(str)
        cells, m = truth_scores(frame, rng)
        scored.append(cells.assign(kind=kind, truth_id=truth_id))
        print(f"{kind} {truth_id:02d}: {len(cells):,} cells", flush=True)
    cells = pd.concat(scored, ignore_index=True)
    cells = cells.merge(roles[["level", "table", "line", "subtotal", "role"]],
                        on=["level", "table", "line"], how="left")
    if not args.subtotals:
        cells = cells[~cells.subtotal]
    cells["truth_zero"] = cells["truth"] == 0
    cells.to_parquet(out / "cells.parquet", index=False)

    by = ["kind", "level", "role"] + (["table"] if args.by_table else []) + ["truth_zero"]
    summary = summarise(cells, by, m)
    summary.to_csv(out / "summary.csv", index=False)
    ranks = rank_histograms(cells, m)
    ranks.to_csv(out / "ranks.csv", index=False)
    use = usability(pd.read_parquet(args.output / "draws.parquet"), args.dh_floor)
    use.to_csv(out / "usability.csv", index=False)
    base = summarise(cells, ["kind", "level", "role", "truth_zero"], m)
    long_table(base).to_csv(out / "table_sim.csv", index=False)
    for kind in base.kind.unique():
        (out / f"table_sim_{kind}.tex").write_text(tabular(base, kind))

    text = [f"# Simulation scores ({'all lines' if args.subtotals else 'leaf cells'}; "
            f"{m} draws; errors and CRPS in units of se; nominal coverage "
            f"{(m - 1) / (m + 1):.3f}, nominal outer-rank share {2 / (m + 1):.3f})", ""]
    for measure in ("abs_error_p50", "abs_error_p90", "crps", "interval", "covered",
                    "outer_ranks"):
        columns = [c for c in summary.columns if c.endswith(measure)]
        table = summary[by + ["cells"] + columns].rename(
            columns={c: c[:-len(measure) - 1] for c in columns})
        text.append(markdown(table, measure))
    text.append(markdown(use, "usability of the draws"))
    (out / "summary.md").write_text("\n".join(text))
    # Last, so that a run cut short leaves the scores unstamped (run_sim.py).
    write_stamp(out / "summary.md", SIM_SCORE_SCRIPTS, subtotals=args.subtotals,
                by_table=args.by_table, dh_floor=args.dh_floor, seed=args.seed)
    print("\n".join(text))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
