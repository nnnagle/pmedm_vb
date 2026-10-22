"""Why does IPF stop short of the block group margins?

IPF (``pmedm_vb.rake.rake_ipf``) fits block group cells only -- with a
roll-up, the roll-up's cells, each a block group's sum of one or more published
categories -- and each cell involves only its own block group's weights, so
whether the margins can be met at all splits into one question per block group.
For each experiment and PUMA, on the cells IPF fitted:

1. **No carrier.** A published nonzero cell no record of the PUMA carries.
2. **Carriers removed by zeros.** IPF meets a published zero exactly, so every
   record carrying that cell gets no weight in the block group; a nonzero cell
   all of whose carriers are so removed cannot be met.
3. **Same-universe totals.** Two tables whose cells give every record the same
   total (e.g. two tables of total population) measure one quantity; where
   their published block group totals differ, no weights meet both. This checks
   the published tables' consistency on the data.
4. **Exact attainability** (``scipy.optimize.linprog``, HiGHS). Per block group:
   do nonnegative weights on the records (where the design allows them there)
   meet every fitted cell exactly? If not, the least total relative relaxation,
   sum |slack| / max(Y, 1), and the cells it relaxes. This decides the question
   whatever the cause, including combinations of records that cannot add up to
   the margins although every cell has carriers. The relaxed cells are one
   minimal set; another may exist.

Each of IPF's cells missed by more than 5% (|log(m / Y)| > log 1.05, its saved
residual) is tagged with these. Writes ``<out>/ipf_diagnosis.txt`` and
``<out>/ipf_cells.csv`` (every fitted cell, its tags, LP slack and residual)::

    PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data \\
    $CONDA_PREFIX/bin/python experiments/ipf_diagnosis.py --root $RUNS/paper_v3 \\
        --out $RUNS/paper_v3/ipf_diagnosis
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import area_slug, rake_output  # noqa: E402

from pmedm_vb.assemble.inputs import PMEDMInputs  # noqa: E402
from pmedm_vb.config import processed_dir  # noqa: E402
from pmedm_vb.rake import _rolled  # noqa: E402

#: A miss IPF is charged with: |log(m / Y)| over log of this.
MISS = 1.05
#: An LP optimum under this counts as exactly attainable.
ATTAINABLE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--experiments", nargs="+",
                        default=["exp01_baseline", "exp02_nullspace", "exp03_rollup"])
    parser.add_argument("--pumas", nargs="+", default=None, help="default: each experiment's")
    parser.add_argument("--top", type=int, default=15, help="worst IPF cells listed")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def fitted_cells(inputs: PMEDMInputs, rollup) -> pd.DataFrame:
    """IPF's block group cells: index into its residual, zone, the categories it
    sums, its table, name and published target."""
    n_z = inputs.Y_B.shape[0]
    split = inputs.Y_T.size
    stacked = split + np.arange(inputs.Y_B.size)
    category, zone = (stacked - split) // n_z, (stacked - split) % n_z
    if rollup is None:
        return pd.DataFrame({
            "row": stacked, "zone": zone, "categories": [[int(k)] for k in category],
            "table": [inputs.bg_constraints[k].split(".")[0] for k in category],
            "cell": [inputs.bg_constraints[k] for k in category],
            "published": inputs.Y_B[zone, category]})
    h = _rolled(inputs, rollup)
    members: dict[int, list[int]] = {}
    where: dict[int, int] = {}
    for c, k, z in zip(h.row_cell[stacked], category, zone):
        if c >= 0:
            members.setdefault(int(c), []).append(int(k))
            where[int(c)] = int(z)
    cells = sorted(members)
    return pd.DataFrame({
        "row": cells, "zone": [where[c] for c in cells], "categories": [members[c] for c in cells],
        "table": [inputs.bg_constraints[members[c][0]].split(".")[0] for c in cells],
        "cell": [h.cell_names[c] for c in cells], "published": h.Y_c[cells].astype(float)})


def loadings(inputs: PMEDMInputs, cells: pd.DataFrame) -> dict[tuple, np.ndarray]:
    """Each distinct category set's loading over the records."""
    X = sp.csc_matrix(inputs.X_B)
    out = {}
    for cats in cells.categories:
        key = tuple(cats)
        if key not in out:
            out[key] = np.asarray(X[:, list(cats)].sum(axis=1)).ravel()
    return out


def universe_conflicts(cells: pd.DataFrame, x: dict[tuple, np.ndarray]) -> tuple[list, set]:
    """Same-universe table pairs, their published total differences, and the
    (zone, table) pairs in conflict."""
    tables = sorted(cells.table.unique())
    totals = {}
    for t in tables:
        part = cells[cells.table == t]
        keys = list(dict.fromkeys(tuple(c) for c in part.categories))
        totals[t] = sum(x[k] for k in keys)
    lines, bad = [], set()
    for a, b in combinations(tables, 2):
        if not np.allclose(totals[a], totals[b]):
            continue
        pa = cells[cells.table == a].groupby("zone").published.sum()
        pb = cells[cells.table == b].groupby("zone").published.sum()
        diff = (pa - pb).dropna()
        off = diff[np.abs(diff) > 0.5]
        bad |= {(z, a) for z in off.index} | {(z, b) for z in off.index}
        lines.append(f"     {a} vs {b}: totals differ in {off.size} of {diff.size} block groups"
                     + (f"; |difference| median {np.median(np.abs(off)):.1f}, max "
                        f"{np.abs(off).max():.1f}" if off.size else ""))
    return lines or ["     none"], bad


def attainability(q_row: np.ndarray, part: pd.DataFrame, x: dict[tuple, np.ndarray]
                  ) -> tuple[float, np.ndarray]:
    """One block group's least relative relaxation and each cell's slack."""
    allowed = np.flatnonzero(q_row > 0)
    A = np.vstack([x[tuple(c)][allowed] for c in part.categories])
    m, n = A.shape
    scale = np.maximum(part.published.to_numpy(), 1.0)
    A_eq = sp.hstack([sp.csr_matrix(A), sp.identity(m), -sp.identity(m)]).tocsr()
    cost = np.concatenate([np.zeros(n), 1 / scale, 1 / scale])
    res = linprog(cost, A_eq=A_eq, b_eq=part.published.to_numpy(), bounds=(0, None),
                  method="highs")
    if not res.success:
        raise RuntimeError(f"linprog: {res.message}")
    return float(res.fun), res.x[n:n + m] - res.x[n + m:]


def diagnose(name: str, spec: dict, exp: Path, puma: str, top: int
             ) -> tuple[list[str], pd.DataFrame]:
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / area_slug(spec["area"]) / puma)
    rollup = spec["model"].get("rollup")
    cells = fitted_cells(inputs, rollup)
    x = loadings(inputs, cells)
    lines = [f"== {name}, PUMA {puma}: {cells.zone.nunique()} block groups, {len(cells):,} "
             f"fitted cells" + (f" (roll-up {rollup})" if rollup else "")]

    # 1-2. Carriers, before and after the zeros.
    carriers = np.array([(x[tuple(c)] > 0).sum() for c in cells.categories])
    alive = np.zeros(len(cells), int)
    for z, part in cells.groupby("zone"):
        dead = np.zeros(inputs.n_units, bool)
        for cats in part[part.published == 0].categories:
            dead |= x[tuple(cats)] > 0
        dead |= inputs.q[z] <= 0
        for i in part.index:
            alive[i] = ((x[tuple(cells.categories[i])] > 0) & ~dead).sum()
    nonzero = cells.published > 0
    cells["carriers"], cells["carriers_after_zeros"] = carriers, alive
    cells["no_carrier"] = nonzero & (carriers == 0)
    cells["killed_by_zeros"] = nonzero & (carriers > 0) & (alive == 0)
    lines.append(f"   nonzero cells with no carrier: {int(cells.no_carrier.sum())}; with every "
                 f"carrier removed by the block group's zeros: {int(cells.killed_by_zeros.sum())};"
                 f" with 1-2 carriers left: {int((nonzero & alive.astype(bool) & (alive <= 2)).sum())}")

    # 3. Same-universe totals.
    lines.append("   tables with the same universe, and their published block group totals:")
    pair_lines, bad = universe_conflicts(cells, x)
    lines += pair_lines
    cells["universe_conflict"] = [(z, t) in bad for z, t in zip(cells.zone, cells.table)]

    # 4. Exact attainability, block group by block group.
    cells["lp_slack"] = 0.0
    relaxation = {}
    for z, part in cells.groupby("zone"):
        total, slack = attainability(inputs.q[z], part, x)
        relaxation[z] = total
        cells.loc[part.index, "lp_slack"] = slack
    cells["lp_relaxed"] = np.abs(cells.lp_slack) > 1e-6 * np.maximum(cells.published, 1)
    stuck = {z for z, v in relaxation.items() if v > ATTAINABLE}
    lines.append(f"   exactly attainable block groups: {len(relaxation) - len(stuck)} of "
                 f"{len(relaxation)}; the others need {int(cells.lp_relaxed.sum())} cells relaxed, "
                 f"least total relative relaxation median "
                 f"{np.median([relaxation[z] for z in stuck]) if stuck else 0:.3g}, max "
                 f"{max([relaxation[z] for z in stuck], default=0):.3g}")

    # IPF's misses, attributed.
    with np.load(rake_output(spec, exp, puma, "ipf")) as saved:
        cells["log_residual"] = saved["residual"][cells.row.to_numpy()]
        converged, sweeps = bool(saved["converged"]), int(saved["n_sweeps"])
    missed = cells.log_residual.abs() > np.log(MISS)
    lines.append(f"   IPF: converged {converged} after {sweeps} sweeps; {int(missed.sum())} of "
                 f"{int(cells.log_residual.notna().sum())} fitted cells miss by more than "
                 f"{MISS - 1:.0%}")
    if missed.any():
        m = cells[missed]
        explained = m.no_carrier | m.killed_by_zeros | m.universe_conflict | m.lp_relaxed
        in_stuck = m.zone.isin(stuck)
        lines.append(
            f"     of those: no carrier {int(m.no_carrier.sum())}, carriers removed by zeros "
            f"{int(m.killed_by_zeros.sum())}, universe conflict {int(m.universe_conflict.sum())}, "
            f"relaxed by the LP {int(m.lp_relaxed.sum())}, in a block group that cannot be met "
            f"exactly {int(in_stuck.sum())}, none of these {int((~explained & ~in_stuck).sum())}")
    worst = cells[cells.log_residual.notna()].reindex(
        cells.log_residual.abs().sort_values(ascending=False).index).head(top)
    worst = worst.assign(block_group=inputs.block_groups.iloc[worst.zone, 0].astype(str).to_numpy())
    lines += ["     largest misses |log(m/Y)|:"] + ["       " + r for r in worst[
        ["block_group", "cell", "published", "carriers", "carriers_after_zeros",
         "universe_conflict", "lp_slack", "log_residual"]].round(3).to_string(index=False)
        .splitlines()]
    lines.append("")
    cells["block_group"] = inputs.block_groups.iloc[cells.zone, 0].astype(str).to_numpy()
    return lines, cells.drop(columns="categories").assign(experiment=name, puma=puma)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lines, frames = [], []
    for name in args.experiments:
        exp = args.root / name
        spec = json.loads((exp / "experiment.json").read_text())
        for puma in args.pumas or spec["pumas"]:
            block, rows = diagnose(name, spec, exp, puma, args.top)
            print("\n".join(block), flush=True)
            lines += block
            frames.append(rows)
    pd.concat(frames).to_csv(args.out / "ipf_cells.csv", index=False)
    (args.out / "ipf_diagnosis.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
