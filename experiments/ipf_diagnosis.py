"""Why does hard IPF stall on the block group margins?

``rake_ipf`` fits block group margins only, so disagreement between levels
cannot be the cause. Two candidates are checked on the assembled inputs, per
PUMA, and IPF's final residuals are attributed to them:

1. **Shared-universe totals.** Two block group tables whose loadings give every
   unit the same row total -- e.g. B01001 and B03002, both total population --
   measure one quantity, ``sum_k m_k``, as a function of ``W``. Where their
   published totals differ in a block group, no ``W`` meets both exactly.
2. **Zero cells.** Hard raking to a published zero drives every unit carrying
   that category to zero weight in that block group. A nonzero cell whose
   carriers there are all killed this way cannot be met.

With ``--rake-run`` (a ``run_map.py rake --method ipf`` results directory per
PUMA, in the same order as ``--puma``), IPF's saved per-row residual
``log(m / Y)`` is attributed: each block group row is tagged with whether its
table is in a conflicting universe pair in that block group, and with how many
of its carriers survive the zero cells; the largest residuals are listed.

Writes ``<out>/ipf_diagnosis.txt`` and ``<out>/ipf_rows.csv`` (every block
group row with its tags and residual)::

    $CONDA_PREFIX/bin/python experiments/ipf_diagnosis.py \\
        --puma 4701501 4701502 --rake-run $RUNS/<ipf job 1> $RUNS/<ipf job 2> \\
        --out $RUNS/ipf_diagnosis
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--puma", nargs="+", required=True)
    parser.add_argument("--rake-run", type=Path, nargs="*", default=[],
                        help="IPF results directory per PUMA, same order as --puma")
    parser.add_argument("--area", default="knox-2024-5yr")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def universe_pairs(inputs: PMEDMInputs) -> list[tuple[str, str]]:
    """Block group table pairs whose loadings have identical unit row totals."""
    X = sp.csc_matrix(inputs.X_B)
    tables = pd.Series([n.split(".")[0] for n in inputs.bg_constraints])
    totals = {t: np.asarray(X[:, np.flatnonzero(tables == t)].sum(axis=1)).ravel()
              for t in tables.unique()}
    return [(a, b) for a, b in combinations(totals, 2) if np.allclose(totals[a], totals[b])]


def diagnose(inputs: PMEDMInputs, rake_run: Path | None) -> tuple[list[str], pd.DataFrame]:
    n_zones, c_b = inputs.Y_B.shape
    split = inputs.Y_T.size
    tables = np.array([n.split(".")[0] for n in inputs.bg_constraints])
    Y = inputs.Y_B
    X = sp.csr_matrix(inputs.X_B)
    lines = [f"== PUMA {inputs.puma}: {n_zones} block groups, {c_b} block group categories "
             f"in {len(set(tables))} tables"]

    # 1. Shared-universe totals.
    pairs = universe_pairs(inputs)
    conflict = np.zeros((n_zones, c_b), bool)
    lines.append("   tables with identical unit totals (one universe), and their published "
                 "block group totals:")
    if not pairs:
        lines.append("     none")
    for a, b in pairs:
        ta, tb = Y[:, tables == a].sum(axis=1), Y[:, tables == b].sum(axis=1)
        diff = ta - tb
        bad = ~np.isclose(ta, tb, rtol=1e-9, atol=0.5)
        conflict[np.ix_(bad, (tables == a) | (tables == b))] = True
        rel = np.abs(diff) / np.maximum(np.maximum(ta, tb), 1)
        lines.append(
            f"     {a} vs {b}: totals differ in {int(bad.sum())} of {n_zones} block groups; "
            f"|difference| median {np.median(np.abs(diff)):.1f}, max {np.abs(diff).max():.1f} "
            f"people; relative median {np.median(rel):.3f}, max {rel.max():.3f}")

    # 2. Zero cells: which units raking to zero kills in each block group, and
    # how many carriers each nonzero cell keeps.
    Xd = X.toarray() > 0  # (units, c_b) carries the category
    alive_carriers = np.zeros((n_zones, c_b), int)
    carriers = np.zeros((n_zones, c_b), int)
    live_q = inputs.q > 0
    for z in range(n_zones):
        zero = Y[z] == 0
        dead = Xd[:, zero].any(axis=1) | ~live_q[z]
        carriers[z] = (Xd & live_q[z][:, None]).sum(axis=0)
        alive_carriers[z] = (Xd & ~dead[:, None]).sum(axis=0)
    nonzero = Y > 0
    killed = nonzero & (alive_carriers == 0)
    lines.append(
        f"   zero cells: {int((Y == 0).sum()):,} of {Y.size:,} block group cells are published 0; "
        f"raking them to 0 removes every carrier of {int(killed.sum()):,} nonzero cells "
        f"(in {int(killed.any(axis=1).sum())} block groups), and leaves 1-2 carriers for "
        f"{int((nonzero & (alive_carriers > 0) & (alive_carriers <= 2)).sum()):,} more")
    by_table = pd.Series(killed.sum(axis=0), index=tables).groupby(level=0).sum()
    by_table = by_table[by_table > 0].sort_values(ascending=False)
    if len(by_table):
        lines.append("     nonzero cells left without carriers, by table: "
                     + ", ".join(f"{t} {n}" for t, n in by_table.items()))

    rows = pd.DataFrame({
        "puma": inputs.puma,
        "block_group": np.tile(inputs.block_groups.iloc[:, 0].astype(str).to_numpy(), c_b),
        "category": np.repeat(inputs.bg_constraints, n_zones),
        "published": Y.ravel(order="F"),
        "carriers": carriers.ravel(order="F"),
        "carriers_after_zeros": alive_carriers.ravel(order="F"),
        "universe_conflict": conflict.ravel(order="F"),
    })
    rows["killed_by_zeros"] = (rows.published > 0) & (rows.carriers_after_zeros == 0)

    # 3. Attribute IPF's final residuals.
    if rake_run is not None:
        path = rake_run / f"{inputs.puma}_ipf.npz"
        with np.load(path) as saved:
            residual = saved["residual"][split:]
            converged, sweeps = bool(saved["converged"]), int(saved["n_sweeps"])
        rows["log_residual"] = residual
        fitted = rows.log_residual.notna()
        big = fitted & (rows.log_residual.abs() > np.log(1.05))
        lines.append(f"   IPF ({path.parent.name}): converged {converged} after {sweeps} sweeps; "
                     f"{int(big.sum())} of {int(fitted.sum())} fitted rows miss by more than 5%")
        if big.any():
            tags = rows[big]
            lines.append(
                f"     of those: in a conflicting universe pair {int(tags.universe_conflict.sum())}, "
                f"all carriers killed by zeros {int(tags.killed_by_zeros.sum())}, "
                f"1-2 carriers left {int(((tags.carriers_after_zeros > 0) & (tags.carriers_after_zeros <= 2)).sum())}, "
                f"none of these {int((~tags.universe_conflict & (tags.carriers_after_zeros > 2)).sum())}")
        worst = rows[fitted].reindex(rows[fitted].log_residual.abs()
                                     .sort_values(ascending=False).index).head(20)
        lines += ["     largest residuals |log(m/Y)|:"] + ["       " + r for r in worst[
            ["block_group", "category", "published", "carriers", "carriers_after_zeros",
             "universe_conflict", "log_residual"]].to_string(index=False).splitlines()]
    lines.append("")
    return lines, rows


def main() -> None:
    args = parse_args()
    if args.rake_run and len(args.rake_run) != len(args.puma):
        raise SystemExit("give one --rake-run per --puma, in the same order")
    args.out.mkdir(parents=True, exist_ok=True)
    lines, frames = [], []
    for i, puma in enumerate(args.puma):
        inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / puma)
        block, rows = diagnose(inputs, args.rake_run[i] if args.rake_run else None)
        lines += block
        frames.append(rows)
    pd.concat(frames).to_csv(args.out / "ipf_rows.csv", index=False)
    text = "\n".join(lines) + "\n"
    (args.out / "ipf_diagnosis.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
