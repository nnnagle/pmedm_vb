"""Count a design's constraint cells per area under three aggregation schemes.

For each scheme, the cells per PUMA (the ``puma`` hierarchy's PUMA rows: the
tract categories summed over the PUMA's tracts), per tract and per block group,
averaged over the areas of every PUMA given, split into count cells (a
published estimate above zero) and zero cells; and the same per table.

- **all** -- every published cell (:mod:`pmedm_vb.assemble.allcells`), from the
  all-cell study area's assembled inputs.
- **pruned** -- the same cells, simplified per area by the zero rules below.
- **rollup** -- exp03's structure-aware roll-up (``puma-r<spec>``) of the
  default (collapsed) categories, built from the default study area's inputs
  exactly as the fits build it (:class:`~pmedm_vb.assemble.collapse.
  CollapsedHierarchy`).

**The zero rules** (``pruned``), applied bottom-up on each table's published
tree in each area, the PUMA's summed estimates included. A zero cell's
variance is modelled and the same for any zero, so only one zero per branch
says anything:

1. A branch whose estimate is zero becomes one zero leaf; everything below it
   goes.
2. Within a branch, the zero children (zero leaves, and branches turned into
   zero leaves by rule 1) are merged into one zero cell.
3. A branch with exactly one nonzero child has the same estimate as that child
   in every replicate (the zeros are zero in all of them), so the child's own
   row is dropped as a duplicate; the child's descendants, and the branch's
   merged zero cell, stay. Not where a branch is not the exact sum of its
   children (``B23001``'s 65-and-over groups and the totals above them, which
   also count the armed forces).

The branch itself stays, as every subtotal does in ``all``; the rules repeat
up the tree because each branch sees its children already simplified.

    $CONDA_PREFIX/bin/python experiments/design_cells.py \\
        --all-area knoxall --default-area knox --pumas 4701501 4701502 4701503 4701504
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pmedm_vb.assemble.allcells import TreeLine, trees
from pmedm_vb.assemble.hierarchy import Hierarchy
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import StudyArea, processed_dir

LEVELS = ("puma", "tract", "block group")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-area", default="knoxall", help="the all-cell study area's name")
    parser.add_argument("--default-area", default="knox", help="the default study area's name")
    parser.add_argument("--state", default="47")
    parser.add_argument("--county", nargs="+", default=["093"])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--span", type=int, default=5)
    parser.add_argument("--pumas", nargs="+", required=True)
    parser.add_argument("--rollup", default="50h20", help="exp03's roll-up spec")
    return parser.parse_args()


@dataclass
class Tally:
    """Cells, count cells and zero cells, per (level, table), and areas per level."""

    cells: dict = field(default_factory=dict)
    areas: dict = field(default_factory=lambda: dict.fromkeys(LEVELS, 0))

    def add(self, level: str, table: str, count: int, zero: int) -> None:
        row = self.cells.setdefault((level, table), [0, 0])
        row[0] += count
        row[1] += zero


def pruned_cells(lines: tuple[TreeLine, ...], y: dict[str, float]) -> tuple[int, int]:
    """(count cells, zero cells) of one table in one area under the zero rules."""
    children: dict[str, list[str]] = {}
    for line in lines:
        if line.parent is not None:
            children.setdefault(line.parent, []).append(line.name)
    additive = {line.name: line.additive for line in lines}

    def walk(name: str) -> tuple[bool, list[bool], bool]:
        """(the head row's zero flag, the zero flags of the cells below it,
        whether the subtree is a zero leaf)."""
        kids = children.get(name, [])
        if y[name] == 0:
            return True, [], True                    # rule 1 (and a zero leaf)
        if not kids:
            return False, [], False
        results = [walk(k) for k in kids]
        below = [True] if any(r[2] for r in results) else []   # rule 2
        nonzero = [r for r in results if not r[2]]
        for head, rest, _ in nonzero:
            if len(nonzero) == 1 and additive[name]:
                below += rest                                  # rule 3
            else:
                below += [head, *rest]
        return False, below, False

    root = lines[0].name
    head, below, _ = walk(root)
    flags = [head, *below]
    return sum(not z for z in flags), sum(flags)


def count_all_and_pruned(inputs: PMEDMInputs, tree: dict[str, tuple[TreeLine, ...]],
                         all_cells: Tally, pruned: Tally) -> None:
    by_level = {"puma": (list(inputs.tract_constraints), inputs.Y_T.sum(axis=0)[None, :]),
                "tract": (list(inputs.tract_constraints), inputs.Y_T),
                "block group": (list(inputs.bg_constraints), inputs.Y_B)}
    for level, (names, Y) in by_level.items():
        tables = np.array([n.split(".", 1)[0] for n in names])
        for tally in (all_cells, pruned):
            tally.areas[level] += Y.shape[0]
        for table in dict.fromkeys(tables):
            cols = np.flatnonzero(tables == table)
            cats = [names[k].split(".", 1)[1] for k in cols]
            for area in range(Y.shape[0]):
                values = Y[area, cols]
                zeros = int((values == 0).sum())
                all_cells.add(level, table, cols.size - zeros, zeros)
                count, zero = pruned_cells(tree[table], dict(zip(cats, values)))
                pruned.add(level, table, count, zero)


def count_rollup(inputs: PMEDMInputs, spec: str, tally: Tally) -> None:
    h = Hierarchy.build(inputs, f"puma-r{spec}")
    n_zones = inputs.Y_B.shape[0]
    tally.areas["puma"] += 1
    tally.areas["tract"] += h.n_tracts
    tally.areas["block group"] += n_zones
    for name, Y in zip(h.puma_names, h.Y_P):
        table = name.removeprefix("P:").split(".", 1)[0]
        tally.add("puma", table, int(Y != 0), int(Y == 0))
    for name, zero, is_tract in zip(h.cell_names, h.cell_zero, h.cell_is_tract):
        table = name.split(":", 1)[1].split(".", 1)[0]
        level = "tract" if is_tract else "block group"
        tally.add(level, table, int(not zero), int(zero))


def report(schemes: dict[str, Tally]) -> None:
    rows = []
    for scheme, tally in schemes.items():
        for (level, table), (count, zero) in tally.cells.items():
            n = tally.areas[level]
            rows.append(dict(scheme=scheme, level=level, table=table,
                             cells=(count + zero) / n, count=count / n, zero=zero / n))
    frame = pd.DataFrame(rows)
    order = {"scheme": list(schemes), "level": list(LEVELS)}
    total = frame.groupby(["level", "scheme"])[["cells", "count", "zero"]].sum().reset_index()
    for column, values in order.items():
        total[column] = pd.Categorical(total[column], values, ordered=True)
        frame[column] = pd.Categorical(frame[column], values, ordered=True)
    print("Average cells per area (count = published estimate above zero)\n")
    print(total.sort_values(["level", "scheme"]).to_string(index=False, float_format="{:.1f}".format))
    print("\nBy table\n")
    wide = frame.pivot_table(index=["level", "table"], columns="scheme", values="cells",
                             observed=True)
    print(wide.round(1).to_string())
    print("\nZero cells by table\n")
    wide = frame.pivot_table(index=["level", "table"], columns="scheme", values="zero",
                             observed=True)
    print(wide.round(1).to_string())


def main() -> None:
    args = parse_args()

    def area(name: str) -> StudyArea:
        return StudyArea(name=name, state=args.state, year=args.year,
                         counties=tuple(args.county), span=args.span)

    all_area, default_area = area(args.all_area), area(args.default_area)
    tree = trees(all_area)
    schemes = {"all": Tally(), "pruned": Tally(), "rollup": Tally()}
    for puma in args.pumas:
        full = PMEDMInputs.load(processed_dir() / "inputs" / all_area.slug / puma)
        count_all_and_pruned(full, tree, schemes["all"], schemes["pruned"])
        default = PMEDMInputs.load(processed_dir() / "inputs" / default_area.slug / puma)
        count_rollup(default, args.rollup, schemes["rollup"])
    report(schemes)


if __name__ == "__main__":
    main()
