"""The census tables the study uses, as a table for the article.

One row per ACS table: its code and published title, its role (constraint or
held out, and for held-out tables how related it is to the constraints), the
number of cells the model uses and how many of them are leaves (the finest
published categories; the rest are totals and subtotals, which only the
all-cell set has, so for a collapsed set the two agree), the levels it enters
at, and its largest and smallest leaf categories by count summed over the
study area's PUMAs, to show what the table is about. Ties are broken at random
(``--seed``).

Levels: ``BG`` and ``tract`` where the table is constrained or held out at
that level; ``PUMA`` is always derived, the tract estimates summed (the PUMA
rows of ``--hierarchy puma`` and the ``*_puma`` scoring subsets), and is marked
as such. Counts come from tract estimates, or block group ones for a table not
used at tract; estimates nest, so either sums to the same PUMA total.

Titles are the Census Bureau's, from the variance replicate table list
(:func:`pmedm_vb.data.variance.table_list`), which the prefetch step caches;
where it is not cached this needs the network, so run it on a login node.
Category names are this project's, lightly tidied (``lt``/``ge`` become
``<``/``>=``, underscores become spaces).

Also, for the constraint tables only, the number of cells at each level
(areas x categories, over all the PUMAs; PUMA cells are the summed tract
estimates) with the published zeros among them in parentheses. For the
all-cell set, each level also has the cells left after pruning, the zero rules
of ``design_cells.py`` (``--rollup pruned``, the cells experiment 1 fits), with
their zeros.

And the same counts per PUMA, over all constraint tables.

Writes ``<out>/table_inventory.md``, ``.tex`` (booktabs) and ``.csv``, the
cell counts by table as ``<out>/table_cells.*`` and by PUMA as
``<out>/puma_cells.*``. ``--name`` picks the assembled inputs: ``knoxall`` for
the all-cell set::

    $CONDA_PREFIX/bin/python experiments/table_inventory.py --name knoxall \\
        --puma 4701501 4701502 4701503 4701504 --out $RUNS/table_inventory
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from design_cells import LEVELS as TREE_LEVELS, Tally, count_all_and_pruned

from pmedm_vb.assemble.allcells import is_subtotal, table_set, trees
from pmedm_vb.assemble.constraints import HELDOUT_RELATION
from pmedm_vb.assemble.heldout import HeldOut
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import StudyArea, processed_dir
from pmedm_vb.data.variance import table_list

LEVELS = ("block group", "tract")
TOKENS = {"lt": "<", "le": "<=", "gt": ">", "ge": ">="}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--puma", nargs="+", required=True)
    parser.add_argument("--name", default="knox")
    parser.add_argument("--state", default="47")
    parser.add_argument("--county", nargs="+", default=["093"])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--span", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="for breaking ties")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def tidy(category: str) -> str:
    return " ".join(TOKENS.get(t, t) for t in category.split("_"))


def collect(area: StudyArea, pumas: list[str]) -> pd.DataFrame:
    """One row per (table, category): role, levels present, count summed over PUMAs."""
    records = {}
    for puma in pumas:
        path = processed_dir() / "inputs" / area.slug / puma
        inputs = PMEDMInputs.load(path)
        sources = [("constraint", "tract", inputs.tract_constraints, inputs.Y_T),
                   ("constraint", "block group", inputs.bg_constraints, inputs.Y_B)]
        if HeldOut.exists(path):
            heldout = HeldOut.load(path)
            sources += [("held out", level, heldout.names[level], heldout.Y[level])
                        for level in LEVELS]
        for role, level, names, Y in sources:
            for name, total in zip(names, np.asarray(Y).sum(axis=0)):
                table, category = name.split(".", 1)
                rec = records.setdefault((role, table, category),
                                         {"levels": set(), "count": {}})
                rec["levels"].add(level)
                rec["count"].setdefault(level, 0.0)
                rec["count"][level] += float(total)
    rows = []
    for (role, table, category), rec in records.items():
        level = "tract" if "tract" in rec["levels"] else "block group"
        rows.append(dict(role=role, table=table, category=category,
                         bg="block group" in rec["levels"], tract="tract" in rec["levels"],
                         count=rec["count"][level]))
    return pd.DataFrame(rows)


def pick(group: pd.DataFrame, largest: bool, rng: np.random.Generator) -> str:
    target = group["count"].max() if largest else group["count"].min()
    tied = group[np.isclose(group["count"], target)]
    row = tied.iloc[rng.integers(len(tied))]
    return f"{tidy(row.category)} ({row['count']:,.0f})"


def inventory(cells: pd.DataFrame, titles: pd.Series, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (role, table), group in cells.groupby(["role", "table"], sort=False):
        group = group.sort_values("category")  # so the tie-break does not depend on load order
        if role == "held out":
            role = f"held out ({HELDOUT_RELATION[table].replace('_', ' ')})"
        levels = [lvl for lvl, have in (("BG", group.bg.any()), ("tract", group.tract.any())) if have]
        # Examples from the leaves: a total or subtotal is always the largest.
        leaves = group[~group.category.map(is_subtotal)]
        rows.append(dict(
            table=table, title=titles.get(table, ""), role=role, cells=len(group),
            leaves=len(leaves), levels=", ".join(levels + ["PUMA (sum)"]),
            largest=pick(leaves, True, rng), smallest=pick(leaves, False, rng),
        ))
    frame = pd.DataFrame(rows)
    frame["_order"] = frame.role.str.startswith("held").astype(int)
    return frame.sort_values(["_order", "table"]).drop(columns="_order").reset_index(drop=True)


HEADERS = ["Table", "Title", "Role", "Cells", "Leaf cells", "Levels", "Largest category (count)",
           "Smallest category (count)"]


def markdown(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
    for r in frame.itertuples(index=False):
        cells = [r.table, r.title, r.role, str(r.cells), str(r.leaves), r.levels, r.largest,
                 r.smallest]
        lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def latex_escape(text: str) -> str:
    text = re.sub(r"([&%$#_{}])", r"\\\1", text)
    return text.replace(">=", r"$\geq$").replace("<=", r"$\leq$").replace(
        "<", r"$<$").replace(">", r"$>$")


def latex(frame: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lp{4.5cm}p{2.2cm}rrp{2.2cm}p{3.5cm}p{3.5cm}}", r"\toprule",
             " & ".join(HEADERS) + r" \\", r"\midrule"]
    for r in frame.itertuples(index=False):
        cells = [r.table, r.title, r.role, str(r.cells), str(r.leaves), r.levels, r.largest,
                 r.smallest]
        lines.append(" & ".join(latex_escape(c) for c in cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


LEVEL_LABELS = {"puma": "PUMA", "tract": "Tract", "block group": "Block Group"}


def count_published(inputs: PMEDMInputs, tally: Tally) -> None:
    """Every constraint cell, per (level, table): ``count_all_and_pruned``'s
    first tally, for a collapsed set, which has no tree to prune."""
    for level, names, Y in (("puma", inputs.tract_constraints, inputs.Y_T.sum(axis=0)[None, :]),
                            ("tract", inputs.tract_constraints, inputs.Y_T),
                            ("block group", inputs.bg_constraints, inputs.Y_B)):
        tables = np.array([n.split(".", 1)[0] for n in names])
        for table in dict.fromkeys(tables):
            block = Y[:, tables == table]
            zeros = int((block == 0).sum())
            tally.add(level, table, block.size - zeros, zeros)


def tallies(area: StudyArea, pumas: list[str]) -> dict[str, tuple[Tally, Tally | None]]:
    """Per PUMA, the published cells and, for the all-cell set, the pruned ones.

    PUMA cells are the tract estimates summed per PUMA, one per category, as
    the ``--hierarchy puma`` rows are.
    """
    tree, out = None, {}
    for puma in pumas:
        inputs = PMEDMInputs.load(processed_dir() / "inputs" / area.slug / puma)
        published, pruned = Tally(), Tally()
        if table_set(list(inputs.tract_constraints)) == "all":
            tree = tree or trees(area)
            count_all_and_pruned(inputs, tree, published, pruned)
        else:
            count_published(inputs, published)
            pruned = None
        out[puma] = (published, pruned)
    return out


def fmt(cell: list[int] | None) -> str:
    """``cells (zeros)`` from a tally's ``[nonzero, zero]``."""
    return "–" if cell is None else f"{cell[0] + cell[1]:,} ({cell[1]:,})"


def columns(pruned: bool) -> list[tuple[str, str, int]]:
    """(header, level, which tally) per column: published, then pruned, per level."""
    out = []
    for level in TREE_LEVELS:
        out.append((f"{LEVEL_LABELS[level]} (zeros)", level, 0))
        if pruned:
            out.append((f"{LEVEL_LABELS[level]} pruned (zeros)", level, 1))
    return out


def add(total: list[int] | None, cell: list[int] | None) -> list[int] | None:
    if cell is None:
        return total
    return list(cell) if total is None else [total[0] + cell[0], total[1] + cell[1]]


def cell_counts(counts: dict[str, tuple[Tally, Tally | None]]) -> pd.DataFrame:
    """Per constraint table and level: cells (areas x categories) and published
    zeros, summed over the PUMAs; with the pruned cells beside them for the
    all-cell set."""
    pruned = all(p is not None for _, p in counts.values())
    cols = columns(pruned)
    tables = sorted({table for pub, _ in counts.values() for _, table in pub.cells})
    rows, total = [], {header: None for header, _, _ in cols}
    for table in tables:
        row = {"Table": table}
        for header, level, which in cols:
            cell = None
            for tally_pair in counts.values():
                cell = add(cell, tally_pair[which].cells.get((level, table)))
            row[header] = cell
            total[header] = add(total[header], cell)
        rows.append(row)
    rows.append({"Table": "Total", **total})
    frame = pd.DataFrame(rows)
    for header, _, _ in cols:
        frame[header] = frame[header].map(fmt)
    return frame


def puma_counts(counts: dict[str, tuple[Tally, Tally | None]]) -> pd.DataFrame:
    """Per PUMA: constraint cells at each level and the published zeros among
    them, over every constraint table; pruned beside them for the all-cell set."""
    pruned = all(p is not None for _, p in counts.values())
    cols = columns(pruned)
    rows, total = [], {header: None for header, _, _ in cols}
    for puma, pair in counts.items():
        row = {"PUMA ID": puma}
        for header, level, which in cols:
            cell = None
            for (lvl, _), value in pair[which].cells.items():
                if lvl == level:
                    cell = add(cell, value)
            row[header] = cell
            total[header] = add(total[header], cell)
        rows.append(row)
    rows.append({"PUMA ID": "Total", **total})
    frame = pd.DataFrame(rows)
    for header, _, _ in cols:
        frame[header] = frame[header].map(fmt)
    return frame


def simple_markdown(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def simple_latex(frame: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{l" + "r" * (len(frame.columns) - 1) + "}", r"\toprule",
             " & ".join(frame.columns) + r" \\", r"\midrule"]
    for r in frame.itertuples(index=False):
        if r[0] == "Total":
            lines.append(r"\midrule")
        lines.append(" & ".join(latex_escape(str(c)).replace("–", "--") for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    area = StudyArea(name=args.name, state=args.state, year=args.year,
                     counties=tuple(args.county), span=args.span)
    titles = table_list(area).set_index("TBLID")["TITLE"]
    frame = inventory(collect(area, args.puma), titles, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "table_inventory.csv", index=False)
    (args.out / "table_inventory.md").write_text(markdown(frame))
    (args.out / "table_inventory.tex").write_text(latex(frame))
    counts = tallies(area, args.puma)
    cells = cell_counts(counts)
    cells.to_csv(args.out / "table_cells.csv", index=False)
    (args.out / "table_cells.md").write_text(simple_markdown(cells))
    (args.out / "table_cells.tex").write_text(simple_latex(cells))
    by_puma = puma_counts(counts)
    by_puma.to_csv(args.out / "puma_cells.csv", index=False)
    (args.out / "puma_cells.md").write_text(simple_markdown(by_puma))
    (args.out / "puma_cells.tex").write_text(simple_latex(by_puma))
    missing = frame.loc[frame.title == "", "table"].tolist()
    print(markdown(frame))
    print(simple_markdown(cells))
    print(simple_markdown(by_puma))
    if missing:
        print(f"no published title found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
