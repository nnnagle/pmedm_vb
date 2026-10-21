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

And the same counts per PUMA, over all constraint tables. Both have a row per
(table or PUMA, level).

With ``--experiments`` (a folder of experiment JSONs), a separate table of each
experiment's parameters per PUMA: the solver's coordinates and, in
parentheses, how many of them the allocation identifies.

Writes ``<out>/table_inventory.md``, ``.tex`` (booktabs) and ``.csv``, the
cell counts by table as ``<out>/table_cells.*`` and by PUMA as
``<out>/puma_cells.*``, and the parameters as ``<out>/parameters.*``.
``--name`` picks the assembled inputs: ``knoxall`` for the all-cell set::

    $CONDA_PREFIX/bin/python experiments/table_inventory.py --name knoxall \\
        --puma 4701501 4701502 4701503 4701504 --experiments experiments/paper_v3 \\
        --out $RUNS/table_inventory
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from design_cells import (LEVELS as TREE_LEVELS, ROLLUP_UNIVERSE, Tally, count_all_and_pruned,
                          count_rollup)

from pmedm_vb.assemble.allcells import all_tables, is_subtotal, table_set, trees
from pmedm_vb.assemble.constraints import HELDOUT_RELATION
from pmedm_vb.assemble.heldout import HeldOut
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.rollup import RollupSpec
from pmedm_vb.config import StudyArea, processed_dir
from pmedm_vb.data.variance import table_list
from pmedm_vb.progress import logger

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
    parser.add_argument("--rollup", default="25h10n",
                        help="the roll-up whose cells are counted beside the published and pruned")
    parser.add_argument("--experiments", type=Path, default=None,
                        help="a folder of experiment JSONs (e.g. experiments/paper_v3): adds each "
                             "experiment's parameters per PUMA (slow: builds every null space)")
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


def tallies(area: StudyArea, pumas: list[str], rollup: RollupSpec
            ) -> dict[str, tuple[Tally, Tally | None, Tally | None]]:
    """Per PUMA, the published cells and, for the all-cell set, the pruned and
    the rolled-up ones (``design_cells.py``'s counts, which match the fitted
    hierarchies').

    PUMA cells are the tract estimates summed per PUMA, one per category, as
    the ``--hierarchy puma`` rows are.
    """
    tree, universe, out = None, None, {}
    for puma in pumas:
        inputs = PMEDMInputs.load(processed_dir() / "inputs" / area.slug / puma)
        published, pruned, rolled = Tally(), Tally(), Tally()
        if table_set(list(inputs.tract_constraints)) == "all":
            if tree is None:
                tree = trees(area)
                universe = {t.table: t.universe for t in all_tables(area)} | ROLLUP_UNIVERSE
            count_all_and_pruned(inputs, tree, published, pruned)
            count_rollup(inputs, tree, universe, rollup, rolled)
        else:
            count_published(inputs, published)
            pruned = rolled = None
        out[puma] = (published, pruned, rolled)
    return out


def fmt(cell: list[int] | None) -> str:
    """``cells (zeros)`` from a tally's ``[nonzero, zero]``."""
    return "–" if cell is None else f"{cell[0] + cell[1]:,} ({cell[1]:,})"


def add(total: list[int] | None, cell: list[int] | None) -> list[int] | None:
    if cell is None:
        return total
    return list(cell) if total is None else [total[0] + cell[0], total[1] + cell[1]]


def long_rows(key: str, groups: list[tuple[str, list, object]], rollup: str) -> pd.DataFrame:
    """One row per (group, level) with the published, pruned and rolled-up
    cells and their zeros; the group named on its first row only, a level
    with no cells left out, and a Total group last.

    ``groups`` is ``(name, tallies, select)``: the PUMAs' tally triples to sum,
    and ``select(table)``, which of their tables count towards the group.
    """
    schemes = [("published (zeros)", 0), ("pruned (zeros)", 1),
               (f"roll-up {rollup} (zeros)", 2)]
    if any(t[1] is None for _, tallies_, _ in groups for t in tallies_):
        schemes = schemes[:1]                       # a collapsed set: nothing to prune
    totals = {level: [None] * len(schemes) for level in TREE_LEVELS}
    rows = []

    def emit(name: str, cells: dict[str, list]) -> None:
        first = True
        for level in TREE_LEVELS:
            if cells[level][0] is None:
                continue
            rows.append({key: name if first else "", "Level": LEVEL_LABELS[level],
                         **{header: fmt(c) for (header, _), c in zip(schemes, cells[level])}})
            first = False

    for name, tallies_, select in groups:
        cells = {level: [None] * len(schemes) for level in TREE_LEVELS}
        for triple in tallies_:
            for j, (_, which) in enumerate(schemes):
                for (level, table), value in triple[which].cells.items():
                    if select(table):
                        cells[level][j] = add(cells[level][j], value)
        for level in TREE_LEVELS:
            totals[level] = [add(t, c) for t, c in zip(totals[level], cells[level])]
        emit(name, cells)
    emit("Total", totals)
    return pd.DataFrame(rows)


def cell_counts(counts: dict[str, tuple[Tally, Tally | None, Tally | None]],
                rollup: str) -> pd.DataFrame:
    """Per constraint table and level: cells (areas x categories) and published
    zeros, summed over the PUMAs; with the pruned and rolled-up cells beside
    them for the all-cell set."""
    tables = sorted({table for triple in counts.values() for _, table in triple[0].cells})
    return long_rows("Table", [(table, list(counts.values()), lambda t, table=table: t == table)
                               for table in tables], rollup)


def puma_counts(counts: dict[str, tuple[Tally, Tally | None, Tally | None]],
                rollup: str) -> pd.DataFrame:
    """Per PUMA: constraint cells at each level and the published zeros among
    them, over every constraint table; pruned and rolled up beside them for the
    all-cell set."""
    return long_rows("PUMA ID", [(puma, [triple], lambda t: True)
                                 for puma, triple in counts.items()], rollup)


def parameters(area: StudyArea, pumas: list[str], folder: Path) -> pd.DataFrame:
    """Per PUMA, each experiment's parameters, as ``coordinates (identified)``:
    the solver's coordinates (the count its MAP and VB logs print) and how many
    of them the allocation identifies.

    The identified count is the coordinates less every direction along which
    no allocation weight moves: the null space of the experiment's own cells,
    found as the null-space level finds it (``nullspace.py``), counting the
    directions it removes and those it keeps as inconsistent alike. Neither
    count depends on ``alpha``: the null space comes from ``X`` and the
    targets and replicates, ``Sigma`` only rescales it, so the first ``alpha``
    serves.
    """
    from paper_tables import EXPERIMENTS
    from run_experiment import level

    from pmedm_vb.assemble.hierarchy import Hierarchy

    labels = dict(EXPERIMENTS)

    specs = [json.loads(path.read_text()) for path in sorted(folder.glob("exp*.json"))]
    rows = []
    for puma in pumas:
        inputs = PMEDMInputs.load(processed_dir() / "inputs" / area.slug / puma)
        row = {"PUMA ID": puma}
        for spec in specs:
            model = spec["model"]
            options = dict(
                alpha=spec["alphas"][0],
                taper=None if model["taper"] == "none" else model["taper"],
                variance_floor=(None if str(model["variance_floor"]) == "none"
                                else model["variance_floor"]))
            fitted = Hierarchy.build(inputs, level(spec), **options)
            null = (fitted if model["hierarchy"] == "nullspace" else
                    Hierarchy.build(inputs, "nullspace" + (f"-r{model['rollup']}"
                                                           if model.get("rollup") else ""),
                                    **options))
            unseen = null.n_removed + null.n_inconsistent
            row[labels.get(spec["name"], spec["name"])] = (fitted.size, null.size - unseen)
            logger.info("%s %s: %d coordinates, %d identified", puma, spec["name"],
                        fitted.size, null.size - unseen)
        rows.append(row)
    columns = [c for c in rows[0] if c != "PUMA ID"]
    rows.append({"PUMA ID": "Total", **{c: tuple(map(sum, zip(*(r[c] for r in rows))))
                                         for c in columns}})
    frame = pd.DataFrame(rows)
    for column in columns:
        frame[column] = frame[column].map(lambda v: f"{v[0]:,} ({v[1]:,})")
    return frame.rename(columns={c: f"{c}: coordinates (identified)" for c in columns})


def simple_markdown(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def simple_latex(frame: pd.DataFrame) -> str:
    text = 2 if "Level" in frame.columns else 1     # left-aligned columns
    lines = [r"\begin{tabular}{" + "l" * text + "r" * (len(frame.columns) - text) + "}",
             r"\toprule", " & ".join(frame.columns) + r" \\", r"\midrule"]
    for i, r in enumerate(frame.itertuples(index=False)):
        if r[0] == "Total":
            lines.append(r"\midrule")
        elif i and r[0] and text == 2:
            lines.append(r"\addlinespace")             # a new group of levels
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
    rollup = RollupSpec.parse(args.rollup)
    counts = tallies(area, args.puma, rollup)
    cells = cell_counts(counts, str(rollup))
    cells.to_csv(args.out / "table_cells.csv", index=False)
    (args.out / "table_cells.md").write_text(simple_markdown(cells))
    (args.out / "table_cells.tex").write_text(simple_latex(cells))
    by_puma = puma_counts(counts, str(rollup))
    by_puma.to_csv(args.out / "puma_cells.csv", index=False)
    (args.out / "puma_cells.md").write_text(simple_markdown(by_puma))
    (args.out / "puma_cells.tex").write_text(simple_latex(by_puma))
    if args.experiments is not None:
        params = parameters(area, args.puma, args.experiments)
        params.to_csv(args.out / "parameters.csv", index=False)
        (args.out / "parameters.md").write_text(simple_markdown(params))
        (args.out / "parameters.tex").write_text(simple_latex(params))
        print(simple_markdown(params))
    missing = frame.loc[frame.title == "", "table"].tolist()
    print(markdown(frame))
    print(simple_markdown(cells))
    print(simple_markdown(by_puma))
    if missing:
        print(f"no published title found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
