"""The census tables the study uses, as a table for the article.

One row per ACS table: its code and published title, its role (constraint or
held out, and for held-out tables how related it is to the constraints), the
number of categories the model uses (after collapsing published cells), the
levels it enters at, and its largest and smallest categories by count summed
over the study area's PUMAs, to show what the table is about. Ties are broken
at random (``--seed``).

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

Writes ``<out>/table_inventory.md``, ``.tex`` (booktabs) and ``.csv``::

    $CONDA_PREFIX/bin/python experiments/table_inventory.py \\
        --puma 4701501 4701502 4701503 4701504 --out $RUNS/table_inventory
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

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
        rows.append(dict(
            table=table, title=titles.get(table, ""), role=role, categories=len(group),
            levels=", ".join(levels + ["PUMA (sum)"]),
            largest=pick(group, True, rng), smallest=pick(group, False, rng),
        ))
    frame = pd.DataFrame(rows)
    frame["_order"] = frame.role.str.startswith("held").astype(int)
    return frame.sort_values(["_order", "table"]).drop(columns="_order").reset_index(drop=True)


HEADERS = ["Table", "Title", "Role", "Categories", "Levels", "Largest category (count)",
           "Smallest category (count)"]


def markdown(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
    for r in frame.itertuples(index=False):
        cells = [r.table, r.title, r.role, str(r.categories), r.levels, r.largest, r.smallest]
        lines.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    return "\n".join(lines) + "\n"


def latex_escape(text: str) -> str:
    text = re.sub(r"([&%$#_{}])", r"\\\1", text)
    return text.replace(">=", r"$\geq$").replace("<=", r"$\leq$").replace(
        "<", r"$<$").replace(">", r"$>$")


def latex(frame: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lp{4.5cm}p{2.2cm}rp{2.2cm}p{3.5cm}p{3.5cm}}", r"\toprule",
             " & ".join(HEADERS) + r" \\", r"\midrule"]
    for r in frame.itertuples(index=False):
        cells = [r.table, r.title, r.role, str(r.categories), r.levels, r.largest, r.smallest]
        lines.append(" & ".join(latex_escape(c) for c in cells) + r" \\")
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
    missing = frame.loc[frame.title == "", "table"].tolist()
    print(markdown(frame))
    if missing:
        print(f"no published title found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
