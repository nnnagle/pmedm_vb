"""Recover each constraint table's universe / sub-universe structure from its data.

A table's published lines include its universe and sub-universe cells, each the
exact sum of the lines beneath it -- in every area and in every one of the 80
replicates, since each replicate is a full re-tabulation. A table has the same
structure in every area and at every level, so it is read once, from the
national file (summary level 010), where almost no line is zero:

1. **Tree.** The lines are published in pre-order (a subtotal, then its
   descendants), and a sub-universe's title ends in ":". A recursive parse
   takes each such line as a subtotal whose children are the subtrees that
   follow it up to where their sum equals it exactly -- to ``--tol`` in the
   national estimate and its 80 replicates. Where nothing sums exactly (a few
   people nationally outside every published line), it closes at the nearest
   run if the relative gap is at most ``--close-rtol``, and says so; titled
   subtotals that still fail are reported. The extent of an all-zero subtotal is not fixed by the
   data; it takes the all-zero lines that follow it.
2. **Check on the study area.** At each level the table constrains, every
   subtotal must equal the sum of its children in every area and replicate.
   Stack the lines as rows of a matrix whose columns are every area's estimate
   and its 80 replicates; its rank deficiency is the number of independent
   exact identities among the lines. The tree and the all-zero lines should
   explain all of them; any excess is a relation the tree does not explain --
   a second margin (a cross-tab publishing both its row and column totals: a
   lattice, not a tree) or a line that repeats another -- and the lines
   involved are listed.
3. **Our categories in the tree.** Each category of the constraint table is a
   set of published lines. It is placed under the lowest tree node containing
   all of them, and categories under the same node are siblings -- the groups
   a structure-aware roll-up merges within before crossing a sub-universe.

4. **The tree over our categories.** For every published node, the set of
   our categories with a line under it; the sets that nest are the nodes of
   the tree the roll-up works on (:mod:`pmedm_vb.assemble.rollup`), printed
   per table, with any sets dropped for crossing another. ``--write-inputs``
   writes these trees into each PUMA's inputs.

Writes ``<out>/table_trees.txt``: per table, the tree (indented, with titles
and national totals), then per level the check and the sibling groups of our
categories with their county counts; and ``<out>/table_trees.csv``, one row per
line and level; and ``<out>/category_trees.json``. Reads the cached replicate files, fetching any missing ones
(so run it where the network is available, e.g. a login node)::

    $CONDA_PREFIX/bin/python experiments/table_trees.py --out $RUNS/table_trees
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.assemble.constraints import default_tables
from pmedm_vb.assemble.inputs import TREES_FILE
from pmedm_vb.assemble.rollup import category_tree, tree_categories
from pmedm_vb.config import StudyArea, processed_dir
from pmedm_vb.data.cache import fetch_cached
from pmedm_vb.data.variance import (REPLICATE_COLUMNS, VRE_BASE, _read_replicate_file,
                                    _span_label, fetch_replicates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="knox")
    parser.add_argument("--state", default="47")
    parser.add_argument("--county", nargs="+", default=["093"])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--span", type=int, default=5)
    parser.add_argument("--levels", nargs="+", default=["tract", "block group"])
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="largest |subtotal - sum of children| counted as exact")
    parser.add_argument("--close-rtol", type=float, default=1e-4,
                        help="where no run of lines sums exactly to a subtotal in the US "
                             "file, close it at the nearest run if its gap relative to the "
                             "subtotal is at most this (the study-area check stays exact)")
    parser.add_argument("--write-inputs", nargs="+", default=[], metavar="AREA",
                        help="also write the category trees into every PUMA's inputs under "
                             "processed/inputs/AREA (e.g. knox-2024-5yr), for the "
                             "structure-aware roll-up (the -r hierarchy levels)")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def line_matrix(frame: pd.DataFrame, cells: list[str] | None = None
                ) -> tuple[list[str], list[str], np.ndarray]:
    """Lines in published order (or in ``cells``' order), their titles, and the
    ``(lines, areas x 81)`` matrix of estimates and replicates."""
    frame = frame.reset_index()
    if cells is None:
        order = frame.groupby("cell")["order"].first().sort_values()
        cells = list(order.index)
    titles = frame.groupby("cell")["title"].first().reindex(cells).tolist()
    blocks = []
    for column in ["estimate", *REPLICATE_COLUMNS]:
        wide = frame.pivot(index="cell", columns="geoid", values=column).reindex(cells)
        blocks.append(wide.to_numpy(dtype=float))
    V = np.nan_to_num(np.hstack(blocks))
    return cells, titles, V


def parse_tree(V: np.ndarray, titles: list[str], tol: float, close_rtol: float = 0.0
               ) -> tuple[dict[int, list[int]], list[int], list[int], list[int], dict[int, float]]:
    """``(children, roots, failed, assumed, approx)``.

    ``children`` maps each subtotal line to its child lines. Candidates are the
    lines whose title ends in ":" (the Census marks sub-universes that way);
    every other line is a leaf. A candidate's children are the subtrees that
    follow it, up to the first point where their sum equals it exactly in
    every column; all-zero leaves right after that point are taken in too.
    No sign is assumed, so a running sum may pass the subtotal on the way.
    Where no run sums exactly, the candidate closes at the run whose sum is
    nearest it -- the smallest largest-over-columns gap relative to the
    subtotal -- if that gap is at most ``close_rtol``; ``approx`` maps each such
    line to its gap in the first column (the estimate). The nearest run, not
    the first within tolerance, so a tiny last line is not left out. ``failed``
    lists candidates no run of following subtrees adds up to (they
    are kept as leaves); ``assumed`` lists all-zero candidates, whose extent
    the data cannot fix and which take the following all-zero lines.
    """
    n = V.shape[0]
    zero = np.abs(V).max(axis=1) <= tol
    colon = [str(t).rstrip().endswith(":") for t in titles]
    sys.setrecursionlimit(max(10_000, 10 * n))
    failed, assumed, approx = set(), set(), {}

    @functools.lru_cache(maxsize=None)
    def parse(i: int) -> tuple[int, tuple[int, ...]]:
        """(end, children): line i's subtree is lines i..end-1."""
        if not colon[i]:
            return i + 1, ()
        if zero[i]:
            assumed.add(i)
            kids, j = [], i + 1
            while j < n and zero[j]:
                kids.append(j)
                j, _ = parse(j)
            return (j, tuple(kids)) if kids else (i + 1, ())
        acc = np.zeros(V.shape[1])
        scale = np.abs(V[i]).max()
        kids, j = [], i + 1
        best = (np.inf, i + 1, (), 0.0)       # (relative gap, end, children, estimate gap)
        while j < n:
            end, _ = parse(j)
            acc = acc + V[j]
            kids.append(j)
            j = end
            gap = np.abs(acc - V[i]).max()
            if gap <= tol:
                return absorb_zeros(j, kids)
            if gap / scale < best[0]:
                best = (gap / scale, j, tuple(kids), float(V[i, 0] - acc[0]))
        if j == i + 1:                        # nothing follows: a lone line
            return i + 1, ()
        if best[0] <= close_rtol:
            approx[i] = best[3]
            return absorb_zeros(best[1], list(best[2]))
        failed.add(i)
        return i + 1, ()

    def absorb_zeros(j: int, kids: list[int]) -> tuple[int, tuple[int, ...]]:
        """All-zero leaves right after a subtotal closes belong to it."""
        while j < n and zero[j] and not colon[j]:
            kids.append(j)
            j += 1
        return j, tuple(kids)

    children, roots, i = {}, [], 0
    while i < n:
        roots.append(i)
        i, _ = parse(i)
    # keep only the parse actually reached from the roots
    stack = list(roots)
    while stack:
        k = stack.pop()
        _, kids = parse(k)
        if kids:
            children[k] = list(kids)
            stack += kids
    return (children, roots, sorted(failed), sorted(assumed & set(children)),
            {k: g for k, g in approx.items() if k in children})


def unexplained(V: np.ndarray, children: dict[int, list[int]], tol: float
                ) -> tuple[int, int, list[list[int]]]:
    """Rank deficiency of V, the number of independent identities the tree and
    the all-zero lines explain, and the line sets of the relations they miss."""
    scale = max(1.0, np.abs(V).max())
    u, s, _ = np.linalg.svd(V / scale, full_matrices=True)
    rank = int((s > 1e-9 * s.max()).sum()) if s.size and s.max() > 0 else 0
    left_null = u[:, rank:]                                        # (lines, nullity)
    known = []
    for p, kids in children.items():
        v = np.zeros(V.shape[0])
        v[p], v[kids] = 1.0, -1.0
        known.append(v)
    zero = np.flatnonzero(np.abs(V).max(axis=1) <= tol)
    for z in zero:
        v = np.zeros(V.shape[0])
        v[z] = 1.0
        known.append(v)
    K = np.array(known).T if known else np.zeros((V.shape[0], 0))
    n_known = int(np.linalg.matrix_rank(K)) if K.size else 0
    # an orthonormal basis of the explained identities (K may be rank deficient)
    if K.size:
        uk, sk, _ = np.linalg.svd(K, full_matrices=False)
        q = uk[:, : n_known]
    else:
        q = K
    rest = left_null - q @ (q.T @ left_null) if q.size else left_null
    uu, ss, _ = np.linalg.svd(rest, full_matrices=False) if rest.size else (rest, np.zeros(0), None)
    extra = uu[:, ss > 1e-6] if rest.size else rest
    groups = [list(np.flatnonzero(np.abs(extra[:, j]) > 1e-6)) for j in range(extra.shape[1])]
    return V.shape[0] - rank, n_known, groups


def ancestors(children: dict[int, list[int]], n: int) -> dict[int, int]:
    parent = {}
    for p, kids in children.items():
        for k in kids:
            parent[k] = p
    return parent


def depths(parent: dict[int, int], n: int) -> dict[int, int]:
    out = {}
    for i in range(n):
        d, k = 0, i
        while k in parent:
            k, d = parent[k], d + 1
        out[i] = d
    return out


def tree_report(cells, titles, V, tol, close_rtol) -> tuple[list[str], dict[int, list[int]]]:
    """Parse the tree from the national file; its summary and the indented tree."""
    children, roots, failed, assumed, approx = parse_tree(V, titles, tol, close_rtol)
    parent = ancestors(children, len(cells))
    depth = depths(parent, len(cells))
    n_zero = int((np.abs(V).max(axis=1) <= tol).sum())
    n_colon = sum(str(t).rstrip().endswith(":") for t in titles)
    lines = [f"   US: {len(cells)} lines ({n_colon} titled as subtotals); {len(children)} subtotals "
             f"parsed; {n_zero} all-zero lines; smallest value {V.min():,.0f}"]
    if failed:
        lines.append("   titled subtotals that no run of following lines sums to: "
                     + ", ".join(f"{cells[k]} {titles[k]}" for k in failed))
    if approx:
        lines.append("   subtotals closed approximately (US estimate - sum of children): "
                     + ", ".join(f"{cells[k]} {titles[k]} ({g:+,.0f})" for k, g in approx.items()))
    if assumed:
        lines.append("   all-zero subtotals (extent taken as the following all-zero lines): "
                     + ", ".join(cells[k] for k in assumed))
    us = V[:, 0]  # the single area's estimate comes first
    for i, (cell, title) in enumerate(zip(cells, titles)):
        tag = " [subtotal]" if i in children else ""
        lines.append(f"     {'  ' * depth[i]}{cell} {title} (US {us[i]:,.0f}){tag}")
    return lines, children


#: Tables whose tree failed the exact check at some level; no trees are written
#: into the inputs while any did.
tree_failures: list[str] = []


def level_report(cells, titles, V, children, table_spec, tol) -> tuple[list[str], pd.DataFrame]:
    """Check the national tree on one level's areas and place our categories in it."""
    parent = ancestors(children, len(cells))
    depth = depths(parent, len(cells))
    residual = max((float(np.abs(V[p] - V[kids].sum(axis=0)).max()) for p, kids in children.items()),
                   default=0.0)
    nullity, n_known, extra = unexplained(V, children, tol)
    n_zero = int((np.abs(V).max(axis=1) <= tol).sum())
    county = V[:, : V.shape[1] // 81].sum(axis=1)  # estimate columns come first
    ok = residual <= tol and not extra
    if not ok:
        tree_failures.append(table_spec.table)
    lines = [f"   max |subtotal - sum of children| {residual:.3g}; {n_zero} all-zero lines; "
             f"smallest value {V.min():,.0f}; rank deficiency {nullity} = {n_known} explained by "
             f"the tree and zero lines + {nullity - n_known} unexplained"
             + ("  -> TREE" if ok else "  -> NOT A TREE (see below)")]
    if residual > tol:
        for p, kids in children.items():
            gap = float(np.abs(V[p] - V[kids].sum(axis=0)).max())
            if gap > tol:
                lines.append(f"   fails here: {cells[p]} {titles[p]} (max gap {gap:,.3g})")
    if extra:
        # State each unexplained relation sparsely: a line outside the tree's
        # identities that is an exact 0/1 sum of the tree's leaves.
        leaves = [i for i in range(len(cells)) if i not in children and np.abs(V[i]).max() > tol]
        involved = sorted(set().union(*map(set, extra)))
        for k in involved:
            others = [i for i in leaves if i != k]
            coef, *_ = np.linalg.lstsq(V[others].T, V[k], rcond=None)
            fit = V[others].T @ np.round(coef)
            if (np.abs(np.round(coef) - coef).max() < 1e-6 and set(np.round(coef)) <= {0.0, 1.0}
                    and np.abs(fit - V[k]).max() <= tol):
                terms = [others[j] for j in np.flatnonzero(np.round(coef) == 1)]
                if len(terms) > 1 and not (k in children and sorted(children[k]) == sorted(terms)):
                    lines.append(f"   extra identity: {cells[k]} {titles[k]} = "
                                 + " + ".join(f"{cells[t]} {titles[t]}" for t in terms))
        lines.append("   unexplained lines (support of the relations the tree misses): "
                     + ", ".join(cells[k] for k in involved))

    # our categories: lowest node containing all of a category's lines
    index = {c: i for i, c in enumerate(cells)}
    def path(i):
        out = [i]
        while out[-1] in parent:
            out.append(parent[out[-1]])
        return out
    @functools.lru_cache(maxsize=None)
    def leaf_set(i: int) -> frozenset:
        return (frozenset().union(*(leaf_set(k) for k in children[i])) if i in children
                else frozenset([i]))

    groups: dict[int, list[str]] = {}
    for cat in table_spec.categories:
        members = [index[c] for c in cat.published if c in index]
        if not members:
            continue
        covered = frozenset().union(*(leaf_set(m) for m in members))
        common = set(path(members[0]))
        for m in members[1:]:
            common &= set(path(m))
        lca = max(common, key=lambda k: depth[k]) if common else -1
        # a category that is exactly one node's leaves is that node; its sibling
        # group is the node's parent. Otherwise it is a partial union under lca.
        if lca >= 0 and leaf_set(lca) == covered:
            home = parent.get(lca, -1)
        else:
            home = lca
        groups.setdefault(home, []).append(f"{cat.name} ({county[members].sum():,.0f})")
    lines.append("   our categories by parent node, with county counts (the sibling groups "
                 "for rolling up):")
    for home, names in sorted(groups.items(), key=lambda kv: kv[0]):
        label = "(top)" if home < 0 else f"{cells[home]} {titles[home]}"
        lines.append(f"     {label}: {', '.join(names)}")
    frame = pd.DataFrame({"cell": cells, "title": titles, "depth": [depth[i] for i in range(len(cells))],
                          "parent": [cells[parent[i]] if i in parent else "" for i in range(len(cells))],
                          "subtotal": [i in children for i in range(len(cells))],
                          "county_estimate": county})
    return lines, frame


def fetch_us(area: StudyArea, table: str) -> pd.DataFrame:
    """The national (summary level 010) replicate file, named ``{table}.csv.zip``."""
    url = f"{VRE_BASE}/{area.year}/data/{_span_label(area)}/010/{table}.csv.zip"
    frame = _read_replicate_file(fetch_cached(url, subdir=f"vre/{area.year}/010"))
    return frame.assign(geoid="US").set_index(["geoid", "cell"])


def main() -> None:
    args = parse_args()
    area = StudyArea(name=args.name, state=args.state, year=args.year,
                     counties=tuple(args.county), span=args.span)
    args.out.mkdir(parents=True, exist_ok=True)
    specs: dict[str, dict[str, object]] = {}
    for level in args.levels:
        for t in default_tables(area):
            if t.geography == level:
                specs.setdefault(t.table, {})[level] = t
    out_lines, frames, trees = [], [], {}
    for table, by_level in specs.items():
        cells, titles, V_us = line_matrix(fetch_us(area, table))
        out_lines.append(f"== {table}: tree from the US file")
        lines, children = tree_report(cells, titles, V_us, args.tol, args.close_rtol)
        out_lines += lines
        for level, spec in by_level.items():
            frame = fetch_replicates(area, [table], geography=level)
            local_cells, _, _ = line_matrix(frame)
            _, _, V = line_matrix(frame, cells)
            out_lines.append(f"  -- checked on {args.name} {level}: {V.shape[1] // 81} areas")
            if set(local_cells) != set(cells):
                out_lines.append(f"   lines differ from the US file: only here "
                                 f"{sorted(set(local_cells) - set(cells))}, only in US "
                                 f"{sorted(set(cells) - set(local_cells))}")
            lines, rows = level_report(cells, titles, V, children, spec, args.tol)
            out_lines += lines
            frames.append(rows.assign(table=table, level=level,
                                      us_estimate=V_us[:, 0]))
        # the tree over our categories, the same at every level
        specs_here = list(by_level.values())
        categories = [(c.name, tuple(c.published)) for c in specs_here[0].categories]
        for other in specs_here[1:]:
            if [(c.name, tuple(c.published)) for c in other.categories] != categories:
                raise ValueError(f"{table}: categories differ between levels")
        tree, dropped = category_tree(children, cells, titles, categories)
        trees[table] = {"universe": specs_here[0].universe, "tree": tree, "dropped": dropped}
        out_lines.append(f"  -- category tree ({specs_here[0].universe}):")
        out_lines += category_tree_lines(tree)
        if dropped:
            out_lines.append("   category sets dropped for crossing another: "
                             + "; ".join("{" + ", ".join(d) + "}" for d in dropped))
        out_lines.append("")
    pd.concat(frames).to_csv(args.out / "table_trees.csv", index=False)
    (args.out / "category_trees.json").write_text(json.dumps(trees, indent=1) + "\n")
    text = "\n".join(out_lines) + "\n"
    (args.out / "table_trees.txt").write_text(text)
    print(text)
    if args.write_inputs:
        if tree_failures:
            raise SystemExit(f"not writing trees into the inputs: {sorted(set(tree_failures))} "
                             f"failed the exact check")
        for name in args.write_inputs:
            root = processed_dir() / "inputs" / name
            dirs = sorted(d for d in root.iterdir() if (d / "manifest.json").exists())
            if not dirs:
                raise SystemExit(f"no inputs under {root}")
            for d in dirs:
                (d / TREES_FILE).write_text(json.dumps(trees, indent=1) + "\n")
            print(f"wrote {TREES_FILE} into {len(dirs)} inputs under {root}")


def category_tree_lines(node: dict, depth: int = 0) -> list[str]:
    """The category tree, indented; a node unnamed, a leaf by its category."""
    out = []
    for child in node["children"]:
        if "cat" in child:
            out.append(f"     {'  ' * depth}{child['cat']}")
        else:
            out.append(f"     {'  ' * depth}[{len(tree_categories(child))} categories]")
            out += category_tree_lines(child, depth + 1)
    return out


if __name__ == "__main__":
    main()
