"""Recover each constraint table's universe / sub-universe structure from its data.

A table's published lines include its universe and sub-universe cells, each the
exact sum of the lines beneath it -- in every area and in every one of the 80
replicates, since each replicate is a full re-tabulation. So the structure can
be read off the variance replicate files themselves, without the titles:

1. **Tree.** The lines are published in pre-order (a subtotal, then its
   descendants), and a sub-universe's title ends in ":". A recursive parse
   takes each such line as a subtotal whose children are the subtrees that
   follow it up to where their sum equals it exactly -- to ``--tol`` in every
   area and replicate. Titled subtotals that nothing sums to are reported. The
   extent of an all-zero subtotal is not fixed by the data; it takes the
   all-zero lines that follow it.
2. **Check against the null space.** Stack the lines as rows of a matrix whose
   columns are every area's estimate and its 80 replicates. Its rank
   deficiency is the number of independent exact identities among the lines.
   The tree accounts for one per subtotal, and each all-zero line for one
   more; any excess is a relation the tree does not explain -- a second margin
   (a cross-tab publishing both its row and column totals: a lattice, not a
   tree) or a line that repeats another -- and the lines involved are listed.
3. **Our categories in the tree.** Each category of the constraint table is a
   set of published lines. It is placed under the lowest tree node containing
   all of them, and categories under the same node are siblings -- the groups
   a structure-aware roll-up merges within before crossing a sub-universe.

Writes ``<out>/table_trees.txt``: per table and level, the tree (indented, with
titles and the county total of each line), the identity count against the
null space, any unexplained relations, and the sibling groups of our
categories; and ``<out>/table_trees.csv``, one row per line. Reads the cached
replicate files, fetching any missing ones (so run it where the network is
available, e.g. a login node)::

    $CONDA_PREFIX/bin/python experiments/table_trees.py --out $RUNS/table_trees
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.assemble.constraints import default_tables
from pmedm_vb.config import StudyArea
from pmedm_vb.data.variance import REPLICATE_COLUMNS, fetch_replicates


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
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def line_matrix(frame: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray]:
    """Lines in published order, their titles, and the ``(lines, areas x 81)``
    matrix of estimates and replicates."""
    frame = frame.reset_index()
    order = frame.groupby("cell")["order"].first().sort_values()
    cells = list(order.index)
    titles = frame.groupby("cell")["title"].first().reindex(cells).tolist()
    blocks = []
    for column in ["estimate", *REPLICATE_COLUMNS]:
        wide = frame.pivot(index="cell", columns="geoid", values=column).reindex(cells)
        blocks.append(wide.to_numpy(dtype=float))
    V = np.nan_to_num(np.hstack(blocks))
    return cells, titles, V


def parse_tree(V: np.ndarray, titles: list[str], tol: float
               ) -> tuple[dict[int, list[int]], list[int], list[int], list[int]]:
    """``(children, roots, failed, assumed)``.

    ``children`` maps each subtotal line to its child lines. Candidates are the
    lines whose title ends in ":" (the Census marks sub-universes that way);
    every other line is a leaf. A candidate's children are the subtrees that
    follow it, up to the first point where their sum equals it exactly in
    every column; all-zero leaves right after that point are taken in too.
    No sign is assumed, so a running sum may pass the subtotal on the way.
    ``failed`` lists candidates no run of following subtrees adds up to (they
    are kept as leaves); ``assumed`` lists all-zero candidates, whose extent
    the data cannot fix and which take the following all-zero lines.
    """
    n = V.shape[0]
    zero = np.abs(V).max(axis=1) <= tol
    colon = [str(t).rstrip().endswith(":") for t in titles]
    sys.setrecursionlimit(max(10_000, 10 * n))
    failed, assumed = set(), set()

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
        kids, j = [], i + 1
        while j < n:
            end, _ = parse(j)
            acc = acc + V[j]
            kids.append(j)
            j = end
            if np.abs(acc - V[i]).max() <= tol:
                while j < n and zero[j] and not colon[j]:
                    kids.append(j)
                    j += 1
                return j, tuple(kids)
        failed.add(i)
        return i + 1, ()

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
    return children, roots, sorted(failed), sorted(assumed & set(children))


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


def analyse(cells, titles, V, table_spec, tol) -> tuple[list[str], pd.DataFrame]:
    children, roots, failed, assumed = parse_tree(V, titles, tol)
    parent = ancestors(children, len(cells))
    depth = {}
    for i in range(len(cells)):
        d, k = 0, i
        while k in parent:
            k, d = parent[k], d + 1
        depth[i] = d
    residual = max((float(np.abs(V[p] - V[kids].sum(axis=0)).max()) for p, kids in children.items()),
                   default=0.0)
    nullity, n_known, extra = unexplained(V, children, tol)
    n_zero = int((np.abs(V).max(axis=1) <= tol).sum())
    n_colon = sum(str(t).rstrip().endswith(":") for t in titles)
    lines = [f"   {len(cells)} lines ({n_colon} titled as subtotals); {len(children)} subtotals parsed, "
             f"max |subtotal - sum of children| {residual:.3g}; {n_zero} all-zero lines; "
             f"smallest value {V.min():,.0f}; rank deficiency {nullity} = {n_known} explained by "
             f"the tree and zero lines + {nullity - n_known} unexplained"
             + ("  -> TREE" if not extra and not failed else "  -> NOT A TREE (see below)")]
    if failed:
        lines.append("   titled subtotals that no run of following lines sums to: "
                     + ", ".join(f"{cells[k]} {titles[k]}" for k in failed))
    if assumed:
        lines.append("   all-zero subtotals (extent taken as the following all-zero lines): "
                     + ", ".join(cells[k] for k in assumed))
    county = V[:, : V.shape[1] // 81].sum(axis=1)  # estimate columns come first
    for i, (cell, title) in enumerate(zip(cells, titles)):
        tag = " [subtotal]" if i in children else ""
        lines.append(f"     {'  ' * depth[i]}{cell} {title} ({county[i]:,.0f}){tag}")
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
        groups.setdefault(home, []).append(cat.name)
    lines.append("   our categories by parent node (the sibling groups for rolling up):")
    for home, names in sorted(groups.items(), key=lambda kv: kv[0]):
        label = "(top)" if home < 0 else f"{cells[home]} {titles[home]}"
        lines.append(f"     {label}: {', '.join(names)}")
    frame = pd.DataFrame({"cell": cells, "title": titles, "depth": [depth[i] for i in range(len(cells))],
                          "parent": [cells[parent[i]] if i in parent else "" for i in range(len(cells))],
                          "subtotal": [i in children for i in range(len(cells))],
                          "county_estimate": county})
    return lines, frame


def main() -> None:
    args = parse_args()
    area = StudyArea(name=args.name, state=args.state, year=args.year,
                     counties=tuple(args.county), span=args.span)
    args.out.mkdir(parents=True, exist_ok=True)
    out_lines, frames = [], []
    for level in args.levels:
        specs = {t.table: t for t in default_tables(area) if t.geography == level}
        for table, spec in specs.items():
            frame = fetch_replicates(area, [table], geography=level)
            cells, titles, V = line_matrix(frame)
            out_lines.append(f"== {table} at {level}: {V.shape[1] // 81} areas")
            lines, rows = analyse(cells, titles, V, spec, args.tol)
            out_lines += lines + [""]
            frames.append(rows.assign(table=table, level=level))
    pd.concat(frames).to_csv(args.out / "table_trees.csv", index=False)
    text = "\n".join(out_lines) + "\n"
    (args.out / "table_trees.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
