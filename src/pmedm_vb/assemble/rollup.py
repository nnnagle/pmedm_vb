"""Structure-aware roll-up of small cells, one area and one table at a time.

**The category tree.** A published table's lines form a tree: each line whose
title ends in ":" is the exact sum of the subtrees below it (see
``experiments/table_trees.py``, which reads it off the national replicate file).
Our categories are unions of published lines, often coarser than the leaves,
and sometimes crossing the published nodes (``B23001``'s age bands each span
several published age groups). So the roll-up works on a tree over *our
categories*: for every published node, the set of our categories with a line
under it. Those sets that nest (any two are nested or disjoint) are the nodes;
a set that crosses another is dropped, and its categories fall to the next
enclosing node. A node that no single published line gives -- ``B23001``'s
"male 16 to 24", the union of three published age groups -- is still exact,
since each category is.

**Parallel nodes.** Adjacent sibling nodes with the same shape, whose
corresponding categories share a published title (``B17024``'s age groups,
each split into below / near / above poverty), form a *block*: a two-way table
whose outer dimension is the siblings and inner dimension their shared shape.

**The roll-up.** For one area's counts, bottom-up:

- A node's zero categories (and all-zero child nodes) form one zero cell. A
  zero cell counts as a class: a published zero is exact.
- A child node below the threshold is merged into one cell. A child node at or
  above it keeps its cells and is not merged with its siblings.
- Cells below the threshold merge with an adjacent sibling cell, the pair
  chosen by the least information lost,

      loss(a, b) = (a + b) ln(a + b) - a ln a - b ln b,

  i.e. N times the drop in the table's entropy. A cell with no sibling cell
  beside it (only child nodes) merges into the nearest cell of the adjacent
  child node.
- In a block, every outer group must keep at least two classes. Outer groups
  (runs of adjacent siblings, starting with each alone) merge in parallel --
  inner category ``k`` of one with inner category ``k`` of the other -- when
  that makes more groups feasible, or loses less information overall; the
  outer dimension is merged away entirely only if nothing else makes the
  block feasible.
- The threshold is hard: where nothing else is possible a node's cells merge
  below two classes.

The result is a partition of the table's categories into cells for that area.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# -- thresholds ------------------------------------------------------------


@dataclass(frozen=True)
class RollupSpec:
    """Thresholds of the roll-up: persons, and households (or housing units)."""

    person: float
    household: float

    @classmethod
    def parse(cls, text: str) -> "RollupSpec":
        """``"15"`` or ``"15h10"`` (the part after ``-r`` in a level string)."""
        person, _, household = text.partition("h")
        p = float(person)
        return cls(person=p, household=float(household) if household else p)

    def __str__(self) -> str:
        return f"{self.person:g}" + ("" if self.household == self.person else f"h{self.household:g}")

    def threshold(self, universe: str) -> float:
        if universe == "person":
            return self.person
        if universe == "household":
            return self.household
        raise ValueError(f"no roll-up threshold for universe {universe!r}")


# -- the category tree ---------------------------------------------------------


def category_tree(children: dict[int, list[int]], cells: list[str], titles: list[str],
                  categories: list[tuple[str, tuple[str, ...]]]) -> tuple[dict, list[list[str]]]:
    """The tree over our categories, and the category sets dropped for crossing.

    ``children`` is the published tree over line indices (from the parse of
    ``experiments/table_trees.py``), ``cells`` and ``titles`` the lines, and
    ``categories`` ``(name, published cells)`` in the table's order. A node is
    ``{"children": [...]}``, a leaf ``{"cat": name, "titles": [...]}``; the root
    is always a node.
    """
    index = {c: i for i, c in enumerate(cells)}

    def leaves(i: int) -> frozenset:
        if i not in children:
            return frozenset([i])
        return frozenset().union(*(leaves(k) for k in children[i]))

    names = [name for name, _ in categories]
    cat_leaves = {}
    for name, published in categories:
        missing = [c for c in published if c not in index]
        if missing:
            raise ValueError(f"category {name} names lines not in the table: {missing}")
        cat_leaves[name] = frozenset().union(*(leaves(index[c]) for c in published))
    first = {name: min(cat_leaves[name]) for name in names}

    sets = {frozenset(names)} | {frozenset([n]) for n in names}
    for i in range(len(cells)):
        under = leaves(i)
        touching = frozenset(n for n in names if cat_leaves[n] & under)
        if touching:
            sets.add(touching)
    crossing = {a for a in sets for b in sets
                if a & b and not a <= b and not b <= a}
    kept = sets - crossing
    dropped = sorted((sorted(s, key=first.get) for s in crossing), key=lambda s: first[s[0]])

    def build(members: frozenset) -> dict:
        inside = [s for s in kept if s < members]
        maximal = [s for s in inside if not any(s < t for t in inside)]
        maximal.sort(key=lambda s: min(first[n] for n in s))
        return {"children": [leaf(s) if len(s) == 1 else build(s) for s in maximal]}

    def leaf(s: frozenset) -> dict:
        (name,) = s
        return {"cat": name, "titles": sorted({titles[i] for i in cat_leaves[name]})}

    root = frozenset(names)
    if len(root) == 1:
        # a one-category table: its tree is that category alone
        return {"children": [leaf(root)]}, dropped
    return build(root), dropped


def _shape(node: dict) -> str:
    if "cat" in node:
        return "L"
    return "(" + ",".join(_shape(c) for c in node["children"]) + ")"


def _leaves(node: dict) -> list[dict]:
    if "cat" in node:
        return [node]
    return [leaf for c in node["children"] for leaf in _leaves(c)]


def _parallel(a: dict, b: dict) -> bool:
    """Same shape, and each pair of corresponding categories shares a title."""
    if "cat" in a or "cat" in b or _shape(a) != _shape(b):
        return False
    return all(set(x["titles"]) & set(y["titles"]) for x, y in zip(_leaves(a), _leaves(b)))


def prune(tree: dict, keep: set[str]) -> dict:
    """The tree restricted to the categories in ``keep`` (a reduced area drops
    some); nodes left empty go, and a node left with one child is that child."""

    def walk(node: dict) -> dict | None:
        if "cat" in node:
            return node if node["cat"] in keep else None
        kids = [k for k in (walk(c) for c in node["children"]) if k is not None]
        if not kids:
            return None
        return kids[0] if len(kids) == 1 and "cat" not in kids[0] else {"children": kids}

    out = walk(tree)
    if out is None:
        return {"children": []}
    return out if "children" in out else {"children": [out]}


def tree_categories(tree: dict) -> list[str]:
    """The tree's categories in its (published) order."""
    return [leaf["cat"] for leaf in _leaves(tree)]


# -- the roll-up -------------------------------------------------------------


def _f(x: float) -> float:
    return x * math.log(x) if x > 0 else 0.0


@dataclass
class _Result:
    """One node's cells: positive ``(categories, count)`` in order, and its zero cell."""

    cells: list[tuple[list[int], float]]
    zero: list[int]
    feasible: bool

    @property
    def classes(self) -> int:
        return len(self.cells) + (1 if self.zero else 0)


@dataclass
class RollupStats:
    """What the roll-up did, summed over the calls that share it."""

    parallel_merges: int = 0     # outer groups merged in a block
    collapsed_nodes: int = 0     # child nodes below the threshold made one cell
    crossings: int = 0           # cells merged into an adjacent child node's cell
    below_threshold: int = 0     # positive cells left below the threshold
    extra: dict = field(default_factory=dict)

    def add(self, other: "RollupStats") -> None:
        self.parallel_merges += other.parallel_merges
        self.collapsed_nodes += other.collapsed_nodes
        self.crossings += other.crossings
        self.below_threshold += other.below_threshold


class _Rollup:
    def __init__(self, tree: dict, counts: np.ndarray, threshold: float) -> None:
        self.tree = tree
        self.counts = np.asarray(counts, dtype=float)
        self.T = threshold
        self.order = {name: k for k, name in enumerate(tree_categories(tree))}
        self.stats = RollupStats()

    def pop(self, cats: list[int]) -> float:
        return float(self.counts[cats].sum()) if cats else 0.0

    # A node is solved over "slots": one list of categories per leaf of its
    # shape, in order -- one category for a real node, one per member for a
    # group of parallel nodes merged together.

    def solve(self, node: dict, slots: list[list[int]], keep2: bool) -> _Result:
        items: list[tuple] = []      # ("leaf", cats, count) | ("barrier", _Result)
        zero: list[int] = []
        offset = 0
        kids = node["children"]
        k = 0
        while k < len(kids):
            child = kids[k]
            if "cat" in child:
                cats = slots[offset]
                offset += 1
                count = self.pop(cats)
                if count == 0:
                    zero += cats
                else:
                    items.append(("leaf", list(cats), count))
                k += 1
                continue
            # a run of parallel child nodes is a block
            run = [child]
            while k + len(run) < len(kids) and _parallel(run[-1], kids[k + len(run)]):
                run.append(kids[k + len(run)])
            member_slots = []
            for member in run:
                width = len(_leaves(member))
                member_slots.append(slots[offset: offset + width])
                offset += width
            if len(run) == 1:
                groups = [(run[0], member_slots[0], None)]
            else:
                groups = self.solve_block(run[0], member_slots)
            for shape, sub_slots, solved in groups:
                cats = [c for s in sub_slots for c in s]
                count = self.pop(cats)
                if count == 0:
                    zero += cats
                elif count < self.T:
                    self.stats.collapsed_nodes += 1
                    items.append(("leaf", cats, count))
                else:
                    if solved is None or not solved.feasible:
                        solved = self.solve(shape, sub_slots, keep2=False)
                    items.append(("barrier", solved))
            k += len(run)

        self.merge_items(items, keep2, zero)
        cells: list[tuple[list[int], float]] = []
        feasible = True
        for item in items:
            if item[0] == "leaf":
                cells.append((item[1], item[2]))
                feasible &= item[2] >= self.T
            else:
                cells += item[1].cells
                zero += item[1].zero
                feasible &= item[1].feasible
        result = _Result(cells=cells, zero=zero, feasible=feasible)
        if keep2 and result.classes < 2:
            result.feasible = False
        return result

    def merge_items(self, items: list[tuple], keep2: bool, zero: list[int]) -> None:
        """Merge small leaf items, in place: adjacent leaf pairs by least loss, then
        into the nearest cell of an adjacent child node."""

        def classes() -> int:
            n = sum(1 if it[0] == "leaf" else it[1].classes for it in items)
            return n + (1 if zero else 0)

        while True:
            best = None
            for i in range(len(items) - 1):
                a, b = items[i], items[i + 1]
                if a[0] != "leaf" or b[0] != "leaf":
                    continue
                if min(a[2], b[2]) >= self.T:
                    continue
                loss = _f(a[2] + b[2]) - _f(a[2]) - _f(b[2])
                if best is None or loss < best[0]:
                    best = (loss, i)
            if best is not None and (not keep2 or classes() - 1 >= 2):
                i = best[1]
                a, b = items[i], items[i + 1]
                items[i: i + 2] = [("leaf", a[1] + b[1], a[2] + b[2])]
                continue
            # a small leaf with no leaf beside it: into the adjacent node's nearest cell
            best = None
            for i, it in enumerate(items):
                if it[0] != "leaf" or it[2] >= self.T:
                    continue
                for j, end in ((i - 1, -1), (i + 1, 0)):
                    if 0 <= j < len(items) and items[j][0] == "barrier" and items[j][1].cells:
                        cats, count = items[j][1].cells[end]
                        loss = _f(count + it[2]) - _f(count) - _f(it[2])
                        if best is None or loss < best[0]:
                            best = (loss, i, j, end)
            if best is None or (keep2 and classes() - 1 < 2):
                return
            _, i, j, end = best
            leaf, barrier = items[i], items[j][1]
            cats, count = barrier.cells[end]
            barrier.cells[end] = (cats + leaf[1], count + leaf[2])
            del items[i]
            self.stats.crossings += 1

    def solve_block(self, shape: dict, member_slots: list[list[list[int]]]
                    ) -> list[tuple[dict, list[list[int]], _Result]]:
        """Group the block's members into runs; each run solved as one node."""

        def group_slots(run: list[int]) -> list[list[int]]:
            return [[c for m in run for c in member_slots[m][k]] for k in range(len(member_slots[0]))]

        cache: dict[tuple[int, ...], _Result] = {}

        def solve_run(run: tuple[int, ...]) -> _Result:
            if run not in cache:
                saved = self.stats
                self.stats = RollupStats()   # only the chosen grouping's work is counted
                cache[run] = (self.solve(shape, group_slots(list(run)), keep2=True), self.stats)
                self.stats = saved
            return cache[run][0]

        def score(grouping: list[tuple[int, ...]]) -> tuple:
            results = [solve_run(run) for run in grouping]
            infeasible = sum(not r.feasible for r in results)
            loss = sum(self.loss(r) for r in results)
            return (infeasible, len(grouping) == 1, loss)

        grouping = [(m,) for m in range(len(member_slots))]
        current = score(grouping)
        while len(grouping) > 1:
            options = []
            for i in range(len(grouping) - 1):
                merged = grouping[:i] + [grouping[i] + grouping[i + 1]] + grouping[i + 2:]
                options.append((score(merged), merged))
            key, merged = min(options, key=lambda o: o[0])
            if key >= current:
                break
            grouping, current = merged, key
            self.stats.parallel_merges += 1
        for run in grouping:
            self.stats.add(cache[run][1])
        return [(shape, group_slots(list(run)), cache[run][0]) for run in grouping]

    def loss(self, result: _Result) -> float:
        """Information lost by ``result``'s cells against its categories alone."""
        members = [c for cats, _ in result.cells for c in cats]
        return (sum(_f(count) for _, count in result.cells)
                - sum(_f(self.counts[c]) for c in members))


def rollup(tree: dict, counts: np.ndarray, threshold: float
           ) -> tuple[list[tuple[list[int], bool]], RollupStats]:
    """Partition one area's categories of one table into cells.

    ``counts`` are the area's estimates of the tree's categories, in
    :func:`tree_categories` order. Returns ``(members, is_zero)`` per cell, the
    members as indices into that order, zero cells first, then the positive
    cells in order; and what the roll-up did.
    """
    job = _Rollup(tree, counts, threshold)
    slots = [[k] for k in range(len(job.order))]
    result = job.solve(tree, slots, keep2=False)
    total = job.pop(list(range(len(job.order))))
    cells = [(sorted(c), False) for c, _ in result.cells]
    zero = sorted(result.zero)
    if 0 < total < threshold:
        # a table below the threshold in this area: one cell, its zeros absorbed
        cells, zero = [(list(range(len(job.order))), False)], []
    out = ([(zero, True)] if zero else []) + cells
    job.stats.below_threshold = sum(1 for c, z in out if not z and job.pop(c) < threshold)
    return out, job.stats
