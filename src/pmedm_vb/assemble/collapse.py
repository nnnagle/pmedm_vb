"""Per-area collapsing of small and zero cells, with sum-to-zero by merge group.

**The collapse.** For every area (the PUMA, each tract, each block group) and
every table, the table's categories are partitioned into cells:

1. all categories whose count in that area is zero become one cell, with
   target 0;
2. the positive cells are then merged smallest with next smallest (by count,
   ties by category order) until every positive cell holds more than
   ``threshold`` people (or households, for household tables).

It runs top-down. The PUMA is partitioned first, from the tract estimates
summed; each tract starts from its PUMA's partition and merges further only
if it must; each block group starts from its tract's partition (restricted to
the categories published at block group). So anything merged at a larger
scale is merged at every smaller one inside it, and a smaller area's cells are
unions of its parent's.

A merged cell's loading is the sum of its categories' loadings (they are
disjoint within a table) and its target the sum of their estimates. Its
replicates are the sums of theirs, so its replicate covariance with every
other cell is exact. Its published variance is its replicate variance plus the
non-replicate part of its members' (``v - s``, summed), except a zero cell,
which takes the largest of its members' variances -- a zero cell's variance is
the area's modelled zero-count variance, the same for any number of zero
categories -- or its replicate variance if that is larger.

**Coordinates.** The collapse is a 0/1 matrix ``C`` from today's stacked rows
to the cells, block diagonal by area and table (a ``C_a`` per area). The
solver's multipliers are one per cell, ``theta``, plus one per PUMA cell,
``lambda_P``, under ``"puma"``. The multipliers the data see, in today's
layout, are

    lambda_data = C' M theta + E_P lambda_P,

``E_P`` adding each PUMA cell's multiplier to the tract rows of its
categories. So ``p``, ``W`` and everything downstream read ``lambda_data``
unchanged, and this object offers the same interface as
:class:`~pmedm_vb.assemble.hierarchy.Hierarchy`.

**Sum-to-zero.** ``M`` removes group means (see ``hierarchy.py`` for the
ridge that makes the removed directions proper). A shift of the tract's
multipliers by ``delta`` on a set of categories ``G``, with every block group
multiplier in ``G`` shifted by ``-delta``, leaves the data unchanged exactly
when every block group can represent it: ``G`` must be a union of each block
group's cells. So within each tract the categories are grouped by
following merges -- two categories are in one group when some block group of
the tract holds them in one cell, and so on transitively -- and each group
gives one constraint: the block group multipliers of every cell inside it sum
to zero. A category merged nowhere keeps its own constraint, as without a
collapse. Under ``"puma"`` the same is done one level up, for tract cells
within the PUMA. Constraints and invisible directions are then equal in
number, so nothing the data inform is fixed by assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.rollup import RollupSpec, RollupStats, prune, rollup, tree_categories
from pmedm_vb.assemble.sigma import Sigma
from pmedm_vb.data.variance import SDR_FACTOR


def _table(name: str) -> str:
    return name.split(".", 1)[0]


def collapse_blocks(counts: np.ndarray, start: list[list[int]], threshold: float,
                    merge_zeros: bool = True) -> list[tuple[list[int], bool]]:
    """Partition categories into cells: ``(members, is_zero)`` per cell.

    ``counts`` per category; ``start`` the initial blocks (lists of category
    indices). Zero blocks merge into one; positive blocks merge smallest with
    next smallest until the smallest exceeds ``threshold``.
    """
    blocks = [sorted(b) for b in start if b]
    sums = [float(counts[b].sum()) for b in blocks]
    zero = [b for b, s in zip(blocks, sums) if s == 0]
    positive = [(s, b) for b, s in zip(blocks, sums) if s != 0]
    out: list[tuple[list[int], bool]] = []
    if zero:
        if merge_zeros:
            out.append((sorted(i for b in zero for i in b), True))
        else:
            out += [(b, True) for b in zero]
    while len(positive) > 1:
        positive.sort(key=lambda sb: (sb[0], sb[1][0]))
        if positive[0][0] > threshold:
            break
        (s1, b1), (s2, b2) = positive[0], positive[1]
        positive = [(s1 + s2, sorted(b1 + b2))] + positive[2:]
    out += [(b, False) for _, b in sorted(positive, key=lambda sb: sb[1][0])]
    return out


class _Partitioner:
    """The structure-aware roll-up for one table in one area, in
    :func:`collapse_blocks`' output form, with a tally per level and table."""

    def __init__(self, inputs: PMEDMInputs, spec: RollupSpec) -> None:
        if inputs.trees is None:
            raise ValueError(
                "the structure-aware roll-up needs the constraint tables' category trees, "
                "and these inputs have none: run experiments/table_trees.py --write-inputs "
                "on a node with network access")
        self.trees = inputs.trees
        self.spec = spec
        self.tally: dict[tuple[str, str], dict] = {}

    def __call__(self, counts: np.ndarray, names: list[str], cats: np.ndarray,
                 where: str) -> list[tuple[list[int], bool]]:
        table = _table(names[int(cats[0])])
        if table not in self.trees:
            raise ValueError(f"no category tree for {table}; rerun table_trees.py --write-inputs")
        entry = self.trees[table]
        position = {names[int(k)].split(".", 1)[1]: int(k) for k in cats}
        unknown = sorted(set(position) - set(tree_categories(entry["tree"])))
        if unknown:
            raise ValueError(f"{table}: categories {unknown} are not in its category tree")
        tree = prune(entry["tree"], set(position))
        order = tree_categories(tree)
        index = np.array([position[name] for name in order])
        threshold = self.spec.threshold(entry["universe"])
        parts, stats = rollup(tree, counts[index], threshold)
        row = self.tally.setdefault((where, table), dict(
            level=where, table=table, universe=entry["universe"], threshold=threshold,
            areas=0, categories=0, cells=0, zero_cells=0, zero_categories=0, stats=RollupStats()))
        row["areas"] += 1
        row["categories"] += len(order)
        row["cells"] += len(parts)
        row["zero_cells"] += sum(z for _, z in parts)
        row["zero_categories"] += int((counts[index] == 0).sum())
        row["stats"].add(stats)
        return [([int(index[i]) for i in members], zero) for members, zero in parts]

    def report(self) -> list[dict]:
        out = []
        for row in self.tally.values():
            stats = row.pop("stats")
            row.update(parallel_merges=stats.parallel_merges, collapsed_nodes=stats.collapsed_nodes,
                       crossings=stats.crossings, cells_below_threshold=stats.below_threshold)
            out.append(row)
        return out


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, members: list[int]) -> None:
        roots = [self.find(i) for i in members]
        for r in roots[1:]:
            self.parent[r] = roots[0]


@dataclass
class CollapsedHierarchy:
    """Cells, their targets and variances, and the maps between coordinates.

    Build with :meth:`build`. Rows of ``theta`` are the tract cells, then the
    block group cells; ``row_cell`` maps each of today's stacked rows to its
    cell, and ``puma_of_row`` each tract row to its PUMA cell (``-1``
    elsewhere, and everywhere below ``"puma"``).
    """

    level: str
    base: str
    threshold: float
    m: int
    m_full: int
    row_cell: np.ndarray
    puma_of_row: np.ndarray
    cell_area: np.ndarray        # taper group: the tract of every cell
    cell_is_tract: np.ndarray
    cell_zero: np.ndarray
    cell_names: list[str]
    group: np.ndarray            # sum-to-zero group per cell, -1 for none
    bg_group: np.ndarray
    Y_c: np.ndarray
    v_c: np.ndarray
    l_c: np.ndarray
    floor_c: np.ndarray
    Y_P: np.ndarray
    v_P: np.ndarray
    l_P: np.ndarray | None
    puma_names: list[str]
    y_ext: np.ndarray
    kappa: float
    n_tracts: int
    tract_cells: list[np.ndarray]  # theta rows of each tract's block: its cells and its block groups'

    # -- construction ------------------------------------------------------

    @classmethod
    def build(cls, inputs: PMEDMInputs, base: str, threshold: "float | RollupSpec",
              merge_zeros: bool = True) -> "CollapsedHierarchy":
        """``threshold`` a number: the collapse above, top-down. A
        :class:`~pmedm_vb.assemble.rollup.RollupSpec`: the structure-aware
        roll-up of :mod:`pmedm_vb.assemble.rollup`, run on each area's own
        counts (the PUMA, each tract, each block group) independently."""
        rolled = isinstance(threshold, RollupSpec)
        if base not in ("none", "tract", "puma"):
            raise ValueError(f"base level must be none, tract or puma, got {base!r}")
        if inputs.sigma_l is None:
            raise ValueError("collapsing needs the replicate deviations, and sigma_l is None")
        inputs.constraint_tracts()  # raises unless A_B is the identity
        n_tracts, c_t = inputs.Y_T.shape
        n_zones, c_b = inputs.Y_B.shape
        split, m_full = inputs.Y_T.size, inputs.n_constraints
        zone_tract = inputs.zone_tracts()
        t_names, b_names = list(inputs.tract_constraints), list(inputs.bg_constraints)
        tract_index = {name: k for k, name in enumerate(t_names)}
        missing = [n for n in b_names if n not in tract_index]
        if missing:
            raise ValueError(f"block group categories with no tract row: {missing}")
        b_to_t = np.array([tract_index[n] for n in b_names], dtype=int)
        t_tables = np.array([_table(n) for n in t_names])
        b_tables = np.array([_table(n) for n in b_names])
        big = np.inf if threshold is None else threshold
        if rolled:
            partition = _Partitioner(inputs, threshold)

            def cells_of(counts, names, cats, start, where):
                return partition(counts, names, cats, where)
        else:
            def cells_of(counts, names, cats, start, where):
                return collapse_blocks(counts, start, big, merge_zeros)

        # PUMA partition per table, from the tract estimates summed.
        puma_counts = inputs.Y_T.sum(axis=0)
        puma_blocks = {}
        for table in dict.fromkeys(t_tables):
            cats = np.flatnonzero(t_tables == table)
            blocks = cells_of(puma_counts, t_names, cats, [[int(k)] for k in cats], "puma")
            puma_blocks[table] = blocks

        cells: list[dict] = []
        row_cell = np.full(m_full, -1)
        # Tract cells: start from the PUMA partition.
        tract_blocks = {}
        for t in range(n_tracts):
            counts = inputs.Y_T[t]
            for table, blocks in puma_blocks.items():
                start = [b for b, _ in blocks]
                parts = cells_of(counts, t_names, np.flatnonzero(t_tables == table), start, "tract")
                tract_blocks[t, table] = [b for b, _ in parts]
                for members, zero in parts:
                    rows = np.array(members) * n_tracts + t
                    row_cell[rows] = len(cells)
                    cells.append(dict(rows=rows, tract=t, is_tract=True, zero=zero,
                                      name=f"T{t}:" + "+".join(t_names[k] for k in members)))
        # Block group cells: start from the tract's partition, restricted.
        bg_parts = {}
        for z in range(n_zones):
            t = int(zone_tract[z])
            counts = inputs.Y_B[z]
            for table in dict.fromkeys(b_tables):
                b_cats = np.flatnonzero(b_tables == table)
                t_to_b = {int(b_to_t[kb]): int(kb) for kb in b_cats}
                start = [[t_to_b[k] for k in blk if k in t_to_b] for blk in tract_blocks[t, table]]
                parts = cells_of(counts, b_names, b_cats, start, "block group")
                bg_parts[z, table] = parts
                for members, zero in parts:
                    rows = split + np.array(members) * n_zones + z
                    row_cell[rows] = len(cells)
                    cells.append(dict(rows=rows, tract=t, is_tract=False, zero=zero,
                                      name=f"B{z}:" + "+".join(b_names[k] for k in members)))
        # Order: tract cells first, then block group cells (already so).
        m = len(cells)
        assert (row_cell >= 0).all()

        # Targets, replicates, variances.
        targets = inputs.targets().astype(float)
        rep_row = SDR_FACTOR * np.square(inputs.sigma_l).sum(axis=1)
        floor_row = inputs.zero_cell_variances()
        Y_c = np.zeros(m)
        np.add.at(Y_c, row_cell, targets)
        l_c = np.zeros((m, inputs.sigma_l.shape[1]))
        np.add.at(l_c, row_cell, inputs.sigma_l)
        s_c = SDR_FACTOR * np.square(l_c).sum(axis=1)
        excess = np.zeros(m)
        np.add.at(excess, row_cell, np.maximum(inputs.sigma_v - rep_row, 0.0))
        max_v = np.zeros(m)
        np.maximum.at(max_v, row_cell, inputs.sigma_v)
        floor_c = np.zeros(m)
        np.maximum.at(floor_c, row_cell, floor_row)
        cell_zero = np.array([c["zero"] for c in cells])
        v_c = np.where(cell_zero, np.maximum(max_v, s_c), s_c + excess)

        cell_area = np.array([c["tract"] for c in cells])
        cell_is_tract = np.array([c["is_tract"] for c in cells])

        # Sum-to-zero groups.
        group = np.full(m, -1)
        next_id = 0
        if base in ("tract", "puma"):
            for t in range(n_tracts):
                zones = np.flatnonzero(zone_tract == t)
                for table in dict.fromkeys(b_tables):
                    b_cats = np.flatnonzero(b_tables == table)
                    position = {int(k): i for i, k in enumerate(b_cats)}
                    uf = _UnionFind(b_cats.size)
                    for z in zones:
                        for members, _ in bg_parts[z, table]:
                            uf.union([position[k] for k in members])
                    # the tract's own cells too: a no-op under the top-down
                    # collapse, where block group cells are unions of them
                    t_to_b = {int(b_to_t[kb]): int(kb) for kb in b_cats}
                    for members in tract_blocks[t, table]:
                        uf.union([position[t_to_b[k]] for k in members if k in t_to_b])
                    roots = {}
                    for z in zones:
                        for members, _ in bg_parts[z, table]:
                            root = uf.find(position[members[0]])
                            if root not in roots:
                                roots[root] = next_id
                                next_id += 1
                            cell = row_cell[split + members[0] * n_zones + z]
                            group[cell] = roots[root]
        bg_group = group.copy()

        puma_of_row = np.full(m_full, -1)
        Y_P = np.zeros(0)
        v_P = np.zeros(0)
        l_P = None
        puma_names: list[str] = []
        if base == "puma":
            # Tract cells: groups by following merges across the PUMA's tracts.
            for table in dict.fromkeys(t_tables):
                cats = np.flatnonzero(t_tables == table)
                position = {int(k): i for i, k in enumerate(cats)}
                uf = _UnionFind(cats.size)
                for t in range(n_tracts):
                    for members in tract_blocks[t, table]:
                        uf.union([position[k] for k in members])
                for members, _ in puma_blocks[table]:
                    uf.union([position[k] for k in members])
                roots = {}
                for t in range(n_tracts):
                    for members in tract_blocks[t, table]:
                        root = uf.find(position[members[0]])
                        if root not in roots:
                            roots[root] = next_id
                            next_id += 1
                        group[row_cell[members[0] * n_tracts + t]] = roots[root]
            # PUMA cells: the PUMA partition, from the tract rows summed.
            l_rows = [inputs.sigma_l[k * n_tracts + np.arange(n_tracts)].sum(axis=0)
                      for k in range(c_t)]
            l_cat = np.stack(l_rows)
            s_cat = SDR_FACTOR * np.square(l_cat).sum(axis=1)
            tract_v = inputs.sigma_v[:split].reshape(inputs.Y_T.shape, order="F").sum(axis=0)
            v_cat = np.where(s_cat > 0, s_cat, tract_v)
            Ys, vs, ls = [], [], []
            for table, blocks in puma_blocks.items():
                for members, zero in blocks:
                    idx = len(Ys)
                    for k in members:
                        puma_of_row[k * n_tracts + np.arange(n_tracts)] = idx
                    l_sum = l_cat[members].sum(axis=0)
                    s_sum = SDR_FACTOR * np.square(l_sum).sum()
                    if zero:
                        v = max(float(v_cat[members].max()), s_sum)
                    else:
                        v = s_sum + float(np.maximum(v_cat[members] - s_cat[members], 0).sum())
                    Ys.append(float(puma_counts[members].sum()))
                    vs.append(v)
                    ls.append(l_sum)
                    puma_names.append("P:" + "+".join(t_names[k] for k in members))
            Y_P, v_P, l_P = np.array(Ys), np.array(vs), np.stack(ls)
        # Dense group numbering.
        used = np.unique(group[group >= 0])
        remap = np.full(max(next_id, 1), -1)
        remap[used] = np.arange(used.size)
        group = np.where(group >= 0, remap[np.maximum(group, 0)], -1)
        bg_group = np.where(bg_group >= 0, remap[np.maximum(bg_group, 0)], -1)

        tract_cells = [np.flatnonzero(cell_area == t) for t in range(n_tracts)]
        if rolled:
            level = f"{base}-r{threshold}"
        else:
            level = f"{base}-c{threshold:g}" if merge_zeros else f"{base}-c{threshold:g}-keepzeros"
        self = cls(level=level, base=base, threshold=threshold, m=m, m_full=m_full,
                   row_cell=row_cell, puma_of_row=puma_of_row, cell_area=cell_area,
                   cell_is_tract=cell_is_tract, cell_zero=cell_zero,
                   cell_names=[c["name"] for c in cells], group=group, bg_group=bg_group,
                   Y_c=Y_c, v_c=v_c, l_c=l_c, floor_c=floor_c, Y_P=Y_P, v_P=v_P, l_P=l_P,
                   puma_names=puma_names,
                   y_ext=np.concatenate([Y_c, Y_P]) / inputs.N, kappa=1.0 / inputs.n,
                   n_tracts=n_tracts, tract_cells=tract_cells)
        self.rollup_report = partition.report() if rolled else None
        return self

    # -- sizes -------------------------------------------------------------

    @property
    def n_puma(self) -> int:
        return self.Y_P.size

    @property
    def size(self) -> int:
        return self.m + self.n_puma

    is_trivial = False
    is_collapsed = True

    @property
    def targets_ext(self) -> np.ndarray:
        return np.concatenate([self.Y_c, self.Y_P])

    @property
    def v_ext(self) -> np.ndarray:
        return np.concatenate([self.v_c, self.v_P])

    def _counts(self, group: np.ndarray) -> np.ndarray:
        return np.bincount(group[group >= 0], minlength=int(group.max()) + 1 if group.max() >= 0 else 0)

    # -- numpy maps --------------------------------------------------------

    def project(self, theta: np.ndarray, which: str = "all") -> np.ndarray:
        from pmedm_vb.assemble.hierarchy import _group_means

        group = self.group if which == "all" else self.bg_group
        if group.max() < 0:
            return np.array(theta, dtype=float)
        return theta - _group_means(theta, group, self._counts(group))

    def null(self, theta: np.ndarray) -> np.ndarray:
        return theta - self.project(theta)

    def collapse(self, g: np.ndarray) -> np.ndarray:
        """``C g``: sum today's rows into cells; ``g`` is ``(m_full,)`` or ``(m_full, k)``."""
        out = np.zeros((self.m,) + g.shape[1:])
        np.add.at(out, self.row_cell, g)
        return out

    def puma_sum(self, g: np.ndarray) -> np.ndarray:
        """``E_P' g``: per PUMA cell, the sum of ``g`` over its tract rows."""
        rows = self.puma_of_row >= 0
        out = np.zeros((self.n_puma,) + g.shape[1:])
        np.add.at(out, self.puma_of_row[rows], g[rows])
        return out

    def lambda_data(self, xi: np.ndarray) -> np.ndarray:
        lam = self.project(xi[: self.m])[self.row_cell]
        if self.n_puma:
            rows = self.puma_of_row >= 0
            lam[rows] += xi[self.m:][self.puma_of_row[rows]]
        return lam

    def lambda_data_T(self, g: np.ndarray) -> np.ndarray:
        return np.concatenate([self.project(self.collapse(g)), self.puma_sum(g)], axis=0)

    def zeta(self, xi: np.ndarray) -> np.ndarray:
        return np.concatenate([self.project(xi[: self.m]), xi[self.m:]], axis=0)

    zeta_T = zeta

    def ridge_gradient(self, xi: np.ndarray) -> np.ndarray:
        out = np.zeros_like(xi, dtype=float)
        out[: self.m] = self.kappa * self.null(xi[: self.m])
        return out

    def extend(self, stacked: np.ndarray) -> np.ndarray:
        """Today's stacked values as cell values, PUMA cells appended: ``(C g, E_P' g)``."""
        return np.concatenate([self.collapse(stacked), self.puma_sum(stacked)], axis=0)

    def tract_group_basis(self) -> np.ndarray:
        """``(size, g)``: unit-norm indicators of the tract-cell groups, whose
        span is the part of ``I - M`` coupling tracts."""
        tract_groups = np.unique(self.group[self.cell_is_tract & (self.group >= 0)])
        U = np.zeros((self.size, tract_groups.size))
        for j, label in enumerate(tract_groups):
            members = np.flatnonzero(self.group == label)
            U[members, j] = 1.0 / np.sqrt(members.size)
        return U

    def local_projector(self, rows: np.ndarray) -> np.ndarray:
        g = self.bg_group[rows]
        Q = np.zeros((rows.size, rows.size))
        for label in np.unique(g[g >= 0]):
            members = np.flatnonzero(g == label)
            Q[np.ix_(members, members)] = 1.0 / members.size
        return Q

    # -- Sigma -------------------------------------------------------------

    def sigma(self, inputs: PMEDMInputs, alpha: float, taper: str | None,
              variance_floor: str | float | None) -> Sigma:
        """``Sigma`` over the cells, then the PUMA cells as their own taper group."""
        if taper not in ("tract", None):
            raise ValueError(f"taper must be 'tract' or None, got {taper!r}")
        v = self.v_c
        if variance_floor == "zero":
            v = np.maximum(v, self.floor_c)
        elif variance_floor is not None:
            v = np.maximum(v, float(variance_floor))
        v = np.concatenate([v, self.v_P])
        l = self.l_c if self.l_P is None else np.vstack([self.l_c, self.l_P])
        groups = None
        if taper == "tract":
            groups = np.concatenate([self.cell_area, np.full(self.n_puma, self.n_tracts)])
        return Sigma(v=v, l=l, alpha=alpha, groups=groups)

    def torch_maps(self, device: str):
        return _TorchCollapsed(self, device)


class _TorchCollapsed:
    def __init__(self, h: CollapsedHierarchy, device: str) -> None:
        import torch

        from pmedm_vb.assemble.hierarchy import _TorchHierarchy

        self.torch = torch
        self.m, self.n_puma, self.kappa = h.m, h.n_puma, h.kappa
        # Reuse the group-mean projection of the uncollapsed maps.
        self._proj = _TorchHierarchy.__new__(_TorchHierarchy)
        grouped = np.flatnonzero(h.group >= 0)
        self._proj.torch = torch
        self._proj.grouped = torch.as_tensor(grouped, device=device)
        self._proj.group = torch.as_tensor(h.group[grouped], device=device)
        counts = h._counts(h.group) if grouped.size else np.zeros(0)
        self._proj.counts = torch.as_tensor(counts, dtype=torch.float64, device=device)
        self.row_cell = torch.as_tensor(h.row_cell, device=device)
        tract = np.flatnonzero(h.puma_of_row >= 0)
        self.tract_rows = torch.as_tensor(tract, device=device)
        self.tract_puma = torch.as_tensor(h.puma_of_row[tract], device=device)

    def project(self, theta):
        return _project(self._proj, theta)

    def lambda_data(self, xi):
        lam = self.project(xi[:, : self.m])[:, self.row_cell]
        if self.n_puma:
            lam = lam.index_add(1, self.tract_rows, xi[:, self.m:][:, self.tract_puma])
        return lam

    def zeta(self, xi):
        return self.torch.cat([self.project(xi[:, : self.m]), xi[:, self.m:]], dim=1)

    def ridge(self, xi):
        theta = xi[:, : self.m]
        null = theta - self.project(theta)
        return 0.5 * self.kappa * (null * null).sum(1)


def _project(p, theta):
    from pmedm_vb.assemble.hierarchy import _TorchHierarchy

    return _TorchHierarchy.project(p, theta)
