"""Which multiplier directions can the data not see, and do the tables agree there?

The data see each unit's tilt in each zone, ``sum_row x_i,row lambda_row``
over the rows that apply to that zone (its tract's and its own). A direction
``v`` of the multipliers that changes no tilt is invisible to the data: the
fit along it is set by ``Sigma`` alone, and removing it (a sum-to-zero
constraint) also removes the published combination ``v'Y``. That is harmless
exactly when ``v'Y`` is 0 with no sampling spread, as for a tract against its
block groups.

For each tract (the data map is block diagonal by tract; PUMA-level
directions are not examined), this finds the null space of that map over the
units each zone can hold (``q > 0``) and splits it into

- **hierarchy** directions: raise the tract's multipliers on a set of
  categories, lower every block group's on the same set -- one per category
  constrained at both levels, or under a collapse one per merge group;
- **other** directions: the rest of the null space, e.g. two tables sharing a
  universe or a margin (households in B19001 and B25003), whose involved
  tables are listed.

For each kind it reports the published contrasts ``V'Y`` and their replicate
spread (``2 V'(Y_r - Y)``, one sampling error per replicate under SDR's
``4/80``), over an orthonormal basis ``V`` of that part. Near 0 in both means
removing those directions loses nothing. The ``other`` basis is arbitrary
within its span, so a direction may mix several table pairs; the table list
is the union.

``--collapse T`` does the same on the per-area collapsed cells of
:mod:`pmedm_vb.assemble.collapse`.

Writes ``<out>/null_directions.txt`` and ``<out>/null_directions.csv`` (one row
per direction)::

    $CONDA_PREFIX/bin/python experiments/null_directions.py \\
        --puma 4701501 4701502 4701503 4701504 --collapse none 15 --out $RUNS/null_directions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.data.variance import SDR_FACTOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--puma", nargs="+", required=True)
    parser.add_argument("--area", default="knox-2024-5yr")
    parser.add_argument("--collapse", nargs="+", default=["none"],
                        help="'none' and/or thresholds, one analysis each")
    parser.add_argument("--rtol", type=float, default=1e-9,
                        help="singular values below rtol x the largest count as zero")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _table(name: str) -> str:
    return name.split(".", 1)[0]


class Layout:
    """Coordinates of one analysis: identity rows, or collapsed cells."""

    def __init__(self, inputs: PMEDMInputs, collapse: float | None) -> None:
        self.inputs = inputs
        n_tracts, c_t = inputs.Y_T.shape
        n_zones, c_b = inputs.Y_B.shape
        split = inputs.Y_T.size
        self.zone_tract = inputs.zone_tracts()
        self.t_names, self.b_names = list(inputs.tract_constraints), list(inputs.bg_constraints)
        row_name = ([self.t_names[r // n_tracts] for r in range(split)]
                    + [self.b_names[(r - split) // n_zones] for r in range(split, inputs.n_constraints)])
        self.row_level = np.array(["tract"] * split + ["block group"] * (inputs.n_constraints - split))
        self.row_table = np.array([_table(n) for n in row_name])
        self.row_category = np.array(row_name)
        if collapse is None:
            self.cell_of_row = np.arange(inputs.n_constraints)
            self.n_cells = inputs.n_constraints
            self.Y = inputs.targets().astype(float)
            self.L = inputs.sigma_l
            self.bg_group = None
        else:
            from pmedm_vb.assemble.collapse import CollapsedHierarchy

            h = CollapsedHierarchy.build(inputs, "tract", collapse)
            self.cell_of_row = h.row_cell
            self.n_cells = h.m
            self.Y, self.L = h.Y_c, h.l_c
            self.bg_group = h.bg_group
        self.n_tracts, self.n_zones, self.split = n_tracts, n_zones, split

    def tract_rows(self, t: int) -> np.ndarray:
        """Today's rows local to tract ``t``: its own, then its block groups'."""
        c_t, c_b = len(self.t_names), len(self.b_names)
        zones = np.flatnonzero(self.zone_tract == t)
        return np.concatenate([np.arange(c_t) * self.n_tracts + t]
                              + [self.split + np.arange(c_b) * self.n_zones + z for z in zones])


def tract_analysis(layout: Layout, t: int, x_t, x_b, rtol: float):
    """``(directions, cells, rank, nullity)`` for tract ``t``."""
    inputs = layout.inputs
    rows = layout.tract_rows(t)
    cells = np.unique(layout.cell_of_row[rows])
    position = {int(c): i for i, c in enumerate(cells)}
    col_of_row = np.array([position[int(layout.cell_of_row[r])] for r in rows])
    c_t = len(layout.t_names)
    zones = np.flatnonzero(layout.zone_tract == t)
    c_b = len(layout.b_names)
    blocks = []
    for i, z in enumerate(zones):
        support = np.flatnonzero(inputs.q[z] > 0)
        # tilt of each supported unit: tract rows + this zone's block group rows
        full = sp.hstack([x_t[support], sp.csr_matrix((support.size, i * c_b)), x_b[support],
                          sp.csr_matrix((support.size, (len(zones) - 1 - i) * c_b))]).tocsc()
        collapse = sp.csr_matrix((np.ones(rows.size), (np.arange(rows.size), col_of_row)),
                                 shape=(rows.size, cells.size))
        blocks.append((full @ collapse).toarray())
    D = np.vstack(blocks)
    _, s, vt = np.linalg.svd(D, full_matrices=D.shape[0] < D.shape[1])  # vt square either way
    rank = int((s > rtol * s.max()).sum()) if s.size else 0
    null = vt[rank:].T                                   # (cells, dim null)

    # Hierarchy directions: tract +1, block groups -1, on each category (or merge group).
    cell_rows = {i: rows[col_of_row == i] for i in range(cells.size)}
    is_tract_cell = np.array([layout.row_level[cell_rows[i][0]] == "tract" for i in range(cells.size)])
    cell_cats = [set(layout.row_category[cell_rows[i]]) for i in range(cells.size)]
    if layout.bg_group is None:
        groups = {}
        for i in np.flatnonzero(~is_tract_cell):
            groups.setdefault(next(iter(cell_cats[i])), []).append(i)
        comps = [({k}, members) for k, members in groups.items()]
    else:
        by_group = {}
        for i in np.flatnonzero(~is_tract_cell):
            by_group.setdefault(int(layout.bg_group[cells[i]]), []).append(i)
        comps = [(set().union(*(cell_cats[i] for i in members)), members)
                 for members in by_group.values()]
    H = []
    for cats, members in comps:
        v = np.zeros(cells.size)
        v[members] = -1.0
        for i in np.flatnonzero(is_tract_cell):
            if cell_cats[i] <= cats:
                v[i] = 1.0
        H.append(v)
    H = np.array(H).T if H else np.zeros((cells.size, 0))
    q_h, _ = np.linalg.qr(H) if H.size else (H, None)
    # the rest of the null space, orthogonal to the hierarchy directions
    rest = null - q_h @ (q_h.T @ null) if q_h.size else null
    u, s2, _ = np.linalg.svd(rest, full_matrices=False) if rest.size else (rest, np.zeros(0), None)
    other = u[:, s2 > 1e-6] if rest.size else rest

    Y = layout.Y[cells]
    L = layout.L[cells]
    out = []
    for kind, V in (("hierarchy", q_h), ("other", other)):
        for j in range(V.shape[1]):
            v = V[:, j]
            involved = np.flatnonzero(np.abs(v) > 1e-6)
            tables = sorted({f"{layout.row_table[cell_rows[i][0]]}"
                             f"{'(T)' if is_tract_cell[i] else '(B)'}" for i in involved})
            contrast = float(v @ Y)
            spread = 2.0 * np.abs(v @ L)
            out.append(dict(tract=t, kind=kind, contrast=contrast,
                            rep_sd=float(np.sqrt(SDR_FACTOR * np.square(v @ L).sum())),
                            rep_max=float(spread.max()) if spread.size else 0.0,
                            tables=" ".join(tables)))
    return out, cells.size, rank, null.shape[1]


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lines, frames = [], []
    for puma in args.puma:
        inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / puma)
        x_t, x_b = sp.csr_matrix(inputs.X_T), sp.csr_matrix(inputs.X_B)
        for spec in args.collapse:
            collapse = None if spec == "none" else float(spec)
            layout = Layout(inputs, collapse)
            rows, coords, ranks, nulls = [], 0, 0, 0
            for t in range(inputs.Y_T.shape[0]):
                found, n_cells, rank, n_null = tract_analysis(layout, t, x_t, x_b, args.rtol)
                rows += found
                coords, ranks, nulls = coords + n_cells, ranks + rank, nulls + n_null
            frame = pd.DataFrame(rows).assign(puma=puma, collapse=spec)
            frames.append(frame)
            lines.append(f"== PUMA {puma}, collapse {spec}: {coords:,} coordinates over "
                         f"{inputs.Y_T.shape[0]} tracts, data map rank {ranks:,}, "
                         f"{nulls:,} invisible directions")
            for kind in ("hierarchy", "other"):
                part = frame[frame.kind == kind] if len(frame) else frame
                if not len(part):
                    lines.append(f"   {kind}: none")
                    continue
                c, r = part.contrast.abs(), part.rep_sd
                lines.append(
                    f"   {kind}: {len(part):,} directions; |published contrast| median "
                    f"{c.median():.3g}, max {c.max():.3g}; replicate sd median {r.median():.3g}, "
                    f"max {r.max():.3g}; exactly consistent (|contrast| and sd < 0.5): "
                    f"{int(((c < 0.5) & (r < 0.5)).sum()):,}")
                if kind == "other":
                    top = part.tables.value_counts().head(12)
                    lines += ["     tables involved (directions):"] + [
                        f"       {n:5d}  {tables}" for tables, n in top.items()]
            lines.append("")
    pd.concat(frames).to_csv(args.out / "null_directions.csv", index=False)
    text = "\n".join(lines) + "\n"
    (args.out / "null_directions.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
