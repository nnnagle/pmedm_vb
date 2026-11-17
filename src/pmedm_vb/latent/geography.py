"""Distances between PUMAs (step 1, piece A6 of ``PLAN_latent.md``).

The model (``MODEL_latent.qmd``) weights atom ``i`` in PUMA ``P``'s reference
by ``γ^{d(P,i)}``, with ``d(P,i) = d(P, PUMA_i)`` the number of adjacency
steps between the two PUMAs:

* **Adjacency** (:func:`adjacency`): two PUMAs are adjacent when their
  boundary rings share at least two vertices (TIGER polygons that share an
  edge share its vertices exactly). This is the rule the regional run's
  region list was checked with (``experiments/latent/make_regions.py``, which
  imports it from here).
* **Distance** (:func:`puma_distances`): breadth-first search on that graph;
  ``d(P,P) = 0``, a neighbour is 1, a neighbour's neighbour 2, and so on.

The 2020 PUMA boundaries are read from the cached TIGER zip
(:func:`~pmedm_vb.embed.puma_map.cached_boundaries`; fetched by
``experiments/embed/prefetch.py``), so this runs offline.
"""

from __future__ import annotations

from collections import deque
from itertools import combinations

import numpy as np
import pandas as pd

from pmedm_vb.config import StudyArea
from pmedm_vb.embed.puma_map import cached_boundaries, read_boundaries


def adjacency(rings: dict[str, list[np.ndarray]], shared: int = 2) -> dict[str, set[str]]:
    """PUMAs whose boundary rings share at least ``shared`` vertices."""
    vertices = {
        g: {tuple(v) for ring in r for v in np.round(ring, 7)} for g, r in rings.items()}
    out = {g: set() for g in rings}
    for a, b in combinations(rings, 2):
        if len(vertices[a] & vertices[b]) >= shared:
            out[a].add(b)
            out[b].add(a)
    return out


def puma_rings(area: StudyArea) -> dict[str, list[np.ndarray]]:
    """The state's 2020 PUMA polygons from the cached boundary zip, by GEOID."""
    return read_boundaries(cached_boundaries(area))


def puma_distances(adjacent: dict[str, set[str]]) -> pd.DataFrame:
    """``d(P,Q)``: adjacency steps between every pair of PUMAs (breadth-first
    search), rows and columns in sorted GEOID order. A pair with no path is
    ``-1`` (:func:`distance_checks` reports it)."""
    pumas = sorted(adjacent)
    position = {g: k for k, g in enumerate(pumas)}
    D = np.full((len(pumas), len(pumas)), -1, dtype=np.int64)
    for source in pumas:
        row = D[position[source]]
        row[position[source]] = 0
        queue = deque([source])
        while queue:
            g = queue.popleft()
            for h in adjacent[g]:
                if row[position[h]] < 0:
                    row[position[h]] = row[position[g]] + 1
                    queue.append(h)
    return pd.DataFrame(D, index=pd.Index(pumas, name="puma"), columns=pumas)


def distance_checks(D: pd.DataFrame) -> dict:
    """Symmetric, zero diagonal, connected (no pair without a path)."""
    values = D.to_numpy()
    return {
        "n_pumas": len(D),
        "symmetric": bool((values == values.T).all()),
        "zero_diagonal": bool((np.diag(values) == 0).all()),
        "connected": bool((values >= 0).all()),
        "off_diagonal_positive": bool((values[~np.eye(len(D), dtype=bool)] > 0).all()),
        "max_distance": int(values.max()),
    }


def load_distances(area: StudyArea) -> pd.DataFrame:
    """``d(P,Q)`` for the state's PUMAs, from the cached boundaries."""
    return puma_distances(adjacency(puma_rings(area)))


def atom_distance(D: pd.DataFrame, puma: str, atom_puma: np.ndarray) -> np.ndarray:
    """``d(P,i)`` for every atom: the distance from ``puma`` to the atom's PUMA."""
    row = D.loc[str(puma)]
    out = row.reindex(pd.Index(atom_puma.astype(str))).to_numpy()
    if np.isnan(out.astype(float)).any():
        missing = sorted(set(atom_puma.astype(str)) - set(D.columns))
        raise ValueError(f"atoms in PUMAs without a boundary: {missing[:5]}")
    return out.astype(np.int64)
