"""The individual-side factors: attribute matrices, aggregation, design weights.

``X`` translates each unit into the constraint cells it contributes to, ``A``
translates zones into the areas constraints are published for, and ``q`` is the
prior on where a unit's weight can go.

All three stay small because a problem is one PUMA. Zones are that PUMA's block
groups, units are its households and group-quarters residents, and neither set
reaches beyond it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.data.geography import TRACT_GEOID_WIDTH


def build_attribute_matrix(counts: pd.DataFrame) -> tuple[sp.csr_matrix, list[str]]:
    """Sparse ``(n_units, n_constraints)`` from per-unit counts.

    Entries are **counts, not indicators**: a household contributes the number
    of its members matching a person-level cell. Sparse because most units
    match few cells -- a household is in one income band, one tenure, and has
    members in a handful of age-sex categories out of a hundred-odd columns.

    Column order is the frame's, which
    :func:`~pmedm_vb.assemble.households.person_counts` takes from the
    specifications, so it matches
    :attr:`~pmedm_vb.assemble.targets.TargetBlock.names` by construction. The
    labels are returned so a caller can assert that rather than trust it.
    """
    matrix = sp.csr_matrix(counts.to_numpy(dtype=float))
    return matrix, list(counts.columns)


def nesting(zones: Sequence[str], width: int = TRACT_GEOID_WIDTH) -> pd.Series:
    """Map each zone GEOID to its parent area GEOID, by prefix.

    Census GEOIDs nest by construction -- a block group's first 11 characters
    are its tract -- so this is string slicing rather than a spatial join.
    """
    index = pd.Index(zones, name="zone")
    return pd.Series([geoid[:width] for geoid in index], index=index, name="area")


def build_aggregation(
    zones: Sequence[str],
    areas: Sequence[str],
    membership: pd.Series | None = None,
) -> sp.csr_matrix:
    """``(n_areas, n_zones)`` operator summing zones into the areas holding them.

    ``membership`` maps each zone to its area; omitted, it is derived by
    :func:`nesting`. Pass ``areas == zones`` to get the identity, which is what
    ``A_B`` is when the zones *are* the block groups being constrained.

    Raises
    ------
    ValueError
        If a zone belongs to no listed area, or an area contains no zone.
        Either means the zone and area sets were built from different places,
        which would otherwise show up as a quietly wrong row of ``Y``.
    """
    zones = list(zones)
    areas = list(areas)
    if membership is None:
        if set(areas) == set(zones):
            # The identity case the docstring promises: nesting() would map a
            # block group to its tract, which is not among these areas.
            membership = pd.Series(zones, index=pd.Index(zones, name="zone"), name="area")
        else:
            membership = nesting(zones)

    area_position = {area: i for i, area in enumerate(areas)}
    rows, columns = [], []
    for column, zone in enumerate(zones):
        area = membership.get(zone)
        if area not in area_position:
            raise ValueError(
                f"zone {zone!r} maps to area {area!r}, which is not among the "
                f"{len(areas)} areas given"
            )
        rows.append(area_position[area])
        columns.append(column)

    empty = set(range(len(areas))) - set(rows)
    if empty:
        missing = [areas[i] for i in sorted(empty)][:5]
        raise ValueError(
            f"{len(empty)} area(s) contain no zone, e.g. {missing}. The zone "
            f"and area sets disagree"
        )

    return sp.csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(areas), len(zones))
    )


def design_weights(
    weights: pd.Series,
    n_zones: int,
    *,
    zone_shares: np.ndarray | None = None,
) -> np.ndarray:
    """``q``, the ``(n_zones, n_units)`` prior, summing to one over all pairs.

    Every unit may receive weight in every zone of its own PUMA, so ``q`` has
    no structural zeros here -- confining a unit to its PUMA is what selecting
    the PUMA already did. The prior is the unit's own sample weight, spread
    across zones and normalised so that ``sum_ij q_ij = 1``, as the derivation
    requires.

    ``zone_shares`` spreads a unit unevenly across zones -- a zone's share of
    the PUMA's population, say. Omitted, the spread is uniform, which is the
    classic PMEDM prior: it says the sample tells us what kinds of unit exist
    and how many, and nothing about where they are. That is the honest starting
    point when the constraints are what carry the spatial information.

    Notes
    -----
    Dense, and comfortably so at one PUMA: roughly 3,500 units by 95 zones is
    about 2.7 MB. The joint problem over four PUMAs would be an order of
    magnitude larger and mostly structural zeros, which is the arithmetic
    behind solving per PUMA.
    """
    unit = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(unit)) or np.any(unit < 0):
        raise ValueError("unit weights must be finite and non-negative")
    total = unit.sum()
    if total <= 0:
        raise ValueError("unit weights sum to zero; no unit could receive weight")

    if zone_shares is None:
        shares = np.full(n_zones, 1.0 / n_zones)
    else:
        shares = np.asarray(zone_shares, dtype=float)
        if shares.shape != (n_zones,):
            raise ValueError(f"zone_shares must be ({n_zones},), got {shares.shape}")
        if np.any(shares < 0) or shares.sum() <= 0:
            raise ValueError("zone_shares must be non-negative and sum above zero")
        shares = shares / shares.sum()

    return np.outer(shares, unit / total)
