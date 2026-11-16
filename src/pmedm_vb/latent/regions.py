"""Inputs for the regional run: regions of PUMAs in a PUMA's role, their PUMAs
in block groups' roles, and the PUMS as the truth.

For a region with PUMAs P_1..P_Z:

* **Units (support):** the region's own units (households and group-quarters
  persons), with their design weights; ``n`` their number and ``N`` their
  weighted total.
* **q:** the PMEDM's design distribution,
  :func:`~pmedm_vb.assemble.design.design_weights` -- each unit's weight
  spread uniformly over the region's zones, normalised over all pairs -- as
  :func:`~pmedm_vb.assemble.build.build_puma` builds it.
* **Constraints:** each PUMA's cells of the leave-one-PUMA-out check's S1 set
  (:func:`~pmedm_vb.embed.puma_check.constraint_sets`), computed from that
  PUMA's own records with the unit weights, as zone-level rows
  (``A_B = I``). There are no tract rows (``X_T`` has no columns), and so
  no analogue of the paper baseline's tract rows or of its ``puma``
  hierarchy. Each zone is its own tract (``A_T = I``) only so that the
  existing dual solver, which assembles its Hessian tract by tract, covers
  every zone's rows.
* **Σ:** the full replicate covariance over every observed cell of the
  region: the deviations ``L`` of the 80 replicate estimates from the
  estimate, so ``(4/80) L L′`` is the successive-difference covariance across
  cells *and* zones. Its diagonal ``v`` is floored at the square of the zone's
  mean unit weight, the check's SE floor (one sampled case). The PMEDM's
  shrinkage ``alpha`` (``sigma_alpha``) and multiplier prior apply on top
  (:func:`~pmedm_vb.latent.model.area_sigma`).
* **Zone weights** for the sum-to-zero constraint: each zone's observed unit
  total (its ``unit_type`` cells), computed from the zone's own records
  whichever table set is fitted -- a zone's population is known in practice.

**Transfer table sets** (:func:`transfer_sets`). Two disjoint sets of
tables, one fitted and the other predicted:

* **A** -- the check's S1 set: the one-way margins of S0 plus the PMEDM's
  remaining constraint tables.
* **B** -- the added tables of the check's S2 set (S2 less S1) that bring in
  at least one attribute A does not contain
  (:data:`~pmedm_vb.embed.attribute_targets.TABLE_ATTRIBUTES`). Added tables
  that only cross A's attributes with each other (household type by size,
  tenure by size, tenure by householder age, age of householder by income,
  employment status) are left out: predicting them from A tests A's
  crosses, not transfer to new attributes. B still shares some attributes
  with A through crosses (tenure by year built, say), which is what a new
  table set looks like in practice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.design import design_weights
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.data.variance import SDR_FACTOR
from pmedm_vb.embed.attribute_targets import ATTRIBUTE_TARGETS, TABLE_ATTRIBUTES
from pmedm_vb.embed.puma_check import CheckData


@dataclass
class Region:
    id: str
    name: str
    pumas: list[str]


def load_regions(path: str | Path) -> list[Region]:
    payload = json.loads(Path(path).read_text())
    return [Region(r["id"], r["name"], [str(g) for g in r["pumas"]]) for r in payload["regions"]]


@dataclass
class RegionProblem:
    """A region's PMEDM inputs and its bookkeeping against the statewide arrays."""

    region: Region
    inputs: PMEDMInputs
    units: np.ndarray          # statewide row index of each region unit
    zone_of_unit: np.ndarray   # zone index (into region.pumas) of each region unit
    zone_weight: np.ndarray    # observed unit total of each zone
    cells: list[str]           # constraint cell names, X_B's columns


def _attributes(table: str) -> set[str]:
    if table == "unit_type":
        return set()
    if table in ATTRIBUTE_TARGETS:
        return {table}
    return set(TABLE_ATTRIBUTES[table])


def transfer_sets(data: CheckData) -> dict[str, tuple[str, ...]]:
    """``{"A": S1 tables, "B": added tables with an attribute outside A}``."""
    a = tuple(data.sets.constraint["S1"])
    a_attributes = set().union(*(_attributes(t) for t in a))
    added = [t for t in data.sets.constraint["S2"] if t not in a]
    b = tuple(t for t in added if _attributes(t) - a_attributes)
    if set(a) & set(b):
        raise ValueError("table sets A and B overlap")
    return {"A": a, "B": b}


def build_region(data: CheckData, region: Region, set_name: str = "S1",
                 tables: tuple[str, ...] | None = None) -> RegionProblem:
    """A region's problem for a constraint set of the check (``set_name``) or
    an explicit list of tables (``tables``)."""
    tables = tuple(tables) if tables is not None else data.sets.constraint[set_name]
    pumas = list(region.pumas)
    in_region = np.isin(data.puma, pumas)
    units = np.flatnonzero(in_region)
    if units.size == 0:
        raise ValueError(f"region {region.id} has no units")
    zone_of_unit = pd.Index(pumas).get_indexer(data.puma[units])
    weight = data.weight[units]
    reps = data.replicates[units]

    X = sp.hstack([data.counts[t] for t in tables]).tocsr()[units]
    cells = [f"{t}:{c.name}" for t in tables for c in data.tables[t].categories]
    if len(cells) != X.shape[1]:
        raise ValueError("cell names do not match the counts' columns")
    Z, C = len(pumas), X.shape[1]

    Y_B = np.zeros((Z, C))
    rep_est = np.zeros((Z, C, reps.shape[1]))
    floors = np.zeros(Z)
    for b in range(Z):
        own = zone_of_unit == b
        Xb = X[own]
        Y_B[b] = Xb.T @ weight[own]
        rep_est[b] = np.asarray(Xb.T @ reps[own])
        floors[b] = weight[own].mean()
    # Stacked order is vec(Y_B) column-major: cell-major, zone fastest.
    deviations = (rep_est - Y_B[:, :, None]).transpose(1, 0, 2).reshape(Z * C, -1)
    replicate_var = SDR_FACTOR * np.square(deviations).sum(axis=1)
    floor = np.tile(np.square(floors), C)
    sigma_v = np.maximum(replicate_var, floor)

    # Every unit has exactly one unit type, so this is the zone's unit total
    # (the sum of its unit_type cells), whichever tables are fitted.
    zone_weight = np.bincount(zone_of_unit, weights=weight, minlength=Z)

    frame = pd.DataFrame({
        "SERIALNO": data.units.index[units].astype(str),
        "weight": weight,
        "is_group_quarters": data.units["is_group_quarters"].to_numpy()[units],
    })
    inputs = PMEDMInputs(
        q=design_weights(pd.Series(weight), n_zones=Z),
        X_T=sp.csr_matrix((units.size, 0)),
        X_B=X,
        A_T=sp.identity(Z, format="csr"),
        A_B=sp.identity(Z, format="csr"),
        Y_T=np.zeros((Z, 0)),
        Y_B=Y_B,
        sigma_v=sigma_v,
        sigma_l=deviations,
        puma=region.id,
        n=int(units.size),
        N=float(weight.sum()),
        units=frame,
        zones=pd.DataFrame({"block_group_geoid": pumas}),
        tracts=pd.DataFrame({"tract_geoid": pumas}),
        block_groups=pd.DataFrame({"block_group_geoid": pumas}),
        tract_constraints=[],
        bg_constraints=cells,
    )
    inputs.validate()
    return RegionProblem(region=region, inputs=inputs, units=units, zone_of_unit=zone_of_unit,
                         zone_weight=zone_weight, cells=cells)


def race_composition(data: CheckData, units: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Each unit's members' shares in ``B03002``'s race/ethnicity categories
    (from the records). Every unit has at least one member."""
    counts = data.counts["B03002"][units].toarray()
    total = counts.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("a unit has no members in B03002")
    names = [f"race_share.{c.name}" for c in data.tables["B03002"].categories]
    return counts / total, names
