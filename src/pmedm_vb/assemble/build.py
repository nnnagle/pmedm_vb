"""From a study area to one solver-ready problem per PUMA.

The step that puts the pieces together, and the only place the stacked
constraint ordering is fixed: ``[vec(Y_T); vec(Y_B)]``, column-major within
each block, with ``sigma_v`` and ``sigma_l`` following the same order. Getting
that wrong is undetectable downstream -- it produces a solver that converges on
wrong uncertainties -- so it is done once, here, and asserted.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from pmedm_vb.assemble.constraints import ConstraintTable
from pmedm_vb.assemble.design import (
    build_aggregation,
    build_attribute_matrix,
    design_weights,
)
from pmedm_vb.assemble.households import build_units, person_counts
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.targets import build_targets
from pmedm_vb.config import StudyArea
from pmedm_vb.data.geography import puma_crosswalk_whole

#: Geography whose zones the weights are defined over. Block group is the
#: finest level the constraints reach, so it is what ``W`` is indexed by; tract
#: constraints reach it through ``A_T``.
ZONE_GEOGRAPHY = "block group"


def build_puma(
    area: StudyArea,
    puma: str,
    tables: Sequence[ConstraintTable],
    *,
    zones: pd.DataFrame | None = None,
    policy: str = "model",
) -> PMEDMInputs:
    """Assemble one PUMA's problem.

    ``zones`` may be passed to avoid re-reading the crosswalk when building
    several PUMAs; it is the full frame, filtered here.
    """
    if zones is None:
        zones = puma_crosswalk_whole(area)
    own = zones[zones["puma_geoid"] == puma]
    if own.empty:
        raise ValueError(f"PUMA {puma!r} has no zones in {area.slug}")

    block_groups = sorted(own["block_group_geoid"].unique())
    tracts = sorted(own["tract_geoid"].unique())

    by_geography: dict[str, list[ConstraintTable]] = {}
    for table in tables:
        by_geography.setdefault(table.geography, []).append(table)
    missing = {"tract", ZONE_GEOGRAPHY} - set(by_geography)
    if missing:
        raise ValueError(f"no constraint tables specified for {sorted(missing)}")

    units = build_units(area, puma, tables)

    blocks = {
        "tract": build_targets(
            area, by_geography["tract"], geography="tract",
            geoids=tracts, policy=policy,
        ),
        ZONE_GEOGRAPHY: build_targets(
            area, by_geography[ZONE_GEOGRAPHY], geography=ZONE_GEOGRAPHY,
            geoids=block_groups, policy=policy,
        ),
    }

    factors = {}
    for geography, block in blocks.items():
        counts = person_counts(units, by_geography[geography])
        matrix, labels = build_attribute_matrix(counts)
        if labels != list(block.names):
            raise ValueError(
                f"{geography}: X's columns and Y's do not agree. "
                f"First difference at "
                f"{next(i for i, (a, b) in enumerate(zip(labels, block.names)) if a != b)}"
            )
        factors[geography] = matrix

    # A_B is the identity: the zones *are* the block groups being constrained.
    A_B = build_aggregation(block_groups, block_groups)
    A_T = build_aggregation(block_groups, tracts)

    q = design_weights(units.units["weight"], n_zones=len(block_groups))

    # Column-major within each block, tract block first. sigma_v and sigma_l
    # inherit that order from the blocks themselves, which built it.
    sigma_v = np.concatenate([blocks["tract"].v, blocks[ZONE_GEOGRAPHY].v])
    sigma_l = np.vstack([blocks["tract"].L, blocks[ZONE_GEOGRAPHY].L])

    problem = PMEDMInputs(
        q=q,
        X_T=factors["tract"],
        X_B=factors[ZONE_GEOGRAPHY],
        A_T=A_T,
        A_B=A_B,
        Y_T=blocks["tract"].Y,
        Y_B=blocks[ZONE_GEOGRAPHY].Y,
        sigma_v=sigma_v,
        sigma_l=sigma_l,
        puma=str(puma),
        n=units.n_units,
        N=float(units.units["weight"].sum()),
        units=units.units.reset_index()[["SERIALNO", "weight", "is_group_quarters"]],
        zones=pd.DataFrame({"block_group_geoid": block_groups}),
        tracts=pd.DataFrame({"tract_geoid": tracts}),
        block_groups=pd.DataFrame({"block_group_geoid": block_groups}),
        tract_constraints=list(blocks["tract"].names),
        bg_constraints=list(blocks[ZONE_GEOGRAPHY].names),
    )
    problem.validate()
    return problem


def build_all(
    area: StudyArea,
    tables: Sequence[ConstraintTable],
    *,
    policy: str = "model",
) -> dict[str, PMEDMInputs]:
    """One problem per PUMA, keyed by PUMA GEOID.

    The crosswalk is read once and shared, since it is a national file.
    """
    zones = puma_crosswalk_whole(area)
    return {
        str(puma): build_puma(area, str(puma), tables, zones=zones, policy=policy)
        for puma in sorted(zones["puma_geoid"].unique())
    }
