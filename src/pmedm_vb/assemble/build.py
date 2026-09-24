"""From a study area to one solver-ready problem per PUMA.

The step that puts the pieces together, and the only place the stacked
constraint ordering is fixed: ``[vec(Y_T); vec(Y_B)]``, column-major within
each block, with ``sigma_v`` and ``sigma_l`` following the same order. Getting
that wrong is undetectable downstream -- it produces a solver that converges on
wrong uncertainties -- so it is done once, here, and asserted.

**Statewide support.** ``build_puma(..., epsilon=...)`` lets a PUMA's weight
land on any record in the state, not only its own: the PUMA's records keep
``(1 - epsilon)`` of their design weight and every state record adds a share
of ``epsilon N`` in proportion to its own
(:func:`~pmedm_vb.assemble.design.support_weights`), so the weights still sum
to the PUMA's ``N``. Records with identical rows of ``X`` are then merged
(:func:`~pmedm_vb.assemble.design.collapse_units`), households and
group-quarters persons kept apart; the merge leaves ``f(lambda)`` unchanged.
``n`` and ``N`` count the PUMA's own records, as before. ``epsilon=None``, the
default, is the PUMA-only problem, built exactly as it always was.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.constraints import ConstraintTable
from pmedm_vb.assemble.design import (
    build_aggregation,
    build_attribute_matrix,
    collapse_units,
    design_weights,
    support_weights,
)
from pmedm_vb.assemble.households import build_units, person_counts
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.targets import build_targets
from pmedm_vb.config import StudyArea
from pmedm_vb.data.geography import puma_crosswalk_whole
from pmedm_vb.progress import logger, stage

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
    epsilon: float | None = None,
) -> PMEDMInputs:
    """Assemble one PUMA's problem.

    ``zones`` may be passed to avoid re-reading the crosswalk when building
    several PUMAs; it is the full frame, filtered here. ``epsilon`` in
    ``[0, 1)`` widens the support to the whole state (see the module
    docstring); ``None`` keeps the PUMA's own records only.
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

    logger.info(
        "PUMA %s: %d tracts, %d block groups", puma, len(tracts), len(block_groups)
    )
    statewide = epsilon is not None
    with stage(f"PUMA {puma}: units from PUMS{' (statewide)' if statewide else ''}") as step:
        units = build_units(area, None if statewide else puma, tables)
        step.detail = f"{units.n_units:,} units"

    blocks = {}
    for geography, geoids in (("tract", tracts), (ZONE_GEOGRAPHY, block_groups)):
        with stage(f"PUMA {puma}: {geography} targets") as step:
            blocks[geography] = build_targets(
                area, by_geography[geography], geography=geography,
                geoids=geoids, policy=policy,
            )
            step.detail = f"{blocks[geography].Y.size:,} constraint rows"

    factors = {}
    for geography, block in blocks.items():
        with stage(f"PUMA {puma}: {geography} attribute matrix"):
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

    weights = units.units["weight"]
    frame = units.units.reset_index()[["SERIALNO", "weight", "is_group_quarters"]]
    n, N = units.n_units, float(units.units["weight"].sum())
    members = support = support_columns = None
    if statewide:
        in_puma = (units.units["puma_geoid"] == puma).to_numpy()
        own_weights = units.units["weight"].to_numpy(dtype=float)
        is_gq = units.units["is_group_quarters"].to_numpy()
        record_weights = support_weights(units.units["weight"], in_puma, epsilon)
        with stage(f"PUMA {puma}: collapse identical rows") as step:
            X_T, X_B, merged, row = collapse_units(
                factors["tract"], factors[ZONE_GEOGRAPHY], is_gq, record_weights
            )
            step.detail = f"{units.n_units:,} records -> {merged.size:,} rows"
        weights = pd.Series(merged)
        factors = {"tract": X_T, ZONE_GEOGRAPHY: X_B}
        n, N = int(in_puma.sum()), float(own_weights[in_puma].sum())
        members = pd.DataFrame({
            "SERIALNO": units.units.index.to_numpy(),
            "row": row,
            "weight": record_weights,
            "in_puma": in_puma,
        })
        frame = unit_frame(members, is_gq, merged.size)
        support, support_columns = support_report(
            members, own_weights, X_T, X_B,
            list(blocks["tract"].names), list(blocks[ZONE_GEOGRAPHY].names), epsilon,
        )
        logger.info(
            "PUMA %s support (epsilon=%g): %s own records in %s rows; %s state records "
            "in %s rows, %s of them new (%.2f%% of the prior); %s own rows widened",
            puma, epsilon, f"{support['own_records']:,}", f"{support['own_rows']:,}",
            f"{support['state_records']:,}", f"{support['rows']:,}",
            f"{support['rows_added']:,}", 100 * support["prior_share_added"],
            f"{support['own_rows_widened']:,}",
        )

    q = design_weights(weights, n_zones=len(block_groups))

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
        n=n,
        N=N,
        units=frame,
        zones=pd.DataFrame({"block_group_geoid": block_groups}),
        tracts=pd.DataFrame({"tract_geoid": tracts}),
        block_groups=pd.DataFrame({"block_group_geoid": block_groups}),
        tract_constraints=list(blocks["tract"].names),
        bg_constraints=list(blocks[ZONE_GEOGRAPHY].names),
        unit_members=members,
        support=support,
        support_columns=support_columns,
    )
    with stage(f"PUMA {puma}: validate"):
        problem.validate()
    return problem


def unit_frame(members: pd.DataFrame, is_gq: np.ndarray, n_rows: int) -> pd.DataFrame:
    """The ``units`` frame of a merged problem, one row per merged row.

    ``SERIALNO`` is a representative record -- the PUMA's own record with the
    largest design weight where the row holds one, else the largest from
    elsewhere -- kept so that reports naming a unit still name a real record;
    ``unit_members`` lists them all.
    """
    kept = members.assign(is_group_quarters=is_gq)
    kept = kept[kept["row"] >= 0]
    grouped = kept.groupby("row")
    representative = (
        kept.sort_values(["in_puma", "weight"], ascending=False, kind="stable")
        .drop_duplicates("row")
        .set_index("row")
    )
    frame = pd.DataFrame({
        "SERIALNO": representative["SERIALNO"],
        "weight": grouped["weight"].sum(),
        "is_group_quarters": representative["is_group_quarters"],
        "n_records": grouped.size(),
        "n_own": grouped["in_puma"].sum().astype(int),
    }).reindex(range(n_rows))
    return frame.reset_index(drop=True)


def support_report(
    members: pd.DataFrame,
    own_weights: np.ndarray,
    X_T,
    X_B,
    tract_names: list[str],
    bg_names: list[str],
    epsilon: float,
) -> tuple[dict, pd.DataFrame]:
    """How much the statewide support adds, overall and per constraint.

    Overall: records and merged rows, own and statewide; how many rows hold no
    PUMA record (``rows_added``) and the share of the prior on them; how many
    rows holding a PUMA record also took in records from elsewhere
    (``own_rows_widened``).

    Per constraint column, for tract and block group alike: the rows and
    records carrying the category (``X > 0``), PUMA-only against statewide,
    and the share of the carriers' prior weight on the single largest
    carrying row -- near one when a category rests on one record type, the
    pattern behind the VB weight blow-ups. The PUMA-only share uses the
    original design weights ``d_puma``.
    """
    row = members["row"].to_numpy()
    in_puma = members["in_puma"].to_numpy()
    kept = row >= 0
    own = in_puma & kept
    rows = int(row.max()) + 1 if kept.any() else 0
    count = np.bincount(row[kept], minlength=rows)
    own_count = np.bincount(row[own], minlength=rows)
    weight = np.bincount(row[kept], weights=members["weight"].to_numpy()[kept], minlength=rows)
    own_weight = np.bincount(row[own], weights=own_weights[own], minlength=rows)
    own_rows = own_count > 0

    support = {
        "epsilon": float(epsilon),
        "own_records": int(in_puma.sum()),
        "state_records": int(row.size),
        "records_dropped": int((~kept).sum()),
        "own_rows": int(own_rows.sum()),
        "rows": rows,
        "rows_added": int((~own_rows).sum()),
        "own_rows_widened": int((own_rows & (count > own_count)).sum()),
        "prior_share_added": float(weight[~own_rows].sum() / weight.sum()),
    }

    columns = []
    for level, matrix, names in (("tract", X_T, tract_names), ("block group", X_B, bg_names)):
        by_column = sp.csc_matrix(matrix)
        for k, name in enumerate(names):
            carrying = by_column.indices[by_column.indptr[k] : by_column.indptr[k + 1]]
            carrying = carrying[by_column.data[by_column.indptr[k] : by_column.indptr[k + 1]] > 0]
            own_carrying = carrying[own_rows[carrying]]
            columns.append({
                "level": level,
                "constraint": name,
                "own_rows": int(own_carrying.size),
                "rows": int(carrying.size),
                "own_records": int(own_count[own_carrying].sum()),
                "records": int(count[carrying].sum()),
                "own_top_row_share": _top_share(own_weight[own_carrying]),
                "top_row_share": _top_share(weight[carrying]),
            })
    return support, pd.DataFrame(columns)


def _top_share(weights: np.ndarray) -> float:
    total = weights.sum()
    return float(weights.max() / total) if weights.size and total > 0 else float("nan")


def build_all(
    area: StudyArea,
    tables: Sequence[ConstraintTable],
    *,
    policy: str = "model",
    epsilon: float | None = None,
) -> dict[str, PMEDMInputs]:
    """One problem per PUMA, keyed by PUMA GEOID.

    The crosswalk is read once and shared, since it is a national file.
    ``epsilon`` is passed to :func:`build_puma`.
    """
    zones = puma_crosswalk_whole(area)
    pumas = sorted(zones["puma_geoid"].unique())
    problems = {}
    for i, puma in enumerate(pumas, start=1):
        with stage(f"PUMA {puma} ({i} of {len(pumas)})"):
            problems[str(puma)] = build_puma(
                area, str(puma), tables, zones=zones, policy=policy, epsilon=epsilon
            )
    return problems
