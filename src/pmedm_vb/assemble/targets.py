"""Constraint targets ``Y`` and the factors of their covariance.

A collapsed constraint is a sum of published cells, and summing cells is where
the replicate files earn their keep. ``Y`` for a category is the sum of its
cells' estimates -- but its **variance is not the sum of their variances**,
because the cells are correlated. Applying the collapse to the *deviations* and
recomputing from there gives the exact variance of the sum, correlations
included, which is what makes merging cells free rather than approximate.

Two consequences that are easy to get backwards:

* The zero-cell policy applies **after** collapsing. A category is a modelled
  zero only when every cell in it is, and a category built from several cells
  is far less often degenerate than its parts.
* Constraints from different tables need not align with one another. Only a
  constraint's own definition has to match its own published cell; stacking
  cells from different tables into one ``L`` is what carries the correlation
  between them.

  That rests on every cell deriving from the same 80 replicate weight sets,
  and it is **verified**: ``B01003_001`` and ``B01001_001`` are both total
  population, and across all 80 replicates they agree to a maximum absolute
  difference of *exactly* zero -- not approximately, which is what shared
  weights and only shared weights produces.

Row order throughout is the stacked constraint vector, **column-major**: for a
block of ``(n_areas, n_categories)``, entry ``(i, j)`` sits at
``i + n_areas * j``. ``v`` and ``L`` follow that order, and a permutation of
one against the other is undetectable downstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from pmedm_vb.assemble.constraints import ConstraintTable
from pmedm_vb.config import StudyArea
from pmedm_vb.data.variance import (
    N_REPLICATES,
    SDR_FACTOR,
    deviations,
    fetch_replicates,
    zero_cell_variance,
)


@dataclass(frozen=True)
class TargetBlock:
    """Targets and covariance factors for one geography's constraints.

    Attributes
    ----------
    geoids:
        Row order of ``Y``.
    names:
        Column order of ``Y``, as ``"{table}.{category}"``. These are what
        ``X``'s columns must match, by name rather than by position.
    Y:
        ``(n_areas, n_categories)`` published estimates, collapsed.
    v:
        ``(n_areas * n_categories,)`` per-cell variances with the zero-cell
        policy applied, in column-major order against ``Y``.
    L:
        ``(n_areas * n_categories, 80)`` collapsed replicate deviations, rows
        in the same order as ``v``.
    """

    geoids: pd.Index
    names: list[str]
    Y: np.ndarray
    v: np.ndarray
    L: np.ndarray

    @property
    def n_areas(self) -> int:
        return self.Y.shape[0]

    @property
    def n_categories(self) -> int:
        return self.Y.shape[1]

    def vec(self) -> np.ndarray:
        """``vec(Y)`` column-major -- the order ``v`` and ``L`` are in."""
        return self.Y.ravel(order="F")

    def validate(self) -> None:
        """Raise if the pieces do not line up, including the ordering."""
        expected = self.n_areas * self.n_categories
        problems = []
        if self.v.shape != (expected,):
            problems.append(f"v is {self.v.shape}, expected {(expected,)}")
        if self.L.shape != (expected, N_REPLICATES):
            problems.append(f"L is {self.L.shape}, expected {(expected, N_REPLICATES)}")
        if len(self.names) != self.n_categories:
            problems.append("names does not match the width of Y")
        if len(self.geoids) != self.n_areas:
            problems.append("geoids does not match the height of Y")
        if self.v.size and not np.all(self.v > 0):
            problems.append(f"{int((self.v <= 0).sum())} variance(s) are not positive")
        if problems:
            raise ValueError("inconsistent targets:\n  " + "\n  ".join(problems))


def build_targets(
    area: StudyArea,
    tables: Sequence[ConstraintTable],
    *,
    geography: str,
    geoids: Sequence[str] | None = None,
    policy: str = "model",
    floor: float | None = None,
) -> TargetBlock:
    """Collapse published cells into constraints and derive their covariance.

    Parameters
    ----------
    tables:
        Constraint specifications, all for ``geography``. Their categories are
        concatenated; names are qualified by table, so two tables may use the
        same category name.
    geoids:
        Exact zones to build for, in place of the study area's own. This is how
        a per-PUMA problem is assembled: a PUMA is built from whole tracts but
        may cross a county line, so its zones are not in general a subset of
        the area's counties. When given, the fetch is statewide and then
        restricted to precisely these geographies -- PUMAs never cross a state
        boundary, so statewide always suffices.
    policy:
        Zero-cell handling, as :func:`pmedm_vb.data.variance.variances`.
        Applied to the *collapsed* variances.

    Notes
    -----
    One fetch covers every table, so the cells land in a single frame and a
    single ``L``. That is not only convenience: cross-table covariance requires
    the deviations to sit in one matrix.
    """
    wrong = [t.table for t in tables if t.geography != geography]
    if wrong:
        raise ValueError(f"tables not specified for {geography!r}: {wrong}")

    # The area the *data* is drawn from. Widened to the whole state whenever an
    # explicit zone set is given, since those zones may lie outside the study
    # area's counties; the restriction below then decides membership, rather
    # than the county filter deciding it by accident.
    source = area if geoids is None else replace(area, counties=())
    frame = fetch_replicates(source, [t.table for t in tables], geography=geography)

    if geoids is not None:
        wanted = set(geoids)
        frame = frame[frame.index.get_level_values("geoid").isin(wanted)]
        missing = wanted - set(frame.index.get_level_values("geoid"))
        if missing:
            raise ValueError(
                f"{len(missing)} requested geograph(ies) are not published for "
                f"these tables at {geography}, e.g. {sorted(missing)[:5]}"
            )

    # cell -> column index, built from the specs. Waived cells are absent and
    # drop out of the aggregation, which is what waiving means.
    names: list[str] = []
    column_of_cell: dict[str, int] = {}
    for table in tables:
        for category in table.categories:
            for cell in category.published:
                column_of_cell[cell] = len(names)
            names.append(f"{table.table}.{category.name}")

    cells = frame.index.get_level_values("cell")
    column = cells.map(column_of_cell)
    used = column.notna()
    if not used.any():
        raise ValueError(
            "no published cell matched any category; the specs and the fetched "
            "tables disagree"
        )

    block = pd.DataFrame(
        deviations(frame)[used],
        index=pd.MultiIndex.from_arrays(
            [frame.index.get_level_values("geoid")[used], column[used].astype(int)],
            names=["geoid", "column"],
        ),
    )
    block["estimate"] = frame["estimate"].to_numpy(float)[used]

    # Summing the deviations *is* the collapse: the variance of the sum follows
    # from the summed deviations, and carries the covariance between the cells
    # merged. Summing their variances instead would assume them independent.
    agg = block.groupby(level=["geoid", "column"]).sum()
    # Sorting column-major gives exactly vec(Y, order="F") row order.
    agg = agg.sort_index(level=["column", "geoid"])

    geoids = pd.Index(sorted(agg.index.get_level_values("geoid").unique()), name="geoid")
    shape = (len(names), len(geoids))
    if len(agg) != shape[0] * shape[1]:
        raise ValueError(
            f"expected {shape[0] * shape[1]} (geography, constraint) pairs, got "
            f"{len(agg)} -- a table is published for a different set of "
            f"geographies than the others"
        )

    L = agg.drop(columns="estimate").to_numpy(float)
    Y = agg["estimate"].to_numpy(float).reshape(shape).T

    v = _apply_zero_policy(
        SDR_FACTOR * np.square(L).sum(axis=1),
        geoids=agg.index.get_level_values("geoid"),
        area=source,
        geography=geography,
        policy=policy,
        floor=floor,
    )

    targets = TargetBlock(geoids=geoids, names=names, Y=Y, v=v, L=L)
    targets.validate()
    return targets


def _apply_zero_policy(
    variance: np.ndarray,
    *,
    geoids: pd.Index,
    area: StudyArea,
    geography: str,
    policy: str,
    floor: float | None,
) -> np.ndarray:
    """Substitute for collapsed cells whose replicate variance is exactly zero.

    Only ``"model"`` and ``"floor"`` are offered here. ``"drop"`` would remove
    rows from ``v`` while ``Y`` and ``L`` kept theirs, so the block would no
    longer describe one rectangular set of constraints; dropping a constraint
    is a decision for the specification, not for the covariance.
    """
    degenerate = variance == 0.0
    if not degenerate.any():
        return variance

    result = variance.copy()
    if policy == "floor":
        if floor is None or floor <= 0:
            raise ValueError("policy='floor' requires a positive floor")
        result[degenerate] = floor
        return result
    if policy != "model":
        raise ValueError(
            f"policy must be 'model' or 'floor' for collapsed targets, got "
            f"{policy!r}; 'drop' would leave v shorter than Y and L"
        )

    modelled = zero_cell_variance(area, geography=geography)
    result[degenerate] = modelled.reindex(geoids[degenerate]).to_numpy()
    if not np.all(result > 0):
        raise ValueError(
            "the zero-cell model left non-positive variances; a geography in "
            "the targets is missing from the population estimates used for k"
        )
    return result
