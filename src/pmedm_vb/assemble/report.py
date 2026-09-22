"""What a run is constraining, and how much covariance structure it has.

Printed when a problem is assembled, because both facts are easy to get wrong
silently. A constraint set is a long list of generated column names, and the
quickest way to notice that a table collapsed to the wrong granularity, or that
a category is missing, is to read the titles and counts next to each other. The
covariance rank says how much independent information the 80 replicates
actually carry for this constraint set, which bounds what a non-diagonal
``Sigma`` can contribute at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from pmedm_vb.assemble.constraints import ConstraintTable
from pmedm_vb.assemble.targets import TargetBlock, build_targets
from pmedm_vb.config import StudyArea
from pmedm_vb.data.geography import puma_crosswalk_whole
from pmedm_vb.data.variance import N_REPLICATES, table_list

#: Order geographies are reported in: coarse to fine, as the constraints nest.
GEOGRAPHY_ORDER = ("tract", "block group")


@dataclass(frozen=True)
class RankReport:
    """Rank of the design covariance behind one constraint block."""

    n_rows: int
    rank: int
    singular_values: np.ndarray
    tolerance: float

    @property
    def ceiling(self) -> int:
        """The most the replicates could support: ``min(n_rows, 80)``."""
        return min(self.n_rows, N_REPLICATES)

    @property
    def deficiency(self) -> int:
        """How far short of that ceiling the actual rank falls."""
        return self.ceiling - self.rank

    @property
    def margin(self) -> float:
        """Smallest retained singular value, as a multiple of the tolerance.

        The evidence that the rank was measured rather than hit a ceiling. A
        margin of many orders of magnitude means every retained direction is
        unambiguously non-degenerate; a margin near 1 means the cutoff landed
        in the middle of the spectrum and the reported rank is a judgement
        about the tolerance rather than a property of the matrix.
        """
        if not self.rank:
            return float("nan")
        return float(self.singular_values[self.rank - 1] / self.tolerance)

    @property
    def coverage(self) -> float:
        """Fraction of the constraint space the replicates say anything about.

        Rank over rows. The interesting number for a block-group run: 80
        directions out of many thousands, with every remaining direction
        covered by ``D`` alone.
        """
        return self.rank / self.n_rows


def covariance_rank(L: np.ndarray, *, tolerance: float | None = None) -> RankReport:
    """Rank of ``(4/80) L L'``, taken from ``L``'s singular values.

    Computed from ``L`` rather than from the product: forming ``L L'`` squares
    the condition number and loses exactly the small singular values that decide
    where the rank cuts off, quite apart from being ``(n, n)``.

    The default tolerance is the usual ``max(shape) * eps * largest``, the same
    rule :func:`numpy.linalg.matrix_rank` applies.
    """
    singular = np.linalg.svd(L, compute_uv=False)
    if tolerance is None:
        tolerance = max(L.shape) * np.finfo(float).eps * singular[0]
    return RankReport(
        n_rows=L.shape[0],
        rank=int((singular > tolerance).sum()),
        singular_values=singular,
        tolerance=float(tolerance),
    )


def _wrap(labels: list[str], width: int = 88, indent: int = 10) -> list[str]:
    """Pack labels onto lines, two spaces apart, wrapped to ``width``."""
    lines, current = [], ""
    for label in labels:
        candidate = f"{current}  {label}" if current else label
        if len(candidate) + indent > width and current:
            lines.append(current)
            current = label
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def format_constraints(area: StudyArea, blocks: Mapping[str, TargetBlock]) -> str:
    """One section per geography: each table, its size, and its column labels.

    The definitions are identical across PUMAs -- same specifications, same
    collapse -- so they are printed once rather than repeated per problem.
    Per-PUMA sizes and ranks go in :func:`format_pumas`.

    Column names arrive as ``"{table}.{category}"``, so the table each belongs
    to is recovered from the name rather than passed in separately: the report
    then describes what was actually built, not what was intended.
    """
    titles = table_list(area).set_index("TBLID")["TITLE"]
    out = []

    ordered = [g for g in GEOGRAPHY_ORDER if g in blocks]
    ordered += [g for g in blocks if g not in GEOGRAPHY_ORDER]

    for geography in ordered:
        block = blocks[geography]
        grouped: dict[str, list[str]] = {}
        for name in block.names:
            table, _, category = name.partition(".")
            grouped.setdefault(table, []).append(category)

        out.append(f"{geography.title()}")
        for table, categories in grouped.items():
            out.append(f"  {table}  {titles.get(table, '(title not published)')}")
            out.append(f"  {'':8}{len(categories)} constraints")
            out.extend(f"  {'':8}{line}" for line in _wrap(categories))
        out.append(f"  {'':8}[{len(grouped)} tables, {block.n_categories} constraints]")
        out.append("")
    return "\n".join(out)


def format_pumas(by_puma: Mapping[str, Mapping[str, TargetBlock]]) -> str:
    """Per-PUMA sizes and covariance rank, one row per PUMA and geography.

    The rank reported is that of ``L``, not of ``Sigma``.
    ``Sigma(alpha) = D + (1-alpha)(4/80) L L'`` is full rank for any
    ``alpha > 0`` because ``D`` is strictly positive; the rank-80 ceiling
    belongs to the replicate part alone.

    ``coverage`` is rank over rows -- the share of constraint directions the
    replicates say anything about, with the rest carried by ``D``. It is the
    more useful reading of the same number, and it is why solving per PUMA
    helps: the rank ceiling is 80 either way, but it describes a far smaller
    space.

    ``margin`` is the smallest retained singular value as a multiple of the
    rank tolerance. It is the evidence that the rank was measured rather than
    assumed -- a rank equal to its own ceiling reads identically either way
    from the number alone.
    """
    out = [
        f"  {'PUMA':<10}{'geography':<14}{'areas':>7}{'rows':>9}"
        f"{'rank':>6}{'ceiling':>9}{'coverage':>10}{'margin':>10}"
    ]
    for puma in sorted(by_puma):
        for geography, block in by_puma[puma].items():
            report = covariance_rank(block.L)
            out.append(
                f"  {puma:<10}{geography:<14}{block.n_areas:>7,}"
                f"{report.n_rows:>9,}{report.rank:>6}{report.ceiling:>9}"
                f"{report.coverage:>9.2%}{report.margin:>10.1e}"
            )
    out.append("")
    out.append("  rank is of L; Sigma(alpha) is full rank for any alpha > 0, since D > 0.")
    out.append("  coverage = rank / rows: the share of directions the replicates inform.")
    out.append("  margin = smallest retained singular value / rank tolerance.")
    return "\n".join(out)


def summarise(
    area: StudyArea,
    by_puma: Mapping[str, Mapping[str, TargetBlock]],
) -> str:
    """The whole report: what is constrained, then the problems it produced."""
    first = by_puma[sorted(by_puma)[0]]

    inconsistent = [
        puma
        for puma, blocks in by_puma.items()
        for geography, block in blocks.items()
        if list(block.names) != list(first[geography].names)
    ]
    if inconsistent:
        raise ValueError(
            f"PUMAs {inconsistent} have different constraint columns from "
            f"{sorted(by_puma)[0]}; the specifications should not vary by PUMA"
        )

    zones = {g: sum(b[g].n_areas for b in by_puma.values()) for g in first}
    counts = ", ".join(f"{n:,} {g}s" for g, n in zones.items())
    return (
        f"Constraints for {area.slug}\n"
        f"  {len(by_puma)} PUMAs, {counts}\n\n"
        + format_constraints(area, first)
        + "\nPer PUMA\n"
        + format_pumas(by_puma)
        + "\n"
    )


def build_by_puma(
    area: StudyArea,
    tables: Sequence[ConstraintTable],
    *,
    policy: str = "model",
) -> dict[str, dict[str, TargetBlock]]:
    """Assemble one target block per PUMA per geography.

    The PUMA is the unit of a problem. Design weights are zero outside a
    record's own PUMA, so the joint problem is block diagonal by PUMA and
    solving it whole computes a great many structural zeros. Splitting also
    keeps every matrix small enough that ``q`` can stay dense, and makes the
    problems independent.

    Zones come from :func:`~pmedm_vb.data.geography.puma_crosswalk_whole`:
    *whole* PUMAs, which may reach past the study area's counties. A partial
    PUMA would be ill-posed -- its PUMS records represent all of it, so
    constraining only the part inside leaves the rest of its population with
    nowhere to go.

    What this gives up is the cross-PUMA blocks of ``Sigma``. Block groups in
    different PUMAs are expected to have correlated ACS error -- on the
    assumption, recorded but **unverified** in
    :mod:`pmedm_vb.assemble.targets`, that every cell derives from the same 80
    replicate weight sets -- and solving separately discards that correlation.
    """
    zones = puma_crosswalk_whole(area)
    by_geography: dict[str, list[ConstraintTable]] = {}
    for table in tables:
        by_geography.setdefault(table.geography, []).append(table)
    for specs in by_geography.values():
        for spec in specs:
            spec.validate(area)

    ordered_geographies = [g for g in GEOGRAPHY_ORDER if g in by_geography]
    ordered_geographies += [g for g in by_geography if g not in GEOGRAPHY_ORDER]

    out: dict[str, dict[str, TargetBlock]] = {}
    for puma, group in zones.groupby("puma_geoid"):
        blocks: dict[str, TargetBlock] = {}
        for geography in ordered_geographies:
            column = (
                "block_group_geoid"
                if geography == "block group"
                else "tract_geoid"
            )
            blocks[geography] = build_targets(
                area,
                by_geography[geography],
                geography=geography,
                geoids=sorted(group[column].unique()),
                policy=policy,
            )
        out[str(puma)] = blocks
    return out


def build_and_summarise(
    area: StudyArea,
    tables: Sequence[ConstraintTable],
    *,
    policy: str = "model",
) -> tuple[dict[str, dict[str, TargetBlock]], str]:
    """Assemble every PUMA's targets and summarise the result."""
    by_puma = build_by_puma(area, tables, policy=policy)
    return by_puma, summarise(area, by_puma)
