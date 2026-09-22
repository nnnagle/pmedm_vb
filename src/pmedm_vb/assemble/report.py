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


def format_constraints(
    area: StudyArea,
    blocks: Mapping[str, TargetBlock],
) -> str:
    """One section per geography: each table, its size, and its column labels.

    Column names arrive as ``"{table}.{category}"``, so the table each belongs
    to is recovered from the name rather than passed in separately -- the report
    then describes what was actually built, not what was intended.
    """
    titles = table_list(area).set_index("TBLID")["TITLE"]
    out = [f"Constraints for {area.slug}", ""]

    ordered = [g for g in GEOGRAPHY_ORDER if g in blocks]
    ordered += [g for g in blocks if g not in GEOGRAPHY_ORDER]

    for geography in ordered:
        block = blocks[geography]
        grouped: dict[str, list[str]] = {}
        for name in block.names:
            table, _, category = name.partition(".")
            grouped.setdefault(table, []).append(category)

        rows = block.n_areas * block.n_categories
        out.append(f"{geography.title()}  --  {block.n_areas:,} areas")
        for table, categories in grouped.items():
            title = titles.get(table, "(title not in the published list)")
            out.append(f"  {table}  {title}")
            out.append(f"  {'':8}{len(categories)} constraints")
            out.extend(f"  {'':8}{line}" for line in _wrap(categories))
        out.append(
            f"  {'':8}"
            f"[{len(grouped)} tables, {block.n_categories} constraints, "
            f"{rows:,} rows]"
        )
        out.append("")
    return "\n".join(out)


def format_rank(blocks: Mapping[str, TargetBlock]) -> str:
    """Rank of each block's design covariance, with the evidence for it.

    Reports the rank of ``L``, not of ``Sigma``. ``Sigma(alpha) = D +
    (1-alpha)(4/80) L L'`` is **full rank** for any ``alpha > 0``, because
    ``D`` is strictly positive -- the rank-80 ceiling belongs to the replicate
    part alone. Collapsing the two into one number would hide the distinction
    that matters: how much structure the replicates carry, versus whether the
    matrix can be inverted.

    The tolerance and the margin above it are printed because a rank that
    equals its own ceiling is indistinguishable, from the number alone, from a
    rank that was never computed. The margin says which it is.
    """
    out = ["Covariance rank", ""]
    for geography, block in blocks.items():
        report = covariance_rank(block.L)
        values = report.singular_values
        out.append(f"  {geography.title()}")
        out.append(f"    constraint rows        {report.n_rows:>10,}")
        out.append(f"    replicates             {N_REPLICATES:>10}")
        out.append(
            f"    rank of L              {report.rank:>10}"
            f"   of a possible {report.ceiling}"
            + (f", short by {report.deficiency}" if report.deficiency else "")
        )
        out.append(
            f"    below tolerance        {len(values) - report.rank:>10}"
            f"   (tolerance {report.tolerance:.2e})"
        )
        if report.rank:
            out.append(
                f"    smallest retained      {values[report.rank - 1]:>10.3e}"
                f"   {report.margin:.1e}x the tolerance"
            )
            out.append(
                f"    spectrum               {values[0]:>10.3e}"
                f" .. {values[report.rank - 1]:.3e}"
                f"   (ratio {values[0] / values[report.rank - 1]:.1f})"
            )
        out.append(
            f"    replicate coverage     {report.coverage:>10.2%}"
            f"   {report.rank} of {report.n_rows:,} directions;"
            f" the rest carry D only"
        )
        out.append(
            f"    Sigma(alpha) rank      {report.n_rows:>10,}"
            f"   full for alpha > 0, since D > 0"
        )
        out.append("")
    return "\n".join(out)


def summarise(area: StudyArea, blocks: Mapping[str, TargetBlock]) -> str:
    """The whole report: what is constrained, then how much structure it has."""
    return format_constraints(area, blocks) + "\n" + format_rank(blocks)


def build_and_summarise(
    area: StudyArea,
    tables: Sequence[ConstraintTable],
    *,
    policy: str = "model",
) -> tuple[dict[str, TargetBlock], str]:
    """Build a target block per geography and summarise the result.

    Groups the specifications by their own ``geography`` rather than taking it
    as an argument, so a table can only be built at the level it was written
    for. Each spec is validated against the published cell list first: a
    misdescribed table should fail before any arithmetic, not produce a report
    that looks reasonable.

    Returns
    -------
    tuple
        The blocks by geography, and the report text.
    """
    by_geography: dict[str, list[ConstraintTable]] = {}
    for table in tables:
        by_geography.setdefault(table.geography, []).append(table)

    blocks: dict[str, TargetBlock] = {}
    for geography in [g for g in GEOGRAPHY_ORDER if g in by_geography] + [
        g for g in by_geography if g not in GEOGRAPHY_ORDER
    ]:
        specs = by_geography[geography]
        for spec in specs:
            spec.validate(area)
        blocks[geography] = build_targets(
            area, specs, geography=geography, policy=policy
        )
    return blocks, summarise(area, blocks)
