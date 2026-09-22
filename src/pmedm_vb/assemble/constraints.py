"""Which published cells constrain a run, and which PUMS records feed each one.

This is the break-for-break correspondence between a Census table cell and a set
of PUMS records. It is the fiddliest part of setting up a PMEDM run and the
usual source of silent misfit, so it is written as data and checked against the
published cell list rather than expressed as branching code that cannot be
audited.

Two facts about the published files shape everything here.

**Collapsing is exact.** The replicate files carry all 80 replicates, not just a
margin of error, so summing published cells is a linear map whose covariance
follows by the same map -- including the correlations between the cells merged.
Granularity is therefore a modelling choice, not a property of the table. That
matters at block group, where fine categories are mostly zeros and every zero
cell falls back on the modelled ``w * k`` variance. A collapse is only exact
where its boundaries are real published boundaries, which
:meth:`ConstraintTable.validate` checks rather than assumes.

**The weighting unit is a household or a group-quarters person.** ``WGTP`` is
exactly zero on every ``TYPEHUGQ`` in ``{2, 3}`` record, with the weight on the
person record instead, and each GQ housing record matches exactly one person
record. So a GQ unit is a one-person household in the published structure
already. A table whose universe is the total population counts those people; one
whose universe is occupied housing units does not, and its GQ units contribute
zero. That is why :attr:`ConstraintTable.universe` exists and is not cosmetic.

Codes below were read from the 2020-2024 data dictionary, not recalled. Every
character column arrives as text (see :func:`pmedm_vb.data.pums.text_columns`),
so the comparisons are string comparisons -- ``HISP == "01"``, never ``== 1`` --
and a ``b`` code means the published field is blank, so it is tested with
``.isna()`` and never matched as a literal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from pmedm_vb.config import StudyArea
from pmedm_vb.data.summary import cell_labels

#: Universes a constraint table can be defined over, and which frame its
#: selector is applied to.
UNIVERSES = ("person", "household")


@dataclass(frozen=True)
class Category:
    """One column of ``X``: a name, a PUMS selector, and the cells it equals.

    Attributes
    ----------
    name:
        Label for the resulting ``X`` column.
    select:
        Takes the person or household frame (per the table's ``universe``) and
        returns a boolean mask. For a person-universe table the column is the
        *count* of matching persons in the unit; for a household-universe table
        the unit itself matches or does not, giving 0/1.
    published:
        Published cell keys this column must equal, e.g.
        ``("B03002_003",)``. More than one means a collapse. **Empty means
        unresolved**, which :meth:`ConstraintTable.validate` rejects -- the
        cells have to be read off the published table, not guessed from its
        structure.
    """

    name: str
    select: Callable[[pd.DataFrame], pd.Series]
    published: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintTable:
    """One published table used as a constraint, and its cell correspondence."""

    table: str
    universe: str
    geography: str
    categories: tuple[Category, ...] = field(default_factory=tuple)
    #: Published cells deliberately not used, so that "unused" and "forgotten"
    #: are distinguishable. A table total is the usual entry: including it
    #: alongside its parts makes ``X`` rank-deficient.
    waived: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.universe not in UNIVERSES:
            raise ValueError(
                f"universe must be one of {UNIVERSES}, got {self.universe!r}"
            )

    @property
    def unresolved(self) -> tuple[str, ...]:
        """Names of categories whose published cells have not been filled in."""
        return tuple(c.name for c in self.categories if not c.published)

    def validate(self, area: StudyArea) -> pd.DataFrame:
        """Check the correspondence against the published cell list.

        Four failures, each of which otherwise produces a converged solver and
        a wrong answer:

        * a category with no published cells -- unresolved, never a default;
        * a published cell that does not exist in this vintage;
        * a published cell claimed by two categories, double-counting it;
        * a published cell neither claimed nor waived, so silently dropped.

        Returns
        -------
        pandas.DataFrame
            The published cells with the category claiming each, for eyeballing
            the correspondence against the titles Census gives them.
        """
        if self.unresolved:
            raise ValueError(
                f"{self.table}: categories with no published cells: "
                f"{list(self.unresolved)}. Read them off the published table "
                f"with cell_labels(); they cannot be inferred from its structure"
            )

        labels = cell_labels(area, [self.table], geography=self.geography)
        available = set(labels.index)

        claimed: dict[str, str] = {}
        for category in self.categories:
            for cell in category.published:
                if cell not in available:
                    raise ValueError(
                        f"{self.table}: category {category.name!r} claims {cell!r}, "
                        f"which is not published at {self.geography} for "
                        f"{area.year}"
                    )
                if cell in claimed:
                    raise ValueError(
                        f"{self.table}: {cell!r} is claimed by both "
                        f"{claimed[cell]!r} and {category.name!r} -- a published "
                        f"cell may feed at most one constraint column"
                    )
                claimed[cell] = category.name

        dropped = available - set(claimed) - set(self.waived)
        if dropped:
            raise ValueError(
                f"{self.table}: {len(dropped)} published cell(s) neither used nor "
                f"waived, e.g. {sorted(dropped)[:5]}. Add them to a category or "
                f"to `waived`, so that leaving a cell out stays a decision"
            )

        return pd.DataFrame(
            {
                "title": labels,
                "category": pd.Series(claimed).reindex(labels.index),
            }
        )


# --------------------------------------------------------------------------
# Table specifications
#
# The selectors below are settled: every code was read from the 2020-2024 data
# dictionary. The `published` tuples are not, and are deliberately left empty --
# a cell key is TBLID plus a 1-based ORDER, so filling them in means reading
# which order Census published the categories in, which `describe()` prints.
# Writing a plausible ordering here would be a guess that validate() could not
# catch, because a wrong-but-existing cell key is indistinguishable from a right
# one until the numbers come out wrong.
# --------------------------------------------------------------------------

#: ``RAC1P`` codes making up each race category of ``B03002``. Nine PUMS codes
#: map onto seven published categories: the table's "American Indian and Alaska
#: Native alone" is three separate ``RAC1P`` codes -- American Indian alone,
#: Alaska Native alone, and AIAN tribes specified -- which is exactly the kind
#: of mismatch neither source announces.
RAC1P_TO_RACE = {
    "white": ("1",),
    "black": ("2",),
    "aian": ("3", "4", "5"),
    "asian": ("6",),
    "nhpi": ("7",),
    "other_race": ("8",),
    "two_or_more": ("9",),
}

#: Not Hispanic or Latino. Character width 2, so this is ``"01"`` and not ``1``:
#: read as a number it is indistinguishable from every other single-digit code.
NOT_HISPANIC = "01"

#: ``JWTRNS`` codes per published category of ``B08301``, with the cells each
#: maps to. The correspondence is exact at the subtotal level: ``JWTRNS`` has
#: one code for car/truck/van against the table's ``_002``, and its five
#: transit codes are exactly the five cells under ``_010``.
#:
#: The table does split car/truck/van into drove-alone and carpooled, and
#: carpooled again by occupancy, which ``JWTRNS`` cannot reproduce -- that
#: needs ``JWRIP``. Constraining the ``_002`` subtotal sidesteps it entirely
#: rather than approximating anything, so ``JWRIP`` is not read at all.
JWTRNS_TO_MODE = {
    "car_truck_van": (("01",), ("B08301_002",)),
    "public_transport": (("02", "03", "04", "05", "06"), ("B08301_010",)),
    "taxicab": (("07",), ("B08301_016",)),
    "motorcycle": (("08",), ("B08301_017",)),
    "bicycle": (("09",), ("B08301_018",)),
    "walked": (("10",), ("B08301_019",)),
    "worked_from_home": (("11",), ("B08301_021",)),
    "other_means": (("12",), ("B08301_020",)),
}

#: ``TEN`` codes per published tenure category. ``4`` is "occupied without
#: payment of rent", grouped with renters here -- believed correct and **not
#: verified**; it is one of the checks in the module notes. Null ``TEN`` is
#: group quarters or vacant, outside this table's occupied-housing-units
#: universe, and is excluded by the ``isin`` rather than by a null test.
TEN_TO_TENURE = {
    "owner": ("1", "2"),
    "renter": ("3", "4"),
}


def adjusted_household_income(households: pd.DataFrame) -> pd.Series:
    """``HINCP`` in constant dollars, the form ``B19001``'s brackets apply to.

    ``ADJINC`` is a *character* column with six implied decimal places, and a
    5-year file carries one factor per survey year, so this is a per-record
    multiply and not a single scalar for the file. ``HINCP`` is blank for
    vacant units and group quarters, which arrives as null and stays null.
    """
    return (
        pd.to_numeric(households["HINCP"], errors="coerce")
        * pd.to_numeric(households["ADJINC"], errors="coerce")
        / 1_000_000
    )


#: ``B19001``'s published brackets as ``(cell, lower, upper)`` in constant
#: dollars, read from the cell titles. ``None`` is an open end: the first
#: bracket is "Less than $10,000", which includes the negative incomes ``HINCP``
#: reports for business losses. Written down once so that a collapse can be
#: derived rather than declared -- a hand-written grouping of cells into bands
#: can disagree with its own boundary, and nothing would catch it.
B19001_BRACKETS = (
    ("B19001_002", None, 10_000),
    ("B19001_003", 10_000, 15_000),
    ("B19001_004", 15_000, 20_000),
    ("B19001_005", 20_000, 25_000),
    ("B19001_006", 25_000, 30_000),
    ("B19001_007", 30_000, 35_000),
    ("B19001_008", 35_000, 40_000),
    ("B19001_009", 40_000, 45_000),
    ("B19001_010", 45_000, 50_000),
    ("B19001_011", 50_000, 60_000),
    ("B19001_012", 60_000, 75_000),
    ("B19001_013", 75_000, 100_000),
    ("B19001_014", 100_000, 125_000),
    ("B19001_015", 125_000, 150_000),
    ("B19001_016", 150_000, 200_000),
    ("B19001_017", 200_000, None),
)


def race_ethnicity(geography: str = "block group") -> ConstraintTable:
    """``B03002``: Hispanic origin crossed with race, over persons.

    The eight categories are mutually exclusive and exhaust the table, summing
    to ``_001``. Everything else is waived because the table is nested: ``_002``
    subtotals the non-Hispanic races, ``_009`` subtotals ``_010``/``_011``, and
    ``_012`` subtotals the whole Hispanic side. Constraining a subtotal
    alongside its parts makes ``X`` rank-deficient.
    """
    published = {
        "nh_white": ("B03002_003",),
        "nh_black": ("B03002_004",),
        "nh_aian": ("B03002_005",),
        "nh_asian": ("B03002_006",),
        "nh_nhpi": ("B03002_007",),
        "nh_other_race": ("B03002_008",),
        "nh_two_or_more": ("B03002_009",),
    }

    def not_hispanic(codes: tuple[str, ...]):
        return lambda p: (p["HISP"] == NOT_HISPANIC) & p["RAC1P"].isin(codes)

    categories = [
        Category(
            name=f"nh_{label}",
            select=not_hispanic(codes),
            published=published[f"nh_{label}"],
        )
        for label, codes in RAC1P_TO_RACE.items()
    ]
    categories.append(
        Category(
            name="hispanic",
            select=lambda p: p["HISP"] != NOT_HISPANIC,
            published=("B03002_012",),
        )
    )
    return ConstraintTable(
        table="B03002",
        universe="person",
        geography=geography,
        categories=tuple(categories),
        waived=(
            "B03002_001",
            "B03002_002",
            "B03002_010",
            "B03002_011",
            *(f"B03002_{order:03d}" for order in range(13, 22)),
        ),
    )


def tenure(geography: str = "block group") -> ConstraintTable:
    """``B25003``: owner- and renter-occupied units, over households.

    The table publishes only owner and renter, so ``TEN`` code 4 -- occupied
    without payment of rent -- is folded into one of them. It is grouped with
    renters: two independent expectations agree on that, which is worth more
    than one and is still not a check, since the published titles cannot settle
    it either way.

    The check is empirical and costs nothing extra, because it falls out of the
    first marginal reproduction: weight each candidate mapping by ``WGTP`` and
    compare against the published ``B25003_002``/``_003``. Only one of them can
    match, so this is confirmed the first time the table is built rather than
    carried as an assumption.
    """
    published = {"owner": ("B25003_002",), "renter": ("B25003_003",)}
    return ConstraintTable(
        table="B25003",
        universe="household",
        geography=geography,
        categories=tuple(
            Category(
                name=label,
                select=(lambda c: lambda h: h["TEN"].isin(c))(codes),
                published=published[label],
            )
            for label, codes in TEN_TO_TENURE.items()
        ),
        waived=("B25003_001",),
    )


def means_of_transportation(geography: str = "block group") -> ConstraintTable:
    """``B08301``: commute mode, over workers, counted per unit.

    The universe is workers 16 and over, and it needs no explicit filter:
    ``JWTRNS`` is blank for everyone outside it -- not in the labor force,
    under 16, unemployed, or employed but not at work -- so a blank reads as
    null and ``isin`` excludes it. Testing ``JWTRNS != "bb"`` instead would
    match every record, since ``bb`` denotes a blank rather than being a value
    the file contains.

    The eight categories partition the table's top level and sum to ``_001``.
    The two subtotals ``_002`` and ``_010`` are *used*, so everything beneath
    them is waived: the carpool-occupancy detail under ``_004`` and the five
    transit modes under ``_010``.
    """
    return ConstraintTable(
        table="B08301",
        universe="person",
        geography=geography,
        categories=tuple(
            Category(
                name=label,
                select=(lambda c: lambda p: p["JWTRNS"].isin(c))(codes),
                published=cells,
            )
            for label, (codes, cells) in JWTRNS_TO_MODE.items()
        ),
        waived=(
            "B08301_001",
            *(f"B08301_{order:03d}" for order in range(3, 10)),
            *(f"B08301_{order:03d}" for order in range(11, 16)),
        ),
    )


def describe(area: StudyArea, tables: list[str], *, geography: str) -> pd.DataFrame:
    """Published cell keys and titles, which is what fills in ``published``.

    The one input the specifications above still need, and the one thing that
    cannot be worked out from the table's structure: which 1-based ``ORDER``
    Census assigned each category.
    """
    return cell_labels(area, tables, geography=geography).to_frame("title")


#: ``B01001``'s 23 published age bands as ``(lower, upper)``, upper exclusive.
#: ``None`` is the open top. Read from the published titles: "Under 5 years",
#: "5 to 9 years", "18 and 19 years", "20 years", "85 years and over". Recorded
#: once so a collapse is derived from them rather than written out.
B01001_BANDS = (
    (0, 5), (5, 10), (10, 15), (15, 18), (18, 20), (20, 21), (21, 22), (22, 25),
    (25, 30), (30, 35), (35, 40), (40, 45), (45, 50), (50, 55), (55, 60),
    (60, 62), (62, 65), (65, 67), (67, 70), (70, 75), (75, 80), (80, 85),
    (85, None),
)

#: Cell ``ORDER`` of each sex's first age band. The table runs total, male
#: total, 23 male bands, female total, 23 female bands -- so band ``i`` is at
#: ``offset + i`` and the two blocks are identical in structure.
B01001_SEX_OFFSET = {"male": 3, "female": 27}

#: ``SEX`` codes. Character width 1, so these are strings.
SEX_CODES = {"male": "1", "female": "2"}

#: Age bands wanted across every table that publishes ages, as upper-exclusive
#: boundaries: 0-4, 5-13, 14-17, 18-34, 35-49, 50-64, 65+. Tables break in
#: different places, so this is the *request*; :func:`snap_boundaries` maps it
#: onto what a given table actually publishes.
PREFERRED_AGE_BOUNDARIES = (5, 14, 18, 35, 50, 65)


def snap_boundaries(
    requested: Sequence[int],
    published: Sequence[int],
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Move each requested boundary to the nearest one a table publishes.

    A collapse is exact only at a published break, so a boundary that falls
    inside a cell cannot be used as asked. Moving it to the nearest break keeps
    the intended split -- one year off -- where dropping it would merge the two
    bands entirely and lose the distinction.

    Ties go to the lower break, which keeps the younger band smaller: for age
    bands the younger side is usually the one being isolated.

    Returns
    -------
    tuple
        The snapped boundaries, deduplicated and sorted, and the moves actually
        made as ``(requested, used)`` pairs. **Check the second element.** The
        snapping is exact arithmetic on the published breaks, but whether a
        one-year shift is acceptable is a modelling judgement this cannot make,
        so it is reported rather than assumed.
    """
    available = sorted(set(published))
    if not available:
        raise ValueError("no published boundaries to snap to")

    snapped, moves = [], []
    for boundary in requested:
        nearest = min(available, key=lambda edge: (abs(edge - boundary), edge))
        snapped.append(nearest)
        if nearest != boundary:
            moves.append((boundary, nearest))
    return tuple(sorted(set(snapped))), tuple(moves)


def b01001_boundaries() -> tuple[int, ...]:
    """:data:`PREFERRED_AGE_BOUNDARIES` snapped onto ``B01001``'s breaks.

    Yields ``(5, 15, 18, 35, 50, 65)``: 0-4, 5-14, 15-17, 18-34, 35-49, 50-64,
    65+. Only 14 moves -- ``B01001`` publishes 10-14 then 15-17, so 14 falls
    inside a cell and 15 is the nearest break. Every other requested boundary
    is published as asked.
    """
    published = [upper for _, upper in B01001_BANDS if upper is not None]
    snapped, _ = snap_boundaries(PREFERRED_AGE_BOUNDARIES, published)
    return snapped


def age_sex(
    boundaries: Sequence[int] | None = None,
    geography: str = "block group",
) -> ConstraintTable:
    """``B01001``: sex crossed with collapsed age bands, over persons.

    ``boundaries`` are the collapsed breaks, each of which **must** be one
    ``B01001`` publishes -- a collapse is exact only where it merges whole
    published cells, and no break inside one can be recovered. Passing ``None``
    uses :func:`b01001_boundaries`, the preferred bands snapped onto this
    table: 0-4, 5-14, 15-17, 18-34, 35-49, 50-64, 65+.

    Two requested splits are not available here, and neither is approximated
    silently. A break at 14 falls inside the published 10-14 cell, so the
    preferred 5-13 / 14-17 becomes 5-14 / 15-17 -- one year off, with the split
    preserved. A break at 6, wanted to separate preschool from school-age
    children, falls inside "5 to 9 years" and has no near alternative at all;
    0-4 is the closest exact band. Category names are generated from the
    boundaries actually used, so they cannot describe a band the cells do not
    contain.

    Which cells make up each band is *derived* from :data:`B01001_BANDS` rather
    than written out, for the reason given on :func:`household_income`: a
    hand-written grouping can contradict its own declared boundary while every
    cell key it names exists and is claimed exactly once, so ``validate`` could
    not catch it.

    Universe is the total population, so this counts group-quarters residents.
    It is usable only because a GQ record enters as a one-person unit weighted
    by ``PWGTP``; weighting by ``WGTP`` alone would leave those people
    unplaceable while the published counts still included them.
    """
    if boundaries is None:
        boundaries = b01001_boundaries()
    published_edges = {upper for _, upper in B01001_BANDS if upper is not None}
    unknown = sorted(set(boundaries) - published_edges)
    if unknown:
        raise ValueError(
            f"{unknown} are not published breaks of B01001, so collapsing there "
            f"would not be exact. Available: {sorted(published_edges)}"
        )

    high = float("inf")
    edges = [0, *sorted(boundaries), high]

    categories = []
    claimed: list[str] = []
    for sex, offset in B01001_SEX_OFFSET.items():
        for lower, upper in zip(edges, edges[1:]):
            cells = tuple(
                f"B01001_{offset + index:03d}"
                for index, (band_low, band_high) in enumerate(B01001_BANDS)
                if band_low >= lower and (high if band_high is None else band_high) <= upper
            )
            claimed.extend(cells)
            span = (
                f"lt_{upper:.0f}" if lower == 0
                else f"ge_{lower:.0f}" if upper == high
                else f"{lower:.0f}_to_{upper:.0f}"
            )

            def select(persons, code=SEX_CODES[sex], lower=lower, upper=upper):
                age = pd.to_numeric(persons["AGEP"], errors="coerce")
                return (persons["SEX"] == code) & (age >= lower) & (age < upper)

            categories.append(
                Category(name=f"{sex}_{span}", select=select, published=cells)
            )

    every = [
        f"B01001_{offset + index:03d}"
        for offset in B01001_SEX_OFFSET.values()
        for index in range(len(B01001_BANDS))
    ]
    if sorted(claimed) != sorted(every):
        raise AssertionError(
            f"age collapse does not partition B01001: "
            f"{sorted(set(every) - set(claimed))} unassigned, "
            f"{[c for c in claimed if claimed.count(c) > 1]} assigned twice"
        )

    return ConstraintTable(
        table="B01001",
        universe="person",
        geography=geography,
        categories=tuple(categories),
        # The table total and each sex's subtotal: constraining a subtotal
        # alongside its parts adds an X column that is their exact sum.
        waived=("B01001_001", "B01001_002", "B01001_026"),
    )


def household_income(
    boundaries: Sequence[float] = (25_000, 50_000, 75_000, 100_000, 150_000),
    geography: str = "block group",
) -> ConstraintTable:
    """``B19001``: household income bands, over households.

    ``boundaries`` are the band edges in constant dollars, and each **must** be
    a break ``B19001`` actually publishes -- a collapse is exact only where it
    merges published cells, and no split inside one can be recovered. The
    mapping from band to published cells is then *derived* from
    :data:`B19001_BRACKETS` rather than written out, so a band and its cells
    cannot disagree; every bracket is asserted to land in exactly one band.

    Income goes through :func:`adjusted_household_income` first: the published
    brackets are in reference-year dollars and ``HINCP`` is not.
    """
    published_edges = {hi for _, _, hi in B19001_BRACKETS if hi is not None}
    unknown = sorted(set(boundaries) - published_edges)
    if unknown:
        raise ValueError(
            f"{unknown} are not published breaks of B19001, so collapsing there "
            f"would not be exact. Available: {sorted(published_edges)}"
        )

    low = float("-inf")
    high = float("inf")
    edges = [low, *sorted(boundaries), high]

    categories = []
    claimed: list[str] = []
    for lower, upper in zip(edges, edges[1:]):
        cells = tuple(
            cell
            for cell, bracket_low, bracket_high in B19001_BRACKETS
            if (low if bracket_low is None else bracket_low) >= lower
            and (high if bracket_high is None else bracket_high) <= upper
        )
        claimed.extend(cells)
        name = (
            f"lt_{upper:.0f}" if lower == low
            else f"ge_{lower:.0f}" if upper == high
            else f"{lower:.0f}_to_{upper:.0f}"
        )

        def select(households, lower=lower, upper=upper):
            income = adjusted_household_income(households)
            return (income >= lower) & (income < upper)

        categories.append(Category(name=name, select=select, published=cells))

    every = [cell for cell, _, _ in B19001_BRACKETS]
    if sorted(claimed) != sorted(every):
        raise AssertionError(
            f"income collapse does not partition B19001: "
            f"{sorted(set(every) - set(claimed))} unassigned, "
            f"{[c for c in claimed if claimed.count(c) > 1]} assigned twice"
        )

    return ConstraintTable(
        table="B19001",
        universe="household",
        geography=geography,
        categories=tuple(categories),
        waived=("B19001_001",),
    )
