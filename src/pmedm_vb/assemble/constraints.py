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

#: ``JWTRNS`` codes per published means-of-transportation category. The
#: published table splits car/truck/van into drove-alone and carpooled, which
#: ``JWTRNS`` cannot do on its own -- that needs ``JWRIP`` -- so the split is
#: collapsed away instead, which is exact.
JWTRNS_TO_MODE = {
    "car_truck_van": ("01",),
    "public_transport": ("02", "03", "04", "05", "06"),
    "taxicab": ("07",),
    "motorcycle": ("08",),
    "bicycle": ("09",),
    "walked": ("10",),
    "worked_from_home": ("11",),
    "other_means": ("12",),
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
    renters here. **That is not verified**: the published titles cannot settle
    it, and the check is empirical, comparing each candidate mapping weighted by
    ``WGTP`` against the published estimates.
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
    """``B08301``: commute mode, over workers, counted per unit."""
    return ConstraintTable(
        table="B08301",
        universe="person",
        geography=geography,
        categories=tuple(
            Category(
                name=label,
                select=(lambda c: lambda p: p["JWTRNS"].isin(c))(codes),
            )
            for label, codes in JWTRNS_TO_MODE.items()
        ),
    )


def describe(area: StudyArea, tables: list[str], *, geography: str) -> pd.DataFrame:
    """Published cell keys and titles, which is what fills in ``published``.

    The one input the specifications above still need, and the one thing that
    cannot be worked out from the table's structure: which 1-based ``ORDER``
    Census assigned each category.
    """
    return cell_labels(area, tables, geography=geography).to_frame("title")


def _bands(values: pd.Series, boundaries: Sequence[float]) -> list[tuple[str, pd.Series]]:
    """Half-open bands ``[-inf, b0), [b0, b1), ... [bn, inf)`` over ``values``."""
    edges = list(boundaries)
    masks = [("lt_%g" % edges[0], values < edges[0])]
    for low, high in zip(edges, edges[1:]):
        masks.append(("%g_to_%g" % (low, high), (values >= low) & (values < high)))
    masks.append(("ge_%g" % edges[-1], values >= edges[-1]))
    return masks


def age_sex(boundaries: Sequence[int], geography: str = "block group") -> ConstraintTable:
    """``B01001``: sex crossed with age bands, over persons.

    ``boundaries`` are the collapsed age breaks, e.g. ``(18, 25, 35, 65)``.
    Each **must** be a break the table actually publishes -- a collapse is exact
    only when it merges published cells, and no boundary inside one can be
    recovered. :func:`describe` prints the published breaks to check against.

    Universe is the total population, so this table counts group-quarters
    residents. It is usable only because a GQ record enters as a one-person unit
    weighted by ``PWGTP``; weighting by ``WGTP`` alone would leave those people
    unplaceable while the published counts still included them.
    """

    def band(sex_code: str, mask_index: int):
        def select(persons: pd.DataFrame) -> pd.Series:
            _, mask = _bands(pd.to_numeric(persons["AGEP"]), boundaries)[mask_index]
            return (persons["SEX"] == sex_code) & mask
        return select

    n_bands = len(boundaries) + 1
    names = [n for n, _ in _bands(pd.Series([0.0]), boundaries)]
    return ConstraintTable(
        table="B01001",
        universe="person",
        geography=geography,
        categories=tuple(
            Category(name=f"{label}_{names[i]}", select=band(code, i))
            for label, code in (("male", "1"), ("female", "2"))
            for i in range(n_bands)
        ),
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
