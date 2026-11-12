"""Builders for the published tables the embedding adds as targets.

Each builder returns a :class:`~pmedm_vb.assemble.constraints.ConstraintTable`
in the style of :mod:`pmedm_vb.assemble.constraints`: a selector per category,
the published cells it equals, and the cells deliberately waived. They are
used here for their **categories and universes**, not as constraints -- stage 1
fits no published values -- but they are written to the same standard so that
a later stage can constrain or score them unchanged.

Conventions, as in ``constraints.py``:

* Codes were read from the **2020-2024** PUMS data dictionary (the vintage the
  project's study areas use), not recalled. Character columns arrive as text,
  so comparisons are string comparisons, and a ``b`` code is a blank tested
  with ``.isna()``.
* Cells were read from the API ``groups`` metadata of the 2024 5-year tables
  (:func:`pmedm_vb.data.acs_api.api_cell_labels`), and every table is checked
  against it by :func:`validate_against_api`: every published cell claimed or
  waived, none twice, and only totals and subtotals waived.
* **Finest reproducible cells.** Each table is used at its finest published
  cells wherever the PUMS reproduces them exactly; where it cannot, the
  nearest published subtotal is used and the reason is written beside it.
* **Categories partition the universe.** Within a table's universe
  (:data:`TABLES` carries it as a selector) every unit or person falls in
  exactly one category, so a unit's universe count is the sum of its category
  counts. That is what the embedding's per-table multinomial needs, and it is
  checked against the statewide PUMS rather than assumed.
* Dollar amounts are put in reference-year dollars first: ``ADJINC`` for
  income (:func:`~pmedm_vb.assemble.constraints.adjusted_household_income`),
  ``ADJHSG`` for housing amounts, as the data dictionary directs.

Substitutions against the list agreed with the user, each for a reason found
while building:

* ``B25024``, ``B25034``, ``B25017``, ``B25041`` (units in structure, year
  built, rooms, bedrooms) count **all housing units, vacant ones included**
  (API ``groups.json`` universe "Housing units"), which the model does not
  carry -- the same reason ``constraints.py`` gives for dropping ``B25034``.
  Their occupied-unit versions, crossed with tenure and published at block
  group, are used instead: ``B25032``, ``B25036``, ``B25020``, ``B25042``.
* ``B07201``'s universe is people "living in a Metropolitan Statistical Area";
  deciding who does needs the residence's geography, which the embedding
  excludes. ``B07003`` (mobility by sex, tract only) has the same mobility
  content over everyone 1 and over, and is used instead.
* ``B25038`` (tenure by year householder moved in) publishes calendar-year
  bands; the PUMS has only ``MV``, a duration band relative to the interview,
  and a 5-year file pools five interview years, so no band boundary is
  reproducible. ``MV`` is carried as an attribute instead
  (:mod:`pmedm_vb.embed.attributes`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pmedm_vb.assemble.constraints import (
    B19001_BRACKETS,
    SEX_CODES,
    TEN_TO_TENURE,
    Category,
    ConstraintTable,
    adjusted_household_income,
)
from pmedm_vb.config import StudyArea
from pmedm_vb.data.acs_api import api_cell_labels

#: ``PUMS`` frame -> boolean mask of the records in a table's universe.
Universe = Callable[[pd.DataFrame], pd.Series]

#: Highest value, for open-ended top bands.
INF = float("inf")


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _cell(table: str, order: int) -> str:
    return f"{table}_{order:03d}"


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def _band_name(lower: float, upper: float) -> str:
    if lower == -INF:
        return f"lt_{_fmt(upper)}"
    if upper == INF:
        return f"ge_{_fmt(lower)}"
    return f"{_fmt(lower)}_to_{_fmt(upper)}"


def _banded(
    table: str,
    bands: Sequence[tuple[float, float]],
    first: int,
    value: Callable[[pd.DataFrame], pd.Series],
    *,
    within: Callable[[pd.DataFrame], pd.Series] | None = None,
    prefix: str = "",
) -> list[Category]:
    """One category per ``[lower, upper)`` band of ``value``, at consecutive
    cells from ``first``, optionally restricted to ``within``."""
    categories = []
    for index, (lower, upper) in enumerate(bands):

        def select(frame, lower=lower, upper=upper):
            v = value(frame)
            mask = (v >= lower) & (v < upper)
            return mask & within(frame) if within is not None else mask

        categories.append(Category(
            name=f"{prefix}{_band_name(lower, upper)}",
            select=select,
            published=(_cell(table, first + index),),
        ))
    return categories


# --------------------------------------------------------------------------
# Universes
# --------------------------------------------------------------------------


def occupied(units: pd.DataFrame) -> pd.Series:
    """Occupied housing units: a tenure is recorded. Blank ``TEN`` is group
    quarters (or vacant, already dropped by the unit frame)."""
    return units["TEN"].notna()


def owner(units: pd.DataFrame) -> pd.Series:
    return units["TEN"].isin(TEN_TO_TENURE["owner"])


def renter(units: pd.DataFrame) -> pd.Series:
    """``TEN`` 3 and 4, as :data:`~pmedm_vb.assemble.constraints.TEN_TO_TENURE`
    groups them; ``B25063``'s "No cash rent" line confirms that the renter
    universe includes ``TEN`` 4, occupied without payment of rent."""
    return units["TEN"].isin(TEN_TO_TENURE["renter"])


def everyone(persons: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=persons.index)


def aged(minimum: int) -> Universe:
    return lambda persons: _num(persons, "AGEP") >= minimum


def commuters(persons: pd.DataFrame) -> pd.Series:
    """Workers 16 and over who did not work from home: a ``JWTRNS`` mode other
    than 11 (worked from home). Blank ``JWTRNS`` is outside the workforce."""
    return persons["JWTRNS"].notna() & (persons["JWTRNS"] != "11")


#: ``ESR`` codes for the armed forces, at work or with a job but not at work.
ESR_ARMED_FORCES = ("4", "5")

#: ``MIL`` code for "Now on active duty".
MIL_ACTIVE_DUTY = "1"

#: ``RELSHIPP`` code of an institutionalized group-quarters resident.
RELSHIPP_INSTITUTIONAL = "37"


def civilian_noninstitutional(persons: pd.DataFrame) -> pd.Series:
    """The civilian noninstitutionalized population (``B27010``, ``B18101``).

    The 2024 ACS Subject Definitions: it "excludes active-duty military
    personnel and the population living in institutional group quarters" and
    includes noninstitutional group quarters. Institutional residents are
    ``RELSHIPP`` 37. Active duty is taken as ``MIL`` 1 *or* an armed-forces
    ``ESR`` (4, 5): in Tennessee 2020-2024, 46 records have an armed-forces
    ``ESR`` with ``MIL`` 3 ("only on active duty for training in Reserves/
    National Guard"), and they are counted as active duty here because ``ESR``
    places them in the armed forces at the time of interview. That reading is
    a judgement, not something the documentation settles.
    """
    military = persons["ESR"].isin(ESR_ARMED_FORCES) | (persons["MIL"] == MIL_ACTIVE_DUTY)
    return (persons["RELSHIPP"] != RELSHIPP_INSTITUTIONAL) & ~military


# --------------------------------------------------------------------------
# Housing stock: tenure by structure, year built, rooms, bedrooms, crowding
# --------------------------------------------------------------------------

#: ``BLD`` code per ``B25032`` cell, as the offset from the tenure subtotal,
#: in the table's published order.
BLD_CELLS = {
    "one_detached": ("02", 1),
    "one_attached": ("03", 2),
    "two": ("04", 3),
    "three_four": ("05", 4),
    "five_to_nine": ("06", 5),
    "ten_to_19": ("07", 6),
    "twenty_to_49": ("08", 7),
    "fifty_plus": ("09", 8),
    "mobile_home": ("01", 9),
    "boat_rv_van": ("10", 10),
}

#: Cell ``ORDER`` of each tenure subtotal, per table.
TENURE_SUBTOTALS = {
    "B25032": {"owner": 2, "renter": 13},
    "B25036": {"owner": 2, "renter": 13},
    "B25020": {"owner": 2, "renter": 12},
    "B25042": {"owner": 2, "renter": 9},
    "B25014": {"owner": 2, "renter": 8},
    "B25007": {"owner": 2, "renter": 12},
    "B25009": {"owner": 2, "renter": 10},
}


def _by_tenure(
    table: str,
    variables: tuple[str, ...],
    detail: Callable[[str, int], list[Category]],
    geography: str,
) -> ConstraintTable:
    """A tenure-crossed household table: ``detail(tenure, subtotal_order)``
    builds one tenure's categories; the total and both subtotals are waived."""
    categories = []
    subtotals = TENURE_SUBTOTALS[table]
    for tenure_label, first in subtotals.items():
        categories.extend(detail(tenure_label, first))
    return ConstraintTable(
        table=table,
        variables=("TEN", *variables),
        universe="household",
        geography=geography,
        categories=tuple(categories),
        waived=(_cell(table, 1), *(_cell(table, o) for o in subtotals.values())),
    )


def _tenure_mask(label: str) -> Callable[[pd.DataFrame], pd.Series]:
    codes = TEN_TO_TENURE[label]
    return lambda units: units["TEN"].isin(codes)


def tenure_by_units_in_structure(geography: str = "block group") -> ConstraintTable:
    """``B25032``: tenure by units in structure, over occupied housing units.

    One ``BLD`` code per published cell, both tenures. Used in place of
    ``B25024``, whose universe includes vacant units (module notes).
    """
    def detail(tenure_label, first):
        within = _tenure_mask(tenure_label)
        return [
            Category(
                name=f"{tenure_label}_{label}",
                select=(lambda c, w: lambda u: w(u) & (u["BLD"] == c))(code, within),
                published=(_cell("B25032", first + offset),),
            )
            for label, (code, offset) in BLD_CELLS.items()
        ]

    return _by_tenure("B25032", ("BLD",), detail, geography)


#: ``B25036``'s year-built bands as ``[lower, upper)`` years, in published
#: order (newest first). ``YRBLT`` publishes decades before 2020 and single
#: years from 2020, each coded by its first year (``1939`` = "1939 or
#: earlier"), so every band is a union of whole codes.
YEAR_BUILT_BANDS = (
    (2020, INF), (2010, 2020), (2000, 2010), (1990, 2000), (1980, 1990),
    (1970, 1980), (1960, 1970), (1950, 1960), (1940, 1950), (-INF, 1940),
)


def tenure_by_year_built(geography: str = "block group") -> ConstraintTable:
    """``B25036``: tenure by year structure built, over occupied housing units.

    Used in place of ``B25034``, whose universe includes vacant units.
    """
    def detail(tenure_label, first):
        return _banded(
            "B25036", YEAR_BUILT_BANDS, first + 1,
            lambda u: _num(u, "YRBLT"), within=_tenure_mask(tenure_label),
            prefix=f"{tenure_label}_built_",
        )

    return _by_tenure("B25036", ("YRBLT",), detail, geography)


#: ``B25020``'s room bands: 1 to 8 rooms singly, then 9 or more.
ROOM_BANDS = tuple((n, n + 1) for n in range(1, 9)) + ((9, INF),)


def tenure_by_rooms(geography: str = "block group") -> ConstraintTable:
    """``B25020``: tenure by rooms (``RMSP``), over occupied housing units.

    Used in place of ``B25017``, whose universe includes vacant units.
    """
    def detail(tenure_label, first):
        return _banded(
            "B25020", ROOM_BANDS, first + 1,
            lambda u: _num(u, "RMSP"), within=_tenure_mask(tenure_label),
            prefix=f"{tenure_label}_rooms_",
        )

    return _by_tenure("B25020", ("RMSP",), detail, geography)


#: ``B25042``'s bedroom bands: none, 1 to 4 singly, then 5 or more.
BEDROOM_BANDS = tuple((n, n + 1) for n in range(0, 5)) + ((5, INF),)


def tenure_by_bedrooms(geography: str = "block group") -> ConstraintTable:
    """``B25042``: tenure by bedrooms (``BDSP``), over occupied housing units.

    Used in place of ``B25041``, whose universe includes vacant units.
    """
    def detail(tenure_label, first):
        return _banded(
            "B25042", BEDROOM_BANDS, first + 1,
            lambda u: _num(u, "BDSP"), within=_tenure_mask(tenure_label),
            prefix=f"{tenure_label}_bedrooms_",
        )

    return _by_tenure("B25042", ("BDSP",), detail, geography)


#: ``B25014``'s occupants-per-room bands as ``(lower, upper]`` -- closed above,
#: as the titles read: "0.50 or less", "0.51 to 1.00", "1.01 to 1.50",
#: "1.51 to 2.00", "2.01 or more".
CROWDING_BANDS = ((-INF, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, INF))


def tenure_by_occupants_per_room(geography: str = "block group") -> ConstraintTable:
    """``B25014``: tenure by occupants per room, over occupied housing units.

    Occupants per room is ``NP / RMSP`` -- persons in the household over rooms
    -- an exact rational, so the published two-decimal boundaries are applied
    to it directly, closed above as the titles read.
    """
    names = ("le_0.50", "0.51_to_1.00", "1.01_to_1.50", "1.51_to_2.00", "ge_2.01")

    def detail(tenure_label, first):
        within = _tenure_mask(tenure_label)
        categories = []
        for index, ((lower, upper), name) in enumerate(zip(CROWDING_BANDS, names)):

            def select(u, lower=lower, upper=upper, within=within):
                ratio = _num(u, "NP") / _num(u, "RMSP")
                return within(u) & (ratio > lower) & (ratio <= upper)

            categories.append(Category(
                name=f"{tenure_label}_{name}", select=select,
                published=(_cell("B25014", first + 1 + index),),
            ))
        return categories

    return _by_tenure("B25014", ("NP", "RMSP"), detail, geography)


def _adjusted_housing(units: pd.DataFrame, column: str) -> pd.Series:
    """A housing dollar amount in reference-year dollars: ``ADJHSG`` carries
    six implied decimal places and one factor per survey year."""
    return _num(units, column) * _num(units, "ADJHSG") / 1_000_000


#: ``B25075``'s value brackets in dollars, ``[lower, upper)``, in published
#: order from cell ``_002``.
VALUE_EDGES = (
    10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 50_000, 60_000,
    70_000, 80_000, 90_000, 100_000, 125_000, 150_000, 175_000, 200_000,
    250_000, 300_000, 400_000, 500_000, 750_000, 1_000_000, 1_500_000, 2_000_000,
)


def _edges_to_bands(edges: Sequence[float]) -> tuple[tuple[float, float], ...]:
    full = (-INF, *edges, INF)
    return tuple(zip(full, full[1:]))


def value(geography: str = "block group") -> ConstraintTable:
    """``B25075``: value, over owner-occupied housing units.

    ``VALP`` adjusted by ``ADJHSG``. ``VALP`` is set for every owner and blank
    for every renter in Tennessee 2020-2024, so the brackets partition owners.
    """
    return ConstraintTable(
        table="B25075",
        variables=("TEN", "VALP", "ADJHSG"),
        universe="household",
        geography=geography,
        categories=tuple(_banded(
            "B25075", _edges_to_bands(VALUE_EDGES), 2,
            lambda u: _adjusted_housing(u, "VALP"), within=owner, prefix="value_",
        )),
        waived=("B25075_001",),
    )


#: ``B25063``'s cash-rent brackets in dollars per month, ``[lower, upper)``,
#: from cell ``_003``.
GROSS_RENT_EDGES = (
    100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800,
    900, 1_000, 1_250, 1_500, 2_000, 2_500, 3_000, 3_500,
)

#: ``TEN`` code of a unit occupied without payment of rent.
TEN_NO_CASH_RENT = "4"


def gross_rent(geography: str = "block group") -> ConstraintTable:
    """``B25063``: gross rent, over renter-occupied housing units.

    Cash renters (``TEN`` 3) by ``GRNTP`` adjusted by ``ADJHSG``; "No cash
    rent" is ``TEN`` 4. ``GRNTP`` is set for every ``TEN`` 3 unit and blank for
    every ``TEN`` 4 unit in Tennessee 2020-2024.
    """
    cash = lambda u: u["TEN"] == "3"  # noqa: E731
    categories = _banded(
            "B25063", _edges_to_bands(GROSS_RENT_EDGES), 3,
        lambda u: _adjusted_housing(u, "GRNTP"), within=cash, prefix="rent_",
    )
    categories.append(Category(
        name="no_cash_rent", select=lambda u: u["TEN"] == TEN_NO_CASH_RENT,
        published=("B25063_027",),
    ))
    return ConstraintTable(
        table="B25063",
        variables=("TEN", "GRNTP", "ADJHSG"),
        universe="household",
        geography=geography,
        categories=tuple(categories),
        waived=("B25063_001", "B25063_002"),
    )


#: Cost-burden bands in percent of income, ``[lower, upper)``, shared by
#: ``B25070`` and ``B25091``: "Less than 10.0 percent" ... "50.0 percent or
#: more". ``GRPIP`` and ``OCPIP`` are whole percents (1-100, 101 for "101 or
#: more"), so a band like "10.0 to 14.9" is 10-14. Whether the PUMS rounds or
#: truncates to the whole percent is not documented in the dictionary, so a
#: household within half a percent of a boundary may sit one band off.
BURDEN_BANDS = _edges_to_bands((10, 15, 20, 25, 30, 35, 40, 50))


def rent_burden(geography: str = "block group") -> ConstraintTable:
    """``B25070``: gross rent as a percentage of household income, over
    renter-occupied housing units.

    "Not computed" is a renter with a blank ``GRPIP``: every ``TEN`` 4 unit
    (no rent payment) and the cash renters with no or negative household
    income (713 in Tennessee 2020-2024), per the dictionary's N/A line.
    """
    categories = _banded(
            "B25070", BURDEN_BANDS, 2,
        lambda u: _num(u, "GRPIP"), within=renter, prefix="burden_",
    )
    categories.append(Category(
        name="not_computed", select=lambda u: renter(u) & u["GRPIP"].isna(),
        published=("B25070_011",),
    ))
    return ConstraintTable(
        table="B25070",
        variables=("TEN", "GRPIP"),
        universe="household",
        geography=geography,
        categories=tuple(categories),
        waived=("B25070_001",),
    )


#: ``TEN`` code per mortgage status, with ``B25091``'s subtotal cell.
#: The 2024 Subject Definitions count every debt secured on the property as a
#: mortgage, home equity loans included, and "not mortgaged" as "owned free and
#: clear of debt" -- exactly ``TEN`` 1 ("Owned with mortgage or loan (include
#: home equity loans)") against ``TEN`` 2 ("Owned free and clear"). ``MRGX``
#: is *not* used: it describes the first mortgage only, and is 3 ("None") for
#: 1,651 ``TEN`` 1 units in Tennessee -- presumably home-equity-only owners,
#: whom the definition counts as mortgaged.
MORTGAGE_STATUS = {"with_mortgage": ("1", 2), "without_mortgage": ("2", 13)}


def owner_cost_burden(geography: str = "block group") -> ConstraintTable:
    """``B25091``: mortgage status by selected monthly owner costs as a
    percentage of household income, over owner-occupied housing units.

    "Not computed" is an owner with a blank ``OCPIP`` (no household income).
    """
    categories = []
    for status, (ten, first) in MORTGAGE_STATUS.items():
        within = (lambda code: lambda u: u["TEN"] == code)(ten)
        categories.extend(_banded(
            "B25091", BURDEN_BANDS, first + 1,
            lambda u: _num(u, "OCPIP"), within=within, prefix=f"{status}_burden_",
        ))
        categories.append(Category(
            name=f"{status}_not_computed",
            select=(lambda w: lambda u: w(u) & u["OCPIP"].isna())(within),
            published=(_cell("B25091", first + 10),),
        ))
    return ConstraintTable(
        table="B25091",
        variables=("TEN", "OCPIP"),
        universe="household",
        geography=geography,
        categories=tuple(categories),
        waived=("B25091_001", "B25091_002", "B25091_013"),
    )


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------

#: ``JWMNP`` code for "Suppressed for data year 2023 for select PUMAs".
#: Such a commuter falls in no time band -- missing, not a category -- and
#: there are none in Tennessee 2020-2024.
JWMNP_SUPPRESSED = 888


def _travel_time(persons: pd.DataFrame) -> pd.Series:
    minutes = _num(persons, "JWMNP")
    return minutes.mask(minutes == JWMNP_SUPPRESSED)


#: ``B08303``'s bands in minutes, ``[lower, upper)``, from cell ``_002``.
TRAVEL_TIME_EDGES = (5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 90)


def travel_time(geography: str = "block group") -> ConstraintTable:
    """``B08303``: travel time to work, over workers 16 and over who did not
    work from home. ``JWMNP`` minutes, 1-200 (top-coded)."""
    categories = _banded(
            "B08303", _edges_to_bands(TRAVEL_TIME_EDGES), 2,
        _travel_time, within=commuters, prefix="minutes_",
    )
    return ConstraintTable(
        table="B08303",
        variables=("JWTRNS", "JWMNP"),
        universe="person",
        geography=geography,
        categories=tuple(categories),
        waived=("B08303_001",),
    )


#: ``B08134``'s time bands in minutes, from each block's first cell.
MODE_TIME_EDGES = (10, 15, 20, 25, 30, 35, 45, 60)

#: ``B08134``'s finest mode blocks: name, selector and first cell ``ORDER``.
#: Car, truck or van (``JWTRNS`` 01) splits by ``JWRIP`` occupancy: 1 is
#: "Drove alone", 2 the 2-person carpool, 3-10 the "3-or-more-person" one.
#: Subway (03) and light rail (05) share a block, as do long-distance or
#: commuter rail (04) and ferry (06); taxi (07), motorcycle (08), bicycle (09)
#: and other (12) share the last. Worked from home (11) is outside the
#: universe.
B08134_MODES = {
    "drove_alone": (lambda p: (p["JWTRNS"] == "01") & (_num(p, "JWRIP") == 1), 22),
    "carpool_2": (lambda p: (p["JWTRNS"] == "01") & (_num(p, "JWRIP") == 2), 42),
    "carpool_3_plus": (lambda p: (p["JWTRNS"] == "01") & (_num(p, "JWRIP") >= 3), 52),
    "bus": (lambda p: p["JWTRNS"] == "02", 72),
    "subway_light_rail": (lambda p: p["JWTRNS"].isin(("03", "05")), 82),
    "long_distance_rail_ferry": (lambda p: p["JWTRNS"].isin(("04", "06")), 92),
    "walked": (lambda p: p["JWTRNS"] == "10", 102),
    "taxi_motorcycle_bicycle_other": (
        lambda p: p["JWTRNS"].isin(("07", "08", "09", "12")), 112,
    ),
}

#: ``B08134`` lines that are margins of the cross-classification: time totals
#: across modes (``_002``-``_010``), and the time lines of the car, carpool and
#: public-transport subtotals, each the sum of finer blocks claimed here. They
#: are waived as subtotals although their titles do not end in a colon.
B08134_MARGINS = (
    *range(2, 11), *range(12, 21), *range(32, 41), *range(62, 71),
)


def mode_by_travel_time(geography: str = "block group") -> ConstraintTable:
    """``B08134``: means of transportation by travel time to work, over
    workers 16 and over who did not work from home. Finest blocks
    (:data:`B08134_MODES`) by nine time bands."""
    categories = []
    claimed = set()
    for mode, (within, first) in B08134_MODES.items():
        categories.extend(_banded(
            "B08134", _edges_to_bands(MODE_TIME_EDGES), first,
            _travel_time, within=within, prefix=f"{mode}_",
        ))
        claimed.update(range(first, first + 9))
    waived = tuple(_cell("B08134", o) for o in range(1, 121) if o not in claimed)
    return ConstraintTable(
        table="B08134",
        variables=("JWTRNS", "JWRIP", "JWMNP"),
        universe="person",
        geography=geography,
        categories=tuple(categories),
        waived=waived,
    )


# --------------------------------------------------------------------------
# Household composition and life stage
# --------------------------------------------------------------------------

#: ``HHT`` codes per ``B11001`` leaf. Nonfamily households split by living
#: alone, whatever the householder's sex: 4 and 6 alone, 5 and 7 not.
B11001_TYPES = {
    "married_couple": (("1",), 3),
    "other_family_male": (("2",), 5),
    "other_family_female": (("3",), 6),
    "nonfamily_alone": (("4", "6"), 8),
    "nonfamily_not_alone": (("5", "7"), 9),
}

#: Family and nonfamily ``HHT`` codes, and the householder's sex for
#: nonfamily households.
HHT_FAMILY_CODES = ("1", "2", "3")
HHT_NONFAMILY_CODES = ("4", "5", "6", "7")
HHT_NONFAMILY_BY_SEX = {"male": ("4", "5"), "female": ("6", "7")}


def household_type(geography: str = "block group") -> ConstraintTable:
    """``B11001``: household type (including living alone), over households.

    ``HHT`` is blank exactly for group quarters among the units carried.
    """
    return ConstraintTable(
        table="B11001",
        variables=("HHT",),
        universe="household",
        geography=geography,
        categories=tuple(
            Category(
                name=label,
                select=(lambda c: lambda u: u["HHT"].isin(c))(codes),
                published=(_cell("B11001", order),),
            )
            for label, (codes, order) in B11001_TYPES.items()
        ),
        waived=("B11001_001", "B11001_002", "B11001_004", "B11001_007"),
    )


def household_type_by_size(geography: str = "block group") -> ConstraintTable:
    """``B11016``: family or nonfamily household by size (``NP``), over
    households. A family has at least two people, so its bands start at 2."""
    family = lambda u: u["HHT"].isin(HHT_FAMILY_CODES)  # noqa: E731
    nonfamily = lambda u: u["HHT"].isin(HHT_NONFAMILY_CODES)  # noqa: E731
    size = lambda u: _num(u, "NP")  # noqa: E731
    categories = _banded(
        "B11016",
        tuple((n, n + 1) for n in range(2, 7)) + ((7, INF),), 3,
        size, within=family, prefix="family_size_",
    ) + _banded(
        "B11016",
        tuple((n, n + 1) for n in range(1, 7)) + ((7, INF),), 10,
        size, within=nonfamily, prefix="nonfamily_size_",
    )
    return ConstraintTable(
        table="B11016",
        variables=("HHT", "NP"),
        universe="household",
        geography=geography,
        categories=tuple(categories),
        waived=("B11016_001", "B11016_002", "B11016_009"),
    )


#: ``B11005`` household types per presence block, with offsets from the
#: block's subtotal (``_002`` with people under 18, ``_011`` without).
B11005_TYPES = {
    "married_couple": (("1",), 2),
    "other_family_male": (("2",), 4),
    "other_family_female": (("3",), 5),
    "nonfamily_male": (HHT_NONFAMILY_BY_SEX["male"], 7),
    "nonfamily_female": (HHT_NONFAMILY_BY_SEX["female"], 8),
}

#: ``R18`` code per ``B11005`` presence block and its subtotal ``ORDER``.
B11005_PRESENCE = {"with_under_18": ("1", 2), "no_under_18": ("0", 11)}


def households_by_presence_of_children(geography: str = "block group") -> ConstraintTable:
    """``B11005``: households by presence of people under 18, by household
    type. Presence is ``R18``; type is ``HHT``, with a nonfamily household
    classed by the householder's sex as ``HHT`` codes it."""
    categories = []
    for presence, (r18, first) in B11005_PRESENCE.items():
        for label, (codes, offset) in B11005_TYPES.items():

            def select(u, r18=r18, codes=codes):
                return (u["R18"] == r18) & u["HHT"].isin(codes)

            categories.append(Category(
                name=f"{presence}_{label}", select=select,
                published=(_cell("B11005", first + offset),),
            ))
    waived = ("B11005_001", "B11005_002", "B11005_003", "B11005_005", "B11005_008",
              "B11005_011", "B11005_012", "B11005_014", "B11005_017")
    return ConstraintTable(
        table="B11005",
        variables=("R18", "HHT"),
        universe="household",
        geography=geography,
        categories=tuple(categories),
        waived=waived,
    )


#: ``B25007``'s householder age bands, ``[lower, upper)``.
HOUSEHOLDER_AGE_BANDS = (
    (15, 25), (25, 35), (35, 45), (45, 55), (55, 60), (60, 65), (65, 75),
    (75, 85), (85, INF),
)


def tenure_by_householder_age(geography: str = "block group") -> ConstraintTable:
    """``B25007``: tenure by age of householder (``HHLDRAGEP``), over
    occupied housing units."""
    def detail(tenure_label, first):
        return _banded(
            "B25007", HOUSEHOLDER_AGE_BANDS, first + 1,
            lambda u: _num(u, "HHLDRAGEP"), within=_tenure_mask(tenure_label),
            prefix=f"{tenure_label}_householder_",
        )

    return _by_tenure("B25007", ("HHLDRAGEP",), detail, geography)


def tenure_by_household_size(geography: str = "block group") -> ConstraintTable:
    """``B25009``: tenure by household size (``NP``), over occupied units."""
    bands = tuple((n, n + 1) for n in range(1, 7)) + ((7, INF),)

    def detail(tenure_label, first):
        return _banded(
            "B25009", bands, first + 1,
            lambda u: _num(u, "NP"), within=_tenure_mask(tenure_label),
            prefix=f"{tenure_label}_size_",
        )

    return _by_tenure("B25009", ("NP",), detail, geography)


#: ``B19037``'s householder age groups ``[lower, upper)`` and subtotal ORDER.
B19037_AGE_GROUPS = (((-INF, 25), 2), ((25, 45), 19), ((45, 65), 36), ((65, INF), 53))


def householder_age_by_income(geography: str = "block group") -> ConstraintTable:
    """``B19037``: age of householder by household income, over households.

    The 16 income brackets of each age group are ``B19001``'s, in the same
    order (:data:`~pmedm_vb.assemble.constraints.B19001_BRACKETS`), applied to
    income in reference-year dollars.
    """
    categories = []
    for (age_low, age_high), first in B19037_AGE_GROUPS:
        for offset, (_, low, high) in enumerate(B19001_BRACKETS, start=1):
            low = -INF if low is None else low
            high = INF if high is None else high

            def select(u, a0=age_low, a1=age_high, i0=low, i1=high):
                age = _num(u, "HHLDRAGEP")
                income = adjusted_household_income(u)
                return (age >= a0) & (age < a1) & (income >= i0) & (income < i1)

            categories.append(Category(
                name=f"householder_{_band_name(age_low, age_high)}"
                     f"_income_{_band_name(low, high)}",
                select=select,
                published=(_cell("B19037", first + offset),),
            ))
    return ConstraintTable(
        table="B19037",
        variables=("HHLDRAGEP", "HINCP", "ADJINC"),
        universe="household",
        geography=geography,
        categories=tuple(categories),
        waived=("B19037_001", *(_cell("B19037", f) for _, f in B19037_AGE_GROUPS)),
    )


#: ``RELSHIPP`` codes per ``B09019`` leaf below the householder. Roommate (34)
#: and other nonrelative (36) together are "Other nonrelatives"; the two
#: group-quarters codes (37 institutional, 38 noninstitutional) together are
#: "In group quarters".
B09019_RELATIONSHIPS = {
    "opposite_sex_spouse": (("21",), 10),
    "same_sex_spouse": (("23",), 11),
    "opposite_sex_partner": (("22",), 12),
    "same_sex_partner": (("24",), 13),
    "biological_child": (("25",), 15),
    "adopted_child": (("26",), 16),
    "stepchild": (("27",), 17),
    "grandchild": (("30",), 18),
    "sibling": (("28",), 19),
    "parent": (("29",), 20),
    "parent_in_law": (("31",), 21),
    "child_in_law": (("32",), 22),
    "other_relative": (("33",), 23),
    "foster_child": (("35",), 24),
    "other_nonrelative": (("34", "36"), 25),
    "group_quarters": (("37", "38"), 26),
}

#: ``RELSHIPP`` code of the householder (reference person).
RELSHIPP_HOUSEHOLDER = "20"

#: ``B09019`` householder leaves: sex, living alone, cell ``ORDER``.
B09019_HOUSEHOLDERS = (
    ("male", True, 5), ("male", False, 6), ("female", True, 8), ("female", False, 9),
)


def relationship(geography: str = "tract") -> ConstraintTable:
    """``B09019``: household type (including living alone) by relationship,
    over the total population. Published at tract, not block group.

    A householder lives alone when the household has one person record. That
    is counted from the person frame itself (records per ``SERIALNO``), which
    holds every person of every unit, so no housing column is needed.
    """
    categories = []
    for sex, alone, order in B09019_HOUSEHOLDERS:

        def select(p, code=SEX_CODES[sex], alone=alone):
            size = p.groupby("SERIALNO")["SERIALNO"].transform("size")
            head = (p["RELSHIPP"] == RELSHIPP_HOUSEHOLDER) & (p["SEX"] == code)
            return head & ((size == 1) if alone else (size > 1))

        categories.append(Category(
            name=f"householder_{sex}_{'alone' if alone else 'not_alone'}",
            select=select, published=(_cell("B09019", order),),
        ))
    for label, (codes, order) in B09019_RELATIONSHIPS.items():
        categories.append(Category(
            name=label,
            select=(lambda c: lambda p: p["RELSHIPP"].isin(c))(codes),
            published=(_cell("B09019", order),),
        ))
    return ConstraintTable(
        table="B09019",
        variables=("RELSHIPP", "SEX"),
        universe="person",
        geography=geography,
        categories=tuple(categories),
        waived=("B09019_001", "B09019_002", "B09019_003", "B09019_004",
                "B09019_007", "B09019_014"),
    )


# --------------------------------------------------------------------------
# Socio-economic
# --------------------------------------------------------------------------

#: ``ESR`` codes per ``B23025`` leaf.
B23025_STATUS = {
    "civilian_employed": (("1", "2"), 4),
    "unemployed": (("3",), 5),
    "armed_forces": (ESR_ARMED_FORCES, 6),
    "not_in_labor_force": (("6",), 7),
}


def employment(geography: str = "block group") -> ConstraintTable:
    """``B23025``: employment status, over the population 16 and over.
    ``ESR`` is blank exactly under 16."""
    return ConstraintTable(
        table="B23025",
        variables=("ESR",),
        universe="person",
        geography=geography,
        categories=tuple(
            Category(
                name=label,
                select=(lambda c: lambda p: p["ESR"].isin(c))(codes),
                published=(_cell("B23025", order),),
            )
            for label, (codes, order) in B23025_STATUS.items()
        ),
        waived=("B23025_001", "B23025_002", "B23025_003"),
    )


#: The six coverage types that count as insurance, as ``HINS`` columns:
#: employer (1), direct purchase (2), Medicare (3), Medicaid/means-tested (4),
#: TRICARE (5), VA (6). Indian Health Service (``HINS7``) is not coverage: the
#: 2024 Subject Definitions count people "whose only health coverage was
#: Indian Health Service" as uninsured, and ``HICOV`` 2 equals "no yes among
#: HINS1-6" on every Tennessee 2020-2024 record.
COVERAGE = {
    "employer": "HINS1", "direct": "HINS2", "medicare": "HINS3",
    "medicaid": "HINS4", "tricare": "HINS5", "va": "HINS6",
}

#: Private and public types, from the same definitions: private is employer,
#: direct purchase and TRICARE ("parts a, b, or e"); public is Medicare,
#: Medicaid and VA ("c, d, or f").
PRIVATE = ("employer", "direct", "tricare")
PUBLIC = ("medicare", "medicaid", "va")

#: ``B27010``'s age groups ``[lower, upper)``, each with its cells as offsets
#: from the group subtotal: one-type leaves, the named two-type pairs, the
#: three "other" combinations, and no coverage. The 65+ group publishes no
#: "Medicaid only" line and the under-35 groups no "direct-purchase and
#: Medicare" pair; a person in an unpublished one-type line would fall in no
#: category (none do in Tennessee 2020-2024), while an unpublished pair is
#: correctly absorbed by "other coverage combinations", being mixed private and
#: public.
B27010_GROUPS = {
    "under_19": ((-INF, 19), 2, {
        "only": {"employer": 2, "direct": 3, "medicare": 4, "medicaid": 5,
                 "tricare": 6, "va": 7},
        "pairs": {("employer", "direct"): 9, ("employer", "medicare"): 10,
                  ("medicare", "medicaid"): 11},
        "other": {"private": 12, "public": 13, "mixed": 14},
        "none": 15,
    }),
    "19_to_34": ((19, 35), 18, {
        "only": {"employer": 2, "direct": 3, "medicare": 4, "medicaid": 5,
                 "tricare": 6, "va": 7},
        "pairs": {("employer", "direct"): 9, ("employer", "medicare"): 10,
                  ("medicare", "medicaid"): 11},
        "other": {"private": 12, "public": 13, "mixed": 14},
        "none": 15,
    }),
    "35_to_64": ((35, 65), 34, {
        "only": {"employer": 2, "direct": 3, "medicare": 4, "medicaid": 5,
                 "tricare": 6, "va": 7},
        "pairs": {("employer", "direct"): 9, ("employer", "medicare"): 10,
                  ("direct", "medicare"): 11, ("medicare", "medicaid"): 12},
        "other": {"private": 13, "public": 14, "mixed": 15},
        "none": 16,
    }),
    "65_plus": ((65, INF), 51, {
        "only": {"employer": 2, "direct": 3, "medicare": 4, "tricare": 5, "va": 6},
        "pairs": {("employer", "direct"): 8, ("employer", "medicare"): 9,
                  ("direct", "medicare"): 10, ("medicare", "medicaid"): 11},
        "other": {"private": 12, "public": 13, "mixed": 14},
        "none": 15,
    }),
}


def _coverage(persons: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per coverage type: a ``"1"`` (Yes) in its ``HINS``."""
    return pd.DataFrame(
        {name: persons[column] == "1" for name, column in COVERAGE.items()},
        index=persons.index,
    )


def health_insurance(geography: str = "block group") -> ConstraintTable:
    """``B27010``: types of health insurance coverage by age, over the civilian
    noninstitutionalized population (:func:`civilian_noninstitutional`).

    A person's coverage is the set of types answered Yes (:data:`COVERAGE`).
    One type: that type's "only" line. Exactly two types forming a named pair:
    that pair's line. Otherwise, with two or more types: all private, "Other
    private only combinations"; all public, "Other public only combinations";
    both, "Other coverage combinations". No type: "No health insurance
    coverage". The named pairs are read as *exactly* those two types, since
    the lines partition "With two or more types".
    """
    categories, waived = [], ["B27010_001"]
    for group, ((low, high), first, cells) in B27010_GROUPS.items():
        waived += [_cell("B27010", first), _cell("B27010", first + 1),
                   _cell("B27010", first + 1 + len(cells["only"]) + 1)]

        def in_group(p, low=low, high=high):
            age = _num(p, "AGEP")
            return civilian_noninstitutional(p) & (age >= low) & (age < high)

        def only(p, kind, g=in_group):
            cover = _coverage(p)
            return g(p) & (cover.sum(axis=1) == 1) & cover[kind]

        def pair(p, kinds, g=in_group):
            cover = _coverage(p)
            return g(p) & (cover.sum(axis=1) == 2) & cover[list(kinds)].all(axis=1)

        def other(p, kind, pairs=tuple(cells["pairs"]), g=in_group):
            cover = _coverage(p)
            count = cover.sum(axis=1)
            named = pd.Series(False, index=p.index)
            for kinds in pairs:
                named |= (count == 2) & cover[list(kinds)].all(axis=1)
            private = cover[list(PRIVATE)].any(axis=1)
            public = cover[list(PUBLIC)].any(axis=1)
            mix = {"private": private & ~public, "public": public & ~private,
                   "mixed": private & public}[kind]
            return g(p) & (count >= 2) & ~named & mix

        for kind, offset in cells["only"].items():
            categories.append(Category(
                name=f"{group}_{kind}_only",
                select=(lambda k, f=only: lambda p: f(p, k))(kind),
                published=(_cell("B27010", first + offset),),
            ))
        for kinds, offset in cells["pairs"].items():
            categories.append(Category(
                name=f"{group}_{'_and_'.join(kinds)}",
                select=(lambda k, f=pair: lambda p: f(p, k))(kinds),
                published=(_cell("B27010", first + offset),),
            ))
        for kind, offset in cells["other"].items():
            categories.append(Category(
                name=f"{group}_other_{kind}_combinations",
                select=(lambda k, f=other: lambda p: f(p, k))(kind),
                published=(_cell("B27010", first + offset),),
            ))
        categories.append(Category(
            name=f"{group}_uninsured",
            select=lambda p, g=in_group: g(p) & (_coverage(p).sum(axis=1) == 0),
            published=(_cell("B27010", first + cells["none"]),),
        ))
    return ConstraintTable(
        table="B27010",
        variables=("AGEP", "RELSHIPP", "ESR", "MIL", *COVERAGE.values()),
        universe="person",
        geography=geography,
        categories=tuple(categories),
        waived=tuple(waived),
    )


#: ``SCHG`` codes per ``B14001`` enrolled level, with the cell ``ORDER``.
B14001_LEVELS = {
    "nursery_preschool": (("01",), 3),
    "kindergarten": (("02",), 4),
    "grade_1_to_4": (("03", "04", "05", "06"), 5),
    "grade_5_to_8": (("07", "08", "09", "10"), 6),
    "grade_9_to_12": (("11", "12", "13", "14"), 7),
    "college_undergraduate": (("15",), 8),
    "graduate_professional": (("16",), 9),
}

#: ``SCH`` codes: not enrolled, and enrolled (public, or private or home).
SCH_NOT_ENROLLED = "1"
SCH_ENROLLED = ("2", "3")


def school_enrollment(geography: str = "tract") -> ConstraintTable:
    """``B14001``: school enrollment by level, over the population 3 and over.
    Published at tract, not block group. ``SCH`` is blank exactly under 3, and
    ``SCHG`` is set for every enrolled person and no one else."""
    categories = [
        Category(
            name=f"enrolled_{label}",
            select=(lambda c: lambda p: p["SCH"].isin(SCH_ENROLLED) & p["SCHG"].isin(c))(codes),
            published=(_cell("B14001", order),),
        )
        for label, (codes, order) in B14001_LEVELS.items()
    ]
    categories.append(Category(
        name="not_enrolled", select=lambda p: p["SCH"] == SCH_NOT_ENROLLED,
        published=("B14001_010",),
    ))
    return ConstraintTable(
        table="B14001",
        variables=("SCH", "SCHG"),
        universe="person",
        geography=geography,
        categories=tuple(categories),
        waived=("B14001_001", "B14001_002"),
    )


#: ``B18101``'s age groups ``[lower, upper)``; each group's subtotal is three
#: cells after the previous one, its two leaves (with, without a disability)
#: the next two.
B18101_AGE_GROUPS = ((-INF, 5), (5, 18), (18, 35), (35, 65), (65, 75), (75, INF))

#: Cell ``ORDER`` of each sex's first age-group subtotal.
B18101_SEX_FIRST = {"male": 3, "female": 22}

#: ``DIS`` codes.
DIS_WITH = "1"
DIS_WITHOUT = "2"


def disability(geography: str = "tract") -> ConstraintTable:
    """``B18101``: sex by age by disability status, over the civilian
    noninstitutionalized population. Published at tract, not block group."""
    categories, waived = [], ["B18101_001", "B18101_002", "B18101_021"]
    for sex, first in B18101_SEX_FIRST.items():
        for index, (low, high) in enumerate(B18101_AGE_GROUPS):
            subtotal = first + 3 * index
            waived.append(_cell("B18101", subtotal))
            for offset, (label, code) in enumerate((("with", DIS_WITH), ("without", DIS_WITHOUT)), 1):

                def select(p, s=SEX_CODES[sex], low=low, high=high, code=code):
                    age = _num(p, "AGEP")
                    return (civilian_noninstitutional(p) & (p["SEX"] == s)
                            & (age >= low) & (age < high) & (p["DIS"] == code))

                categories.append(Category(
                    name=f"{sex}_{_band_name(low, high)}_{label}_disability",
                    select=select, published=(_cell("B18101", subtotal + offset),),
                ))
    return ConstraintTable(
        table="B18101",
        variables=("SEX", "AGEP", "DIS", "RELSHIPP", "ESR", "MIL"),
        universe="person",
        geography=geography,
        categories=tuple(categories),
        waived=tuple(waived),
    )


# --------------------------------------------------------------------------
# Language, nativity and mobility
# --------------------------------------------------------------------------

#: ``HHL`` code per ``C16002`` language, with the first cell of its pair
#: (limited English, then not limited); English only has a single cell.
C16002_LANGUAGES = {
    "spanish": ("2", 4),
    "other_indo_european": ("3", 7),
    "asian_pacific_island": ("4", 10),
    "other": ("5", 13),
}

#: ``LNGI`` codes: 2 is "No one in the household 14 and over speaks English
#: only or speaks English 'very well'" -- a limited English speaking household.
LNGI_LIMITED = "2"
LNGI_NOT_LIMITED = "1"


def household_language(geography: str = "block group") -> ConstraintTable:
    """``C16002``: household language by limited English speaking status, over
    households. ``HHL`` and ``LNGI`` are blank exactly for group quarters among
    the units carried, and every English-only household has ``LNGI`` 1."""
    categories = [Category(
        name="english_only", select=lambda u: u["HHL"] == "1", published=("C16002_002",),
    )]
    for label, (code, first) in C16002_LANGUAGES.items():
        for offset, (status, lngi) in enumerate(
            (("limited", LNGI_LIMITED), ("not_limited", LNGI_NOT_LIMITED))
        ):
            categories.append(Category(
                name=f"{label}_{status}",
                select=(lambda c, g: lambda u: (u["HHL"] == c) & (u["LNGI"] == g))(code, lngi),
                published=(_cell("C16002", first + offset),),
            ))
    return ConstraintTable(
        table="C16002",
        variables=("HHL", "LNGI"),
        universe="household",
        geography=geography,
        categories=tuple(categories),
        waived=("C16002_001", "C16002_003", "C16002_006", "C16002_009", "C16002_012"),
    )


#: Census region of each state FIPS code, from the Census Bureau's
#: ``state-geocodes-v2023.xlsx`` (Population Division, www2.census.gov), read
#: for this module. ``POBP`` codes a U.S. state as its FIPS code, zero-padded to
#: three characters.
STATE_REGION = {
    "northeast": ("09", "23", "25", "33", "34", "36", "42", "44", "50"),
    "midwest": ("17", "18", "19", "20", "26", "27", "29", "31", "38", "39", "46", "55"),
    "south": ("01", "05", "10", "11", "12", "13", "21", "22", "24", "28", "37",
              "40", "45", "47", "48", "51", "54"),
    "west": ("02", "04", "06", "08", "15", "16", "30", "32", "35", "41", "49", "53", "56"),
}

#: World area of birth by ``POBP`` code, as the data dictionary defines
#: ``WAOB``: Latin America 303 and 310-399, Asia 158-159, 161 and 200-299,
#: Europe 100-157, 160 and 162-199, Africa 400-499, Northern America 300-302
#: and 304-309, Oceania 500-554 "also includes POB=060 for 2018 or later".
#: ``WAOB`` itself is not used: on eight foreign-born Tennessee records born in
#: American Samoa (``POBP`` 060) the file carries ``WAOB`` 2 ("PR and US
#: Island Areas"), contradicting that definition and matching no foreign-born
#: line of ``B05002``; the ``POBP`` ranges place them in Oceania.
def _world_area(pobp: pd.Series) -> pd.Series:
    code = pd.to_numeric(pobp, errors="coerce")
    area = pd.Series(pd.NA, index=pobp.index, dtype="object")
    rules = (
        ("latin_america", (code == 303) | code.between(310, 399)),
        ("asia", code.between(158, 159) | (code == 161) | code.between(200, 299)),
        ("europe", code.between(100, 157) | (code == 160) | code.between(162, 199)),
        ("africa", code.between(400, 499)),
        ("northern_america", code.between(300, 302) | code.between(304, 309)),
        ("oceania", code.between(500, 554) | (code == 60)),
    )
    for name, mask in rules:
        area = area.mask(mask & area.isna(), name)
    return area


#: ``B05002``'s foreign-born world areas in published order.
FOREIGN_AREAS = ("europe", "asia", "africa", "oceania", "latin_america", "northern_america")

#: ``CIT`` codes. 1 born in the U.S.; 2 born in Puerto Rico, Guam, the U.S.
#: Virgin Islands or the Northern Marianas; 3 born abroad of U.S. citizen
#: parents; 4 naturalized; 5 not a citizen.
CIT_FOREIGN = {"naturalized": ("4", 15), "not_citizen": ("5", 22)}

#: ``POBP`` codes of the U.S. Island Areas: American Samoa, Guam, the Northern
#: Mariana Islands, the U.S. Virgin Islands. Puerto Rico (072) has its own line.
POBP_ISLAND_AREAS = ("060", "066", "069", "078")
POBP_PUERTO_RICO = "072"


def place_of_birth(geography: str = "tract") -> ConstraintTable:
    """``B05002``: place of birth by nativity and citizenship, over the total
    population. Published at tract, not block group.

    "Born in state of residence" compares ``POBP`` with the record's own
    ``STATE`` (zero-padded to three), so the builder does not depend on the
    study area.
    """
    def born_in_state(p):
        return (p["CIT"] == "1") & (p["POBP"] == p["STATE"].str.zfill(3))

    categories = [Category(name="native_born_in_state", select=born_in_state,
                           published=("B05002_003",))]
    for offset, (region, states) in enumerate(STATE_REGION.items()):
        codes = tuple(s.zfill(3) for s in states)
        categories.append(Category(
            name=f"native_born_other_state_{region}",
            select=(lambda c: lambda p: (p["CIT"] == "1") & p["POBP"].isin(c)
                    & ~born_in_state(p))(codes),
            published=(_cell("B05002", 5 + offset),),
        ))
    categories += [
        Category(name="native_born_puerto_rico",
                 select=lambda p: (p["CIT"] == "2") & (p["POBP"] == POBP_PUERTO_RICO),
                 published=("B05002_010",)),
        Category(name="native_born_island_areas",
                 select=lambda p: (p["CIT"] == "2") & p["POBP"].isin(POBP_ISLAND_AREAS),
                 published=("B05002_011",)),
        Category(name="native_born_abroad_of_american_parents",
                 select=lambda p: p["CIT"] == "3", published=("B05002_012",)),
    ]
    for status, (cit, first) in CIT_FOREIGN.items():
        for offset, area_name in enumerate(FOREIGN_AREAS):
            categories.append(Category(
                name=f"foreign_{status}_{area_name}",
                select=(lambda c, a: lambda p: (p["CIT"] == c)
                        & (_world_area(p["POBP"]) == a).fillna(False))(cit, area_name),
                published=(_cell("B05002", first + offset),),
            ))
    return ConstraintTable(
        table="B05002",
        variables=("CIT", "POBP"),
        universe="person",
        geography=geography,
        categories=tuple(categories),
        waived=("B05002_001", "B05002_002", "B05002_004", "B05002_009",
                "B05002_013", "B05002_014", "B05002_021"),
    )


#: ``MIG`` codes: 1 same house; 2 outside the United States and Puerto Rico;
#: 3 a different house in the United States or Puerto Rico. ``MIGSP`` 072 is
#: Puerto Rico, which the 2024 Subject Definitions class as "Abroad" in tables
#: for the population with current residence in the United States.
MIG_SAME_HOUSE = "1"
MIG_OUTSIDE_US_PR = "2"
MIG_DIFFERENT_HOUSE = "3"
MIGSP_PUERTO_RICO = "072"

#: ``B07003`` cells by mobility and sex. Within the United States the table
#: splits same county, different county and different state; telling those
#: apart needs the previous residence's county (``MIGPUMA``), which is
#: geography and is excluded, so the three published lines are claimed
#: together -- an exact collapse, since they are the whole of "moved within
#: the United States". A mover whose ``MIGSP`` is 888 ("Suppressed for data
#: year 2022 for select cases"; five such records in Tennessee) cannot be told
#: from a Puerto Rico mover and is counted as moving within the United States.
B07003_CELLS = {
    "same_house": {"male": (5,), "female": (6,)},
    "moved_within_us": {"male": (8, 11, 14), "female": (9, 12, 15)},
    "moved_from_abroad": {"male": (17,), "female": (18,)},
}

#: ``B07003``'s sex totals (``_002``, ``_003``) are margins of the
#: cross-classification, waived although their titles end without a colon.
B07003_MARGINS = (2, 3)


def mobility(geography: str = "tract") -> ConstraintTable:
    """``B07003``: residence 1 year ago by sex, over the population 1 and over.
    Published at tract, not block group. Used in place of ``B07201``, whose
    universe is restricted to metropolitan residents (module notes).
    ``MIG`` is blank exactly under 1 year old."""
    def abroad(p):
        return (p["MIG"] == MIG_OUTSIDE_US_PR) | (
            (p["MIG"] == MIG_DIFFERENT_HOUSE) & (p["MIGSP"] == MIGSP_PUERTO_RICO))

    status = {
        "same_house": lambda p: p["MIG"] == MIG_SAME_HOUSE,
        "moved_within_us": lambda p: (p["MIG"] == MIG_DIFFERENT_HOUSE) & ~abroad(p),
        "moved_from_abroad": abroad,
    }
    categories, claimed = [], []
    for label, by_sex in B07003_CELLS.items():
        for sex, orders in by_sex.items():
            cells = tuple(_cell("B07003", o) for o in orders)
            claimed += cells
            categories.append(Category(
                name=f"{label}_{sex}",
                select=(lambda f, s: lambda p: f(p) & (p["SEX"] == s))(status[label], SEX_CODES[sex]),
                published=cells,
            ))
    waived = tuple(_cell("B07003", o) for o in range(1, 19) if _cell("B07003", o) not in claimed)
    return ConstraintTable(
        table="B07003",
        variables=("MIG", "MIGSP", "SEX"),
        universe="person",
        geography=geography,
        categories=tuple(categories),
        waived=waived,
    )


# --------------------------------------------------------------------------
# Registry and validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TableSpec:
    """An added table: its builder, the lowest geography it is published at
    (2024 5-year Data Product List), and its universe as a selector on the
    frame its universe names (``persons`` or ``units``)."""

    build: Callable[[], ConstraintTable]
    published_at: str
    universe: Universe
    universe_title: str
    #: Cells that are margins of a cross-classification -- sums of claimed
    #: cells whose titles do not mark them as subtotals -- and may be waived.
    margins: tuple[int, ...] = ()


#: The added tables, by ID.
TABLES: dict[str, TableSpec] = {
    "B25032": TableSpec(tenure_by_units_in_structure, "block group", occupied,
                        "Occupied housing units"),
    "B25036": TableSpec(tenure_by_year_built, "block group", occupied,
                        "Occupied housing units"),
    "B25020": TableSpec(tenure_by_rooms, "block group", occupied, "Occupied housing units"),
    "B25042": TableSpec(tenure_by_bedrooms, "block group", occupied, "Occupied housing units"),
    "B25014": TableSpec(tenure_by_occupants_per_room, "block group", occupied,
                        "Occupied housing units"),
    "B25075": TableSpec(value, "block group", owner, "Owner-occupied housing units"),
    "B25063": TableSpec(gross_rent, "block group", renter, "Renter-occupied housing units"),
    "B08303": TableSpec(travel_time, "block group", commuters,
                        "Workers 16 years and over who did not work from home"),
    "B08134": TableSpec(mode_by_travel_time, "block group", commuters,
                        "Workers 16 years and over who did not work from home",
                        margins=B08134_MARGINS),
    "B11001": TableSpec(household_type, "block group", occupied, "Households"),
    "B11016": TableSpec(household_type_by_size, "block group", occupied, "Households"),
    "B11005": TableSpec(households_by_presence_of_children, "block group", occupied,
                        "Households"),
    "B25007": TableSpec(tenure_by_householder_age, "block group", occupied,
                        "Occupied housing units"),
    "B25009": TableSpec(tenure_by_household_size, "block group", occupied,
                        "Occupied housing units"),
    "B19037": TableSpec(householder_age_by_income, "block group", occupied, "Households"),
    "B09019": TableSpec(relationship, "tract", everyone, "Total population"),
    "B23025": TableSpec(employment, "block group", aged(16), "Population 16 years and over"),
    "B25070": TableSpec(rent_burden, "block group", renter, "Renter-occupied housing units"),
    "B25091": TableSpec(owner_cost_burden, "block group", owner,
                        "Owner-occupied housing units"),
    "B27010": TableSpec(health_insurance, "block group", civilian_noninstitutional,
                        "Civilian noninstitutionalized population"),
    "B14001": TableSpec(school_enrollment, "tract", aged(3), "Population 3 years and over"),
    "B18101": TableSpec(disability, "tract", civilian_noninstitutional,
                        "Civilian noninstitutionalized population"),
    "C16002": TableSpec(household_language, "block group", occupied, "Households"),
    "B05002": TableSpec(place_of_birth, "tract", everyone, "Total population"),
    "B07003": TableSpec(mobility, "tract", aged(1), "Population 1 year and over",
                        margins=B07003_MARGINS),
}


def build_table(table: str) -> ConstraintTable:
    """Build one added table at the lowest geography it is published at."""
    spec = TABLES[table]
    return spec.build(geography=spec.published_at)


def validate_against_api(area: StudyArea, table: ConstraintTable) -> pd.DataFrame:
    """Check an added table's cells against the API's published cell list.

    The checks of :meth:`ConstraintTable.validate`, which reads VRE labels and
    so cannot see these tables, plus one more:

    * every category has published cells;
    * every claimed cell exists in this vintage;
    * no cell is claimed twice;
    * every published cell is claimed or waived;
    * **every waived cell is a total or subtotal**: its title has children (a
      longer title it prefixes), or it is a declared margin of the
      cross-classification (:attr:`TableSpec.margins`). Waiving a leaf would
      silently drop part of the universe.

    Returns
    -------
    pandas.DataFrame
        The published cells with their titles and the category claiming each
        (``waived`` for waived ones), for reading the correspondence by eye.
    """
    labels = api_cell_labels(area, table.table)
    available = set(labels.index)
    if table.unresolved:
        raise ValueError(f"{table.table}: categories with no published cells: "
                         f"{list(table.unresolved)}")
    claimed: dict[str, str] = {}
    for category in table.categories:
        for cell in category.published:
            if cell not in available:
                raise ValueError(f"{table.table}: {category.name!r} claims {cell!r}, "
                                 f"which is not published for {area.year}")
            if cell in claimed:
                raise ValueError(f"{table.table}: {cell!r} is claimed by both "
                                 f"{claimed[cell]!r} and {category.name!r}")
            claimed[cell] = category.name
    waived = set(table.waived)
    unknown = waived - available
    if unknown:
        raise ValueError(f"{table.table}: waives unpublished cells {sorted(unknown)}")
    both = waived & set(claimed)
    if both:
        raise ValueError(f"{table.table}: cells both claimed and waived {sorted(both)}")
    dropped = available - set(claimed) - waived
    if dropped:
        raise ValueError(f"{table.table}: {len(dropped)} published cell(s) neither "
                         f"claimed nor waived: {sorted(dropped)}")

    spec = TABLES.get(table.table)
    margins = {_cell(table.table, o) for o in (spec.margins if spec else ())}
    # Colons mark subtotals inside API titles ("Total: | Car, truck, or van: |
    # ..."), so they are dropped before comparing a title with its children.
    titles = labels.str.replace(":", "", regex=False)
    leaves_waived = []
    for cell in sorted(waived):
        title = titles[cell]
        has_children = titles.drop(cell).str.startswith(f"{title} | ").any()
        if not (has_children or cell in margins):
            leaves_waived.append(f"{cell} ({labels[cell]})")
    if leaves_waived:
        raise ValueError(f"{table.table}: waives cells that are neither totals, "
                         f"subtotals nor declared margins: {leaves_waived}")

    return pd.DataFrame({
        "title": labels,
        "category": pd.Series(claimed).reindex(labels.index).fillna("waived"),
    })


def coverage_counts(
    frame: pd.DataFrame, table: ConstraintTable, universe: Universe
) -> pd.DataFrame:
    """How many records of ``frame`` each category claims, and how many in the
    universe fall in none or several -- the partition check.

    Returns one row per record class: ``in_universe``, ``none`` (in the
    universe, no category), ``several`` (in two or more), ``outside`` (outside
    the universe but claimed by a category).
    """
    hits = np.column_stack([
        category.select(frame).fillna(False).to_numpy(dtype=bool)
        for category in table.categories
    ])
    inside = universe(frame).fillna(False).to_numpy(dtype=bool)
    count = hits.sum(axis=1)
    return pd.Series({
        "in_universe": int(inside.sum()),
        "none": int((inside & (count == 0)).sum()),
        "several": int((inside & (count > 1)).sum()),
        "outside": int((~inside & (count > 0)).sum()),
    })
