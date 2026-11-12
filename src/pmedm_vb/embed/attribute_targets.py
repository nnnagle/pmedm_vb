"""Stage 1 targets as PUMS attributes, each appearing once (configuration v2).

Published tables cross a few attributes many times over -- tenure in about ten
of the configured tables, income in six -- so with tables as targets
(``stage1_v1``) a shared attribute is weighted by how often it is published,
and holding out one table leaks it through the others. Here each attribute is
one target, and the published tables become a layer used for prefetch, scoring
and stage 2, no longer targets.

**Categories: the common refinement.** Each attribute's categories are the
common refinement of the breakpoints that the configured tables use for it, so
that every table's categories for that attribute are unions of the attribute's
categories. The breakpoints are *derived* from the constants of the existing
builders (``assemble/constraints.py`` and ``embed/tables.py``), never written
out again: age, for instance, takes ``B01001``'s published bands plus the
breaks of ``B17024``, ``B27010``, ``B18101`` and ``B23001``'s collapse, which
adds 6, 12 and 19. A universe boundary that coincides with an attribute's own
universe (``ESR`` blank under 16, ``SCH`` under 3, ``MIG`` under 1, ``MSP``
under 15) needs no age break: the attribute's blank already separates it.
:func:`refinement_violations` checks the result against the PUMS, table by
table, rather than assuming it.

**Person attributes** are member-count multinomials, as person tables were:
each unit counts its members in each category of the attribute's universe.
Age by sex stays one within-person cross (``B01001``'s structure). Other
within-person crosses that tables publish (sex by occupation, age by
disability...) are *not* targets; the positions are left to learn them.

**Household attributes** are one categorical per occupied housing unit; group
quarters are outside every household universe, and their type is carried by
the fixed indicators of φ, as before.

**Dependencies.** Some attributes are computed from others (household size
from the members, poverty ratio from income and the family), and some have a
universe that reveals another attribute (a value is reported only by owners).
:data:`ATTRIBUTE_TARGETS` records each attribute's ``derived_from``, with the
kind of dependence; :func:`mask_closure` turns that into the set of targets
masked together in check 1 (:mod:`pmedm_vb.embed.fit`): masking a primary
attribute masks every attribute depending on it, masking a dependent masks
only itself.

Codes are those the builders already read from the 2020-2024 data dictionary;
the few used here for the first time (``HHT`` 1-7, ``TEN`` 1-4, ``JWRIP``,
``VEH``) were read from the same dictionary for this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from pmedm_vb.assemble.constraints import (
    B01001_BANDS,
    B12001_STATUS,
    B15002_ATTAINMENT,
    B17024_AGE_GROUPS,
    B17024_RATIO_COLLAPSE,
    B19001_BRACKETS,
    B25044_VEHICLES,
    C17002_BANDS,
    C24030_CELL_NAMES,
    ESR_CIVILIAN_EMPLOYED,
    HFL_TO_CELL,
    NOT_HISPANIC,
    OCCP_TO_MAJOR_GROUP,
    RAC1P_TO_RACE,
    SEX_CODES,
    Category,
    ConstraintTable,
    adjusted_household_income,
    employment_status,
    household_income,
    industry_crosswalk,
    occupation_crosswalk,
    race_ethnicity,
)
from pmedm_vb.config import StudyArea
from pmedm_vb.embed import tables as T
from pmedm_vb.embed.attributes import attribute_table

INF = float("inf")


# --------------------------------------------------------------------------
# Breakpoints derived from the builders
# --------------------------------------------------------------------------


def _finite(values) -> set[float]:
    return {float(v) for v in values if v is not None and np.isfinite(v)}


#: Person age edges: ``B01001``'s published bands, refined by the age breaks of
#: every other configured person table that crosses age.
AGE_EDGES = tuple(sorted((
    _finite(upper for _, upper in B01001_BANDS)
    | _finite(low for _, low, _ in B17024_AGE_GROUPS)
    | _finite(low for (low, _), _, _ in T.B27010_GROUPS.values())
    | _finite(low for low, _ in T.B18101_AGE_GROUPS)
    | _finite(employment_status.__defaults__[0])
) - {0.0}))

#: Householder age edges: ``B25007``'s bands refined by ``B19037``'s groups.
#: The lowest, 15, is the universe floor (``HHLDRAGEP`` starts at 15), not a
#: break, so it is dropped.
HOUSEHOLDER_AGE_EDGES = tuple(sorted(
    _finite(low for low, _ in T.HOUSEHOLDER_AGE_BANDS)
    | _finite(low for (low, _), _ in T.B19037_AGE_GROUPS)
))[1:]

#: Household income edges: ``B19001``'s 16 brackets (used whole by ``B19037``)
#: refined by the constraint set's collapse of ``B19001``.
INCOME_EDGES = tuple(sorted(
    _finite(high for _, _, high in B19001_BRACKETS)
    | _finite(household_income.__defaults__[0])
))

#: Income-to-poverty edges (``POVPIP`` percent): ``C17002``'s bands, the
#: constraint set's collapse of ``B17024``, and ``B17010``'s poverty line.
POVERTY_EDGES = tuple(sorted((
    _finite(low for _, low, _ in C17002_BANDS)
    | _finite(high for _, _, high in C17002_BANDS)
    | _finite(v for pair in B17024_RATIO_COLLAPSE.values() for v in pair)
    | {100.0}
) - {0.0}))

#: Travel-time edges: ``B08303``'s bands refined by ``B08134``'s.
TRAVEL_TIME_EDGES = tuple(sorted(_finite(T.TRAVEL_TIME_EDGES) | _finite(T.MODE_TIME_EDGES)))

#: Household-size edges: ``B11016``'s and ``B25009``'s bands, 1 to 6 then 7+.
HOUSEHOLD_SIZE_EDGES = (2, 3, 4, 5, 6, 7)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _bands(edges: Sequence[float], value, *, within=None, prefix: str = "") -> list[Category]:
    """``[lower, upper)`` bands over ``edges``, open at both ends; no cells."""
    full = (-INF, *edges, INF)
    out = []
    for lower, upper in zip(full, full[1:]):

        def select(frame, lower=lower, upper=upper):
            v = value(frame)
            mask = (v >= lower) & (v < upper)
            return mask & within(frame) if within is not None else mask

        out.append(Category(name=f"{prefix}{T._band_name(lower, upper)}", select=select))
    return out


def _codes(variable: str, mapping: dict[str, tuple[str, ...]], *, within=None) -> list[Category]:
    out = []
    for name, codes in mapping.items():

        def select(frame, codes=codes):
            mask = frame[variable].isin(codes)
            return mask & within(frame) if within is not None else mask

        out.append(Category(name=name, select=select))
    return out


def _table(name: str, universe: str, variables, categories, householder=()) -> ConstraintTable:
    return ConstraintTable(
        table=name, universe=universe, geography="none", variables=tuple(variables),
        categories=tuple(categories), householder=tuple(householder),
    )


def _strip(table: ConstraintTable, name: str, *, drop=()) -> ConstraintTable:
    """A published table's categories as an attribute: no cells, no waivers."""
    categories = tuple(replace(c, published=()) for c in table.categories if c.name not in drop)
    return replace(table, table=name, geography="none", categories=categories, waived=())


def _age(frame):
    return T._num(frame, "AGEP")


# --------------------------------------------------------------------------
# Person attributes
# --------------------------------------------------------------------------


def age_sex(area: StudyArea) -> ConstraintTable:
    """Sex by age over everyone, on :data:`AGE_EDGES`: 26 bands per sex."""
    categories = []
    for sex, code in SEX_CODES.items():
        within = (lambda c: lambda p: p["SEX"] == c)(code)
        categories += _bands(AGE_EDGES, _age, within=within, prefix=f"{sex}_")
    return _table("age_sex", "person", ("AGEP", "SEX"), categories)


def race_ethnicity_attr(area: StudyArea) -> ConstraintTable:
    """``B03002``'s eight categories (Hispanic, and non-Hispanic by race)."""
    return _strip(race_ethnicity(), "race_ethnicity")


def relationship(area: StudyArea) -> ConstraintTable:
    """Relationship to the householder, over persons in households.

    ``B09019``'s leaves below the householder, with the householder as one
    category (its sex and living alone are in ``age_sex`` and
    ``household_size``). The group-quarters line is left out: whether a person
    lives in group quarters is the unit type, carried by φ's indicators.
    """
    mapping = {"householder": (T.RELSHIPP_HOUSEHOLDER,)}
    mapping.update({k: codes for k, (codes, _) in T.B09019_RELATIONSHIPS.items()
                    if k != "group_quarters"})
    return _table("relationship", "person", ("RELSHIPP",), _codes("RELSHIPP", mapping))


def marital_status(area: StudyArea) -> ConstraintTable:
    """``MSP``, ``B12001``'s six statuses, over persons 15 and over (blank
    below)."""
    mapping = {k: (code,) for k, (code, _) in B12001_STATUS.items()}
    return _table("marital_status", "person", ("MSP",), _codes("MSP", mapping))


def educational_attainment(area: StudyArea) -> ConstraintTable:
    """``SCHL`` grouped as ``B15002``, over persons 25 and over -- the only
    configured table using attainment, so its universe."""
    mapping = {k: codes for k, (codes, _) in B15002_ATTAINMENT.items()}
    return _table("educational_attainment", "person", ("AGEP", "SCHL"),
                  _codes("SCHL", mapping, within=lambda p: _age(p) >= 25))


def school_enrollment(area: StudyArea) -> ConstraintTable:
    """``B14001``'s enrolled levels and not enrolled, over persons 3 and over."""
    return _strip(T.school_enrollment(), "school_enrollment")


def employment(area: StudyArea) -> ConstraintTable:
    """``ESR`` as ``B23025``: civilian employed, unemployed, armed forces, not in
    the labour force, over persons 16 and over. ``B23001``'s "employed"
    (civilian and armed forces) is a union of two."""
    return _strip(T.employment(), "employment")


def occupation(area: StudyArea) -> ConstraintTable:
    """``C24010``'s five major groups, over the civilian employed."""
    crosswalk = occupation_crosswalk(area)
    categories = []
    for group in OCCP_TO_MAJOR_GROUP:
        codes = tuple(crosswalk.index[crosswalk == group])
        categories.append(Category(
            name=group,
            select=(lambda c: lambda p: p["ESR"].isin(ESR_CIVILIAN_EMPLOYED) & p["OCCP"].isin(c))(codes),
        ))
    return _table("occupation", "person", ("ESR", "OCCP"), categories)


def industry(area: StudyArea) -> ConstraintTable:
    """``C24030``'s sixteen sectors, over the civilian employed."""
    crosswalk = industry_crosswalk(area)
    categories = []
    for order, label in C24030_CELL_NAMES.items():
        codes = tuple(crosswalk.index[crosswalk == order])
        categories.append(Category(
            name=label,
            select=(lambda c: lambda p: p["ESR"].isin(ESR_CIVILIAN_EMPLOYED) & p["INDP"].isin(c))(codes),
        ))
    return _table("industry", "person", ("ESR", "INDP"), categories)


#: Means of transportation: the common refinement of ``B08301`` (eight modes,
#: worked from home included) and ``B08134`` (car split by occupancy, transit
#: split three ways, four small modes pooled), over workers with a ``JWTRNS``.
TRANSPORT_MODES = {
    "drove_alone": T.B08134_MODES["drove_alone"][0],
    "carpool_2": T.B08134_MODES["carpool_2"][0],
    "carpool_3_plus": T.B08134_MODES["carpool_3_plus"][0],
    "bus": T.B08134_MODES["bus"][0],
    "subway_light_rail": T.B08134_MODES["subway_light_rail"][0],
    "long_distance_rail_ferry": T.B08134_MODES["long_distance_rail_ferry"][0],
    "taxi": lambda p: p["JWTRNS"] == "07",
    "motorcycle": lambda p: p["JWTRNS"] == "08",
    "bicycle": lambda p: p["JWTRNS"] == "09",
    "walked": lambda p: p["JWTRNS"] == "10",
    "other_means": lambda p: p["JWTRNS"] == "12",
    "worked_from_home": lambda p: p["JWTRNS"] == "11",
}


def transport_mode(area: StudyArea) -> ConstraintTable:
    categories = [Category(name=k, select=f) for k, f in TRANSPORT_MODES.items()]
    return _table("transport_mode", "person", ("JWTRNS", "JWRIP"), categories)


def travel_time(area: StudyArea) -> ConstraintTable:
    """Minutes to work on :data:`TRAVEL_TIME_EDGES`, over workers who did not
    work from home; ``JWMNP`` 888 (suppressed) is missing."""
    return _table("travel_time", "person", ("JWTRNS", "JWMNP"),
                  _bands(TRAVEL_TIME_EDGES, T._travel_time, within=T.commuters,
                         prefix="minutes_"))


def disability(area: StudyArea) -> ConstraintTable:
    """``DIS``, over the civilian noninstitutionalized population
    (``B18101``'s universe)."""
    mapping = {"with_disability": (T.DIS_WITH,), "without_disability": (T.DIS_WITHOUT,)}
    return _table("disability", "person", ("DIS", "RELSHIPP", "ESR", "MIL"),
                  _codes("DIS", mapping, within=T.civilian_noninstitutional))


#: The named two-type pairs, the union over ``B27010``'s age groups.
COVERAGE_PAIRS = tuple(dict.fromkeys(
    pair for _, _, cells in T.B27010_GROUPS.values() for pair in cells["pairs"]
))


def health_insurance(area: StudyArea) -> ConstraintTable:
    """Coverage over the civilian noninstitutionalized population: one type
    only (six), each named pair (the union over ``B27010``'s age groups), the
    other private, public and mixed combinations, and none. ``B27010``'s
    categories within an age group are unions of these: a pair it does not
    name falls in its "other" line."""
    cn = T.civilian_noninstitutional

    def only(p, kind):
        cover = T._coverage(p)
        return cn(p) & (cover.sum(axis=1) == 1) & cover[kind]

    def pair(p, kinds):
        cover = T._coverage(p)
        return cn(p) & (cover.sum(axis=1) == 2) & cover[list(kinds)].all(axis=1)

    def other(p, kind):
        cover = T._coverage(p)
        count = cover.sum(axis=1)
        named = pd.Series(False, index=p.index)
        for kinds in COVERAGE_PAIRS:
            named |= (count == 2) & cover[list(kinds)].all(axis=1)
        private = cover[list(T.PRIVATE)].any(axis=1)
        public = cover[list(T.PUBLIC)].any(axis=1)
        mix = {"private": private & ~public, "public": public & ~private,
               "mixed": private & public}[kind]
        return cn(p) & (count >= 2) & ~named & mix

    categories = [Category(name=f"{k}_only", select=(lambda k: lambda p: only(p, k))(k))
                  for k in T.COVERAGE]
    categories += [Category(name="_and_".join(ks), select=(lambda ks: lambda p: pair(p, ks))(ks))
                   for ks in COVERAGE_PAIRS]
    categories += [Category(name=f"other_{k}_combinations",
                            select=(lambda k: lambda p: other(p, k))(k))
                   for k in ("private", "public", "mixed")]
    categories.append(Category(
        name="uninsured", select=lambda p: cn(p) & (T._coverage(p).sum(axis=1) == 0)))
    return _table("health_insurance", "person",
                  ("RELSHIPP", "ESR", "MIL", *T.COVERAGE.values()), categories)


def nativity(area: StudyArea) -> ConstraintTable:
    """``B05002``'s place of birth by nativity and citizenship, over everyone."""
    return _strip(T.place_of_birth(), "nativity")


def mobility(area: StudyArea) -> ConstraintTable:
    """Residence one year ago: same house, elsewhere in the United States,
    abroad (``B07003``'s mobility without its sex split), over persons 1 and
    over."""
    table = T.mobility()
    merged = {}
    for category in table.categories:
        status = category.name.rsplit("_", 1)[0]
        merged.setdefault(status, []).append(category.select)
    categories = [
        Category(name=status, select=(lambda fs: lambda p: fs[0](p) | fs[1](p))(fs))
        for status, fs in merged.items()
    ]
    return _table("mobility", "person", ("MIG", "MIGSP", "SEX"), categories)


def poverty_ratio(area: StudyArea) -> ConstraintTable:
    """``POVPIP`` on :data:`POVERTY_EDGES`, over persons whose poverty status is
    determined (blank otherwise)."""
    return _table("poverty_ratio", "person", ("POVPIP",),
                  _bands(POVERTY_EDGES, lambda p: T._num(p, "POVPIP"), prefix="ratio_"))


# --------------------------------------------------------------------------
# Household attributes (occupied housing units)
# --------------------------------------------------------------------------


def income(area: StudyArea) -> ConstraintTable:
    """Household income in reference-year dollars on :data:`INCOME_EDGES`."""
    return _table("income", "household", ("HINCP", "ADJINC", "TEN"),
                  _bands(INCOME_EDGES, adjusted_household_income, within=T.occupied,
                         prefix="income_"))


#: ``TEN`` codes. ``B25003`` pools 1-2 and 3-4; ``B25091`` splits owners by
#: mortgage (1, 2); ``B25063`` splits renters into cash rent (3) and no cash
#: rent (4). The common refinement is the four codes.
TENURE_CODES = {
    "owned_with_mortgage": ("1",), "owned_free_and_clear": ("2",),
    "rented": ("3",), "occupied_without_rent": ("4",),
}


def tenure(area: StudyArea) -> ConstraintTable:
    return _table("tenure", "household", ("TEN",), _codes("TEN", TENURE_CODES))


def units_in_structure(area: StudyArea) -> ConstraintTable:
    mapping = {k: (code,) for k, (code, _) in T.BLD_CELLS.items()}
    return _table("units_in_structure", "household", ("BLD", "TEN"),
                  _codes("BLD", mapping, within=T.occupied))


def year_built(area: StudyArea) -> ConstraintTable:
    edges = _finite(v for band in T.YEAR_BUILT_BANDS for v in band)
    return _table("year_built", "household", ("YRBLT", "TEN"),
                  _bands(sorted(edges), lambda u: T._num(u, "YRBLT"), within=T.occupied,
                         prefix="built_"))


def rooms(area: StudyArea) -> ConstraintTable:
    edges = _finite(v for band in T.ROOM_BANDS for v in band) - {1.0}
    return _table("rooms", "household", ("RMSP", "TEN"),
                  _bands(sorted(edges), lambda u: T._num(u, "RMSP"), within=T.occupied,
                         prefix="rooms_"))


def bedrooms(area: StudyArea) -> ConstraintTable:
    edges = _finite(v for band in T.BEDROOM_BANDS for v in band) - {0.0}
    return _table("bedrooms", "household", ("BDSP", "TEN"),
                  _bands(sorted(edges), lambda u: T._num(u, "BDSP"), within=T.occupied,
                         prefix="bedrooms_"))


def value(area: StudyArea) -> ConstraintTable:
    """``B25075``'s brackets, over owners."""
    return _table("value", "household", ("TEN", "VALP", "ADJHSG"),
                  _bands(T.VALUE_EDGES, lambda u: T._adjusted_housing(u, "VALP"),
                         within=T.owner, prefix="value_"))


def gross_rent(area: StudyArea) -> ConstraintTable:
    """``B25063``'s cash-rent brackets, over cash renters (``TEN`` 3). Its "no
    cash rent" line is the tenure ``occupied_without_rent``."""
    return _table("gross_rent", "household", ("TEN", "GRNTP", "ADJHSG"),
                  _bands(T.GROSS_RENT_EDGES, lambda u: T._adjusted_housing(u, "GRNTP"),
                         within=lambda u: u["TEN"] == "3", prefix="rent_"))


def vehicles(area: StudyArea) -> ConstraintTable:
    mapping = {k: codes for k, (codes, _) in B25044_VEHICLES.items()}
    return _table("vehicles", "household", ("VEH",), _codes("VEH", mapping))


def heating_fuel(area: StudyArea) -> ConstraintTable:
    mapping = {k: (code,) for k, (code, _) in HFL_TO_CELL.items()}
    return _table("heating_fuel", "household", ("HFL",), _codes("HFL", mapping))


#: ``HHT``: the seven codes are the common refinement of ``B11001`` (living
#: alone), ``B11005`` (nonfamily by householder's sex) and ``B11016`` (family
#: or not).
HHT_CODES = {
    "married_couple": ("1",), "other_family_male": ("2",), "other_family_female": ("3",),
    "nonfamily_male_alone": ("4",), "nonfamily_male_not_alone": ("5",),
    "nonfamily_female_alone": ("6",), "nonfamily_female_not_alone": ("7",),
}


def household_type(area: StudyArea) -> ConstraintTable:
    return _table("household_type", "household", ("HHT",), _codes("HHT", HHT_CODES))


def household_size(area: StudyArea) -> ConstraintTable:
    """``NP``, 1 to 6 then 7 or more (``B11016``, ``B25009``)."""
    return _table("household_size", "household", ("NP", "TEN"),
                  _bands(HOUSEHOLD_SIZE_EDGES, lambda u: T._num(u, "NP"), within=T.occupied,
                         prefix="size_"))


def household_language(area: StudyArea) -> ConstraintTable:
    """``C16002``'s language by limited-English categories."""
    return _strip(T.household_language(), "household_language")


def householder_age_sex(area: StudyArea) -> ConstraintTable:
    """Householder's sex by age on :data:`HOUSEHOLDER_AGE_EDGES` (``B25007``,
    ``B19037``; the sex for ``B09019``'s householder lines)."""
    categories = []
    for sex, code in SEX_CODES.items():
        within = (lambda c: lambda u: u["householder_SEX"] == c)(code)
        categories += _bands(HOUSEHOLDER_AGE_EDGES, lambda u: T._num(u, "HHLDRAGEP"),
                             within=within, prefix=f"{sex}_")
    return _table("householder_age_sex", "household", ("HHLDRAGEP",), categories,
                  householder=("SEX",))


def householder_race_ethnicity(area: StudyArea) -> ConstraintTable:
    """Householder's race by Hispanic origin: 7 races x 2 origins.

    The common refinement of what the configured tables use: ``B17010A``-``G``
    iterate by the householder's race alone, whatever the origin; ``B17010H``
    is White alone not Hispanic; ``B17010I`` is Hispanic of any race. Neither
    ``B03002``'s eight categories (which fold every Hispanic householder into
    one) nor race alone reproduces all nine, so both splits are kept. Race
    codes are :data:`RAC1P_TO_RACE`'s (AIAN alone is ``RAC1P`` 3-5) and the
    origin :data:`NOT_HISPANIC`, read from the householder's person record as
    :func:`~pmedm_vb.assemble.constraints.householder_mask` reads them.

    Household-level on purpose: race/ethnicity sorts households, and in
    stage 2 it is the household that is reweighted. The members' composition
    stays in the person attribute ``race_ethnicity``.
    """
    categories = []
    for race, codes in RAC1P_TO_RACE.items():
        for origin, hispanic in (("not_hispanic", False), ("hispanic", True)):
            def select(u, codes=codes, hispanic=hispanic):
                hisp = u["householder_HISP"]
                same = (hisp != NOT_HISPANIC) if hispanic else (hisp == NOT_HISPANIC)
                return u["householder_RAC1P"].isin(codes) & hisp.notna() & same
            categories.append(Category(name=f"{race}_{origin}", select=select))
    return _table("householder_race_ethnicity", "household", (), categories,
                  householder=("RAC1P", "HISP"))


def occupants_per_room(area: StudyArea) -> ConstraintTable:
    """``B25014``'s five bands of ``NP / RMSP``, closed above."""
    owners = [c.name.removeprefix("owner_") for c in T.tenure_by_occupants_per_room().categories
              if c.name.startswith("owner_")]
    categories = []
    for band, (lower, upper) in zip(owners, T.CROWDING_BANDS):

        def select(u, lower=lower, upper=upper):
            ratio = T._num(u, "NP") / T._num(u, "RMSP")
            return T.occupied(u) & (ratio > lower) & (ratio <= upper)

        categories.append(Category(name=band, select=select))
    return _table("occupants_per_room", "household", ("NP", "RMSP", "TEN"), categories)


def housing_cost_burden(area: StudyArea) -> ConstraintTable:
    """Housing costs as a percentage of income: ``GRPIP`` for renters
    (``B25070``), ``OCPIP`` for owners (``B25091``), on the shared bands, and
    "not computed" (no or negative income, or no rent paid)."""
    def burden(u):
        return T._num(u, "GRPIP").where(T.renter(u), T._num(u, "OCPIP"))

    edges = sorted(_finite(v for band in T.BURDEN_BANDS for v in band))
    categories = _bands(edges, burden, within=T.occupied, prefix="burden_")
    categories.append(Category(name="not_computed",
                               select=lambda u: T.occupied(u) & burden(u).isna()))
    return _table("housing_cost_burden", "household", ("TEN", "GRPIP", "OCPIP"), categories)


def _existing(variable: str, name: str):
    return lambda area: replace(attribute_table(variable), table=name)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributeTarget:
    """One attribute target: its builder, level, universe and dependencies.

    ``derived_from`` maps each attribute it depends on to the kind of
    dependence: ``"value"`` (computed from it) or ``"universe"`` (its universe
    reveals a category of it).
    """

    build: Callable[[StudyArea], ConstraintTable]
    level: str
    universe: Callable[[pd.DataFrame], pd.Series]
    universe_title: str
    derived_from: dict[str, str]


def _cn(p):
    return T.civilian_noninstitutional(p)


def _in_household(p):
    return pd.to_numeric(p["RELSHIPP"], errors="coerce").between(20, 36)


ATTRIBUTE_TARGETS: dict[str, AttributeTarget] = {
    # Person attributes.
    "age_sex": AttributeTarget(age_sex, "person", T.everyone, "Total population", {}),
    "race_ethnicity": AttributeTarget(race_ethnicity_attr, "person", T.everyone,
                                      "Total population", {}),
    "relationship": AttributeTarget(relationship, "person", _in_household,
                                    "Population in households", {}),
    "marital_status": AttributeTarget(marital_status, "person", lambda p: p["MSP"].notna(),
                                      "Population 15 and over", {}),
    "educational_attainment": AttributeTarget(educational_attainment, "person",
                                              lambda p: _age(p) >= 25,
                                              "Population 25 and over", {}),
    "school_enrollment": AttributeTarget(school_enrollment, "person", T.aged(3),
                                         "Population 3 and over", {}),
    "employment": AttributeTarget(employment, "person", T.aged(16),
                                  "Population 16 and over", {}),
    "occupation": AttributeTarget(occupation, "person",
                                  lambda p: p["ESR"].isin(ESR_CIVILIAN_EMPLOYED),
                                  "Civilian employed 16 and over", {"employment": "universe"}),
    "industry": AttributeTarget(industry, "person",
                                lambda p: p["ESR"].isin(ESR_CIVILIAN_EMPLOYED),
                                "Civilian employed 16 and over", {"employment": "universe"}),
    "transport_mode": AttributeTarget(transport_mode, "person", lambda p: p["JWTRNS"].notna(),
                                      "Workers 16 and over", {}),
    "travel_time": AttributeTarget(travel_time, "person", T.commuters,
                                   "Workers who did not work from home",
                                   {"transport_mode": "universe"}),
    "disability": AttributeTarget(disability, "person", _cn,
                                  "Civilian noninstitutionalized population", {}),
    "health_insurance": AttributeTarget(health_insurance, "person", _cn,
                                        "Civilian noninstitutionalized population", {}),
    "nativity": AttributeTarget(nativity, "person", T.everyone, "Total population", {}),
    "mobility": AttributeTarget(mobility, "person", T.aged(1), "Population 1 and over", {}),
    "poverty_ratio": AttributeTarget(poverty_ratio, "person", lambda p: p["POVPIP"].notna(),
                                     "Population for whom poverty status is determined",
                                     {"income": "value", "age_sex": "value",
                                      "relationship": "value"}),
    # Household attributes.
    "income": AttributeTarget(income, "household", T.occupied, "Households", {}),
    "tenure": AttributeTarget(tenure, "household", T.occupied, "Occupied housing units", {}),
    "units_in_structure": AttributeTarget(units_in_structure, "household", T.occupied,
                                          "Occupied housing units", {}),
    "year_built": AttributeTarget(year_built, "household", T.occupied,
                                  "Occupied housing units", {}),
    "rooms": AttributeTarget(rooms, "household", T.occupied, "Occupied housing units", {}),
    "bedrooms": AttributeTarget(bedrooms, "household", T.occupied, "Occupied housing units", {}),
    "value": AttributeTarget(value, "household", T.owner, "Owner-occupied housing units",
                             {"tenure": "universe"}),
    "gross_rent": AttributeTarget(gross_rent, "household", lambda u: u["TEN"] == "3",
                                  "Renter-occupied units paying cash rent",
                                  {"tenure": "universe"}),
    "vehicles": AttributeTarget(vehicles, "household", T.occupied, "Occupied housing units", {}),
    "heating_fuel": AttributeTarget(heating_fuel, "household", T.occupied,
                                    "Occupied housing units", {}),
    "household_type": AttributeTarget(household_type, "household", T.occupied, "Households",
                                      {"relationship": "value", "age_sex": "value"}),
    "household_size": AttributeTarget(household_size, "household", T.occupied, "Households",
                                      {"age_sex": "value"}),
    "household_language": AttributeTarget(household_language, "household", T.occupied,
                                          "Households", {}),
    "householder_age_sex": AttributeTarget(householder_age_sex, "household", T.occupied,
                                           "Households",
                                           {"age_sex": "value", "relationship": "value"}),
    "householder_race_ethnicity": AttributeTarget(householder_race_ethnicity, "household",
                                                  T.occupied, "Households",
                                                  {"race_ethnicity": "value",
                                                   "relationship": "value"}),
    "occupants_per_room": AttributeTarget(occupants_per_room, "household", T.occupied,
                                          "Occupied housing units",
                                          {"rooms": "value", "age_sex": "value"}),
    "housing_cost_burden": AttributeTarget(housing_cost_burden, "household", T.occupied,
                                           "Occupied housing units",
                                           {"income": "value", "tenure": "value",
                                            "gross_rent": "value"}),
    "multigenerational": AttributeTarget(_existing("MULTG", "multigenerational"), "household",
                                         T.occupied, "Households", {"relationship": "value"}),
    "couple_type": AttributeTarget(_existing("CPLT", "couple_type"), "household", T.occupied,
                                   "Households", {"relationship": "value", "age_sex": "value"}),
    "children_presence_age": AttributeTarget(_existing("HUPAC", "children_presence_age"),
                                             "household", T.occupied, "Households",
                                             {"age_sex": "value", "relationship": "value"}),
    "time_since_move": AttributeTarget(_existing("MV", "time_since_move"), "household",
                                       T.occupied, "Households", {}),
}

#: Attributes a configured table's categories are expected to be a function
#: of, for :func:`refinement_violations`. The group-quarters unit type is
#: always available as well. ``B26001`` is the unit type itself.
TABLE_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "B01001": ("age_sex",),
    "B03002": ("race_ethnicity",),
    "B08301": ("transport_mode",),
    "B19001": ("income",),
    "B25003": ("tenure",),
    "C24010": ("age_sex", "occupation"),
    "B26001": (),
    "B17024": ("age_sex", "poverty_ratio"),
    "B23001": ("age_sex", "employment"),
    "B25044": ("tenure", "vehicles"),
    "C17002": ("poverty_ratio",),
    "C24030": ("age_sex", "industry"),
    "B12001": ("age_sex", "marital_status"),
    "B15002": ("age_sex", "educational_attainment"),
    "B25040": ("heating_fuel",),
    **{f"B17010{s}": ("household_type", "householder_race_ethnicity", "poverty_ratio")
       for s in "ABCDEFGHI"},
    "B25032": ("tenure", "units_in_structure"),
    "B25036": ("tenure", "year_built"),
    "B25020": ("tenure", "rooms"),
    "B25042": ("tenure", "bedrooms"),
    "B25014": ("tenure", "occupants_per_room"),
    "B25075": ("value",),
    "B25063": ("tenure", "gross_rent"),
    "B08303": ("travel_time",),
    "B08134": ("transport_mode", "travel_time"),
    "B11001": ("household_type",),
    "B11016": ("household_type", "household_size"),
    "B11005": ("household_type", "age_sex"),
    "B25007": ("tenure", "householder_age_sex"),
    "B25009": ("tenure", "household_size"),
    "B19037": ("householder_age_sex", "income"),
    "B09019": ("relationship", "age_sex", "household_size"),
    "B23025": ("employment",),
    "B25070": ("tenure", "housing_cost_burden"),
    "B25091": ("tenure", "housing_cost_burden"),
    "B27010": ("age_sex", "health_insurance"),
    "B14001": ("school_enrollment",),
    "B18101": ("age_sex", "disability"),
    "C16002": ("household_language",),
    "B05002": ("nativity",),
    "B07003": ("age_sex", "mobility"),
    "MULTG": ("multigenerational",),
    "CPLT": ("couple_type",),
    "HUPAC": ("children_presence_age",),
    "MV": ("time_since_move",),
}


def build_attribute_target(area: StudyArea, name: str) -> ConstraintTable:
    return ATTRIBUTE_TARGETS[name].build(area)


def is_attribute_target(name: str) -> bool:
    return name in ATTRIBUTE_TARGETS


def mask_closure(ids: Sequence[str]) -> dict[str, list[str]]:
    """For each target in ``ids``: the targets in ``ids`` masked with it in
    check 1 -- itself and, transitively, every attribute depending on it
    (:data:`ATTRIBUTE_TARGETS` ``derived_from``). Unknown ids mask only
    themselves."""
    ids = list(ids)
    dependents: dict[str, set[str]] = {i: set() for i in ids}
    for i in ids:
        spec = ATTRIBUTE_TARGETS.get(i)
        for parent in (spec.derived_from if spec else {}):
            if parent in dependents:
                dependents[parent].add(i)
    closure = {}
    for i in ids:
        seen, stack = {i}, [i]
        while stack:
            for child in dependents[stack.pop()]:
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        closure[i] = [j for j in ids if j in seen]
    return closure


# --------------------------------------------------------------------------
# Checks against the PUMS
# --------------------------------------------------------------------------


def labels(table: ConstraintTable, frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Each record's category index in ``table`` (-1 for none) and the number
    of records falling in two or more."""
    out = np.full(len(frame), -1, dtype=np.int32)
    several = np.zeros(len(frame), dtype=bool)
    for j, category in enumerate(table.categories):
        mask = category.select(frame)
        mask = mask.fillna(False).to_numpy(dtype=bool) if isinstance(mask, pd.Series) \
            else np.asarray(mask, dtype=bool)
        several |= mask & (out >= 0)
        out[mask] = j
    return out, int(several.sum())


def partition_report(units: pd.DataFrame, persons: pd.DataFrame,
                     names: Sequence[str], built: dict[str, ConstraintTable]) -> pd.DataFrame:
    """For each attribute: records in its universe, in none of its
    categories, in several, and claimed although outside the universe."""
    rows = []
    for name in names:
        spec, table = ATTRIBUTE_TARGETS[name], built[name]
        frame = persons if spec.level == "person" else units
        lab, several = labels(table, frame)
        inside = spec.universe(frame).fillna(False).to_numpy(dtype=bool)
        rows.append({"attribute": name, "level": spec.level, "categories": len(table.categories),
                     "in_universe": int(inside.sum()), "none": int((inside & (lab < 0)).sum()),
                     "several": several, "outside_claimed": int((~inside & (lab >= 0)).sum())})
    return pd.DataFrame(rows)


def refinement_violations(
    units: pd.DataFrame,
    persons: pd.DataFrame,
    tables: dict[str, ConstraintTable],
    built: dict[str, ConstraintTable],
) -> pd.DataFrame:
    """Whether each table's categories are unions of its attributes' categories.

    For every record of a table's frame (persons for a person table, units for
    a household table), the record's table category (or "none") must be a
    function of the attributes :data:`TABLE_ATTRIBUTES` lists for it plus the
    unit type: a person record sees its own person-attribute categories and
    its unit's household-attribute categories; a unit record sees its
    household-attribute categories and, for person attributes, its members'
    count vector. ``violating`` counts records in tuples that map to more than
    one table category.
    """
    unit_row = {}
    person_unit = units.index.get_indexer(persons["SERIALNO"])
    unit_type = units["TYPEHUGQ"].astype("string").fillna("1").to_numpy()
    for name, table in built.items():
        level = ATTRIBUTE_TARGETS[name].level
        frame = persons if level == "person" else units
        unit_row[name], _ = labels(table, frame)

    rows = []
    for table_id, table in tables.items():
        attrs = TABLE_ATTRIBUTES.get(table_id)
        if attrs is None or any(a not in built for a in attrs):
            rows.append({"table": table_id, "checked": False})
            continue
        if table.universe == "person":
            frame = persons
            columns = {"unit_type": unit_type[person_unit]}
            for a in attrs:
                lab = unit_row[a]
                columns[a] = lab if ATTRIBUTE_TARGETS[a].level == "person" else lab[person_unit]
        else:
            frame = units
            columns = {"unit_type": unit_type}
            for a in attrs:
                lab = unit_row[a]
                if ATTRIBUTE_TARGETS[a].level == "household":
                    columns[a] = lab
                else:
                    k = len(built[a].categories)
                    valid = lab >= 0
                    counts = np.zeros((len(units), k), dtype=np.int16)
                    np.add.at(counts, (person_unit[valid], lab[valid]), 1)
                    columns[a] = pd.Series(list(map(bytes, counts))).to_numpy()
        target, _ = labels(table, frame)
        key = pd.DataFrame(columns)
        key["__t"] = target
        spread = key.groupby(list(columns), dropna=False)["__t"].transform("nunique")
        bad = spread.to_numpy() > 1
        rows.append({"table": table_id, "checked": True, "records": len(frame),
                     "in_table": int((target >= 0).sum()), "violating": int(bad.sum())})
    return pd.DataFrame(rows)
