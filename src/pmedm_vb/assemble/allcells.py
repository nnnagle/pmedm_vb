"""Every published cell of every table: the all-cell constraint and held-out sets.

The tables of :func:`~pmedm_vb.assemble.constraints.default_tables` and
:func:`~pmedm_vb.assemble.constraints.heldout_tables`, but with nothing
collapsed and nothing waived: each published line is one category, totals and
subtotals included. The redundancy that adds -- a subtotal's ``X`` column is
the sum of its children's, and its replicates the sum of theirs -- is left to
the model: the null space removes it (``--hierarchy nullspace``), and at small
``alpha`` the replicate covariance in ``Sigma`` carries it. What a subtotal does
add is a published zero's own precision: a zero subtotal is one modelled
``w k`` for the whole block, where its zero leaves are one each.

**A table is written as its published tree.** Each :class:`Line` is one
published cell, by its ``ORDER``; a leaf carries a PUMS selector and a subtotal
its children. A subtotal's selector is the union of its leaves', so its ``X``
column is exactly the sum of theirs. The one exception is declared where it
occurs (``B23001`` at 65 and over). :func:`tree_table` walks the tree in
pre-order and requires the orders to come out 1, 2, ..., n: the published lines
are in pre-order, so a line placed under the wrong parent, or a line missing,
shows up as an order out of sequence, which :meth:`ConstraintTable.validate
<pmedm_vb.assemble.constraints.ConstraintTable.validate>` (every cell claimed
once) cannot see. The trees were read off ``experiments/table_trees.py`` (the
constraint tables, each subtotal checked to equal the sum of its children in
every area and replicate) and the published cell titles (the held-out tables).

**A subtotal's name ends in ``":"``,** as its published title does, so that
whatever reads the names can tell the leaves -- which partition the table --
from the totals over them (``compare_methods.py`` computes shares over the
leaves only).

The finer cells need PUMS fields the collapsed tables did not read, all from
the 2020-2024 data dictionary: ``RACNUM`` and ``RACSOR`` (``B03002``'s
two-or-more-races split), ``JWRIP`` (``B08301``'s carpool sizes), ``SOCP``
(``C24010``'s occupation lines) and the ``INDP`` code ranges of the 2022 Census
Industry Code List (``C24030``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from operator import or_

import pandas as pd

from pmedm_vb.assemble.constraints import (
    B01001_BANDS,
    B12001_STATUS,
    B15002_ATTAINMENT,
    B17024_AGE_GROUPS,
    B17024_RATIO_BANDS,
    B19001_BRACKETS,
    B23001_AGE_GROUPS,
    B23001_FEMALE_OFFSET,
    B25044_VEHICLES,
    C17002_BANDS,
    ESR_CIVILIAN_EMPLOYED,
    HFL_TO_CELL,
    HELDOUT_RELATION,
    INDP_TO_C24030_CELL,
    NOT_HISPANIC,
    OCCP_TO_MAJOR_GROUP,
    RAC1P_TO_RACE,
    SEX_CODES,
    TEN_TO_TENURE,
    TYPEHUGQ_GROUP_QUARTERS,
    Category,
    ConstraintTable,
)
from pmedm_vb.config import StudyArea
from pmedm_vb.data.pums import variable_labels

#: The table sets a study area can be assembled with.
TABLE_SETS = ("default", "all")

#: A subtotal's name ends in this, as its published title does.
SUBTOTAL_MARK = ":"

Select = Callable[[pd.DataFrame], pd.Series]


def is_subtotal(name: str) -> bool:
    """Whether a category name (``"{table}.{category}"`` or bare) is a subtotal."""
    return name.endswith(SUBTOTAL_MARK)


def table_set(names: list[str]) -> str:
    """Which :data:`TABLE_SETS` member built these constraint names: only the
    all-cell set has subtotals."""
    return "all" if any(is_subtotal(n) for n in names) else "default"


@dataclass(frozen=True)
class Line:
    """One published cell: its ``ORDER``, a label, and a selector or children.

    A leaf has ``select`` and no children. A subtotal has children, and a
    ``select`` only where the published subtotal is not the sum of its children
    (``B23001``'s 65-and-over groups); otherwise it is their union.
    """

    order: int
    label: str
    select: Select | None = None
    children: tuple[Line, ...] = ()


def union(selects: list[Select]) -> Select:
    """The union of disjoint selectors: the persons (or units) any of them selects."""
    return lambda frame: reduce(or_, (select(frame) for select in selects))


def tree_table(table: str, universe: str, variables: tuple[str, ...], geography: str,
               root: Line, n_cells: int) -> ConstraintTable:
    """One category per published line, in pre-order, every cell used."""
    categories: list[Category] = []

    def walk(line: Line, path: tuple[str, ...]) -> list[Select]:
        """Append ``line``'s category (pre-order) and return its leaf selectors."""
        name = "_".join(path) if path else "total"
        if not line.children:
            if line.select is None:
                raise ValueError(f"{table}_{line.order:03d} is a leaf with no selector")
            categories.append(Category(name=name, select=line.select,
                                       published=(f"{table}_{line.order:03d}",)))
            return [line.select]
        index = len(categories)
        categories.append(None)  # placeholder: pre-order, filled once the leaves are known
        leaves = [s for child in line.children for s in walk(child, (*path, child.label))]
        categories[index] = Category(
            name=name + SUBTOTAL_MARK,
            select=line.select if line.select is not None else union(leaves),
            published=(f"{table}_{line.order:03d}",),
        )
        return leaves

    walk(root, ())
    orders = [int(c.published[0].rsplit("_", 1)[1]) for c in categories]
    if orders != list(range(1, n_cells + 1)):
        raise AssertionError(
            f"{table}: the tree's lines are not the published 1..{n_cells} in pre-order: "
            f"first out of sequence at position "
            f"{next((i for i, o in enumerate(orders) if o != i + 1), len(orders))}"
        )
    names = [c.name for c in categories]
    if len(set(names)) != len(names):
        raise AssertionError(f"{table}: duplicate category names")
    return ConstraintTable(table=table, universe=universe, variables=variables,
                           geography=geography, categories=tuple(categories))


def _age(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["AGEP"], errors="coerce")


def _between(values: pd.Series, lower: float | None, upper: float | None) -> pd.Series:
    """``lower <= values < upper``; ``None`` is an open end. Null fails."""
    mask = values.notna()
    if lower is not None:
        mask &= values >= lower
    if upper is not None:
        mask &= values < upper
    return mask


def _sex(sex: str) -> Select:
    return lambda p: p["SEX"] == SEX_CODES[sex]


def _and(*selects: Select) -> Select:
    return lambda frame: reduce(lambda a, b: a & b, (s(frame) for s in selects))


# -- constraint tables ---------------------------------------------------------


def age_sex(geography: str = "block group") -> ConstraintTable:
    """``B01001``: total, each sex, and its 23 age bands. 49 cells."""
    def bands(sex: str, first: int) -> tuple[Line, ...]:
        return tuple(
            Line(first + i, f"{lo}" if hi is None else f"{lo}_{hi}",
                 _and(_sex(sex), lambda p, lo=lo, hi=hi: _between(_age(p), lo, hi)))
            for i, (lo, hi) in enumerate(B01001_BANDS)
        )
    root = Line(1, "total", children=(
        Line(2, "male", children=bands("male", 3)),
        Line(26, "female", children=bands("female", 27)),
    ))
    return tree_table("B01001", "person", ("AGEP", "SEX"), geography, root, 49)


def race_ethnicity(geography: str = "block group") -> ConstraintTable:
    """``B03002``: Hispanic origin by race, both sides to the two-or-more split. 21 cells.

    "Two races including Some other race" is ``RACNUM`` 2 with ``RACSOR`` 1 --
    ``RACNUM`` counts the six major race groups (the five OMB groups and Some
    other race), the groups the table's "two races" counts. The rest of
    ``RAC1P`` 9, two races without Some other race and three or more, is the
    other line.
    """
    def side(first: int, hispanic: bool) -> tuple[Line, ...]:
        origin = (lambda p: p["HISP"] != NOT_HISPANIC) if hispanic else (
            lambda p: p["HISP"] == NOT_HISPANIC)
        alone = [
            Line(first + i, label,
                 _and(origin, lambda p, c=codes: p["RAC1P"].isin(c)))
            for i, (label, codes) in enumerate(RAC1P_TO_RACE.items())
            if label != "two_or_more"
        ]

        def two_with_sor(p):
            return (p["RAC1P"] == "9") & (p["RACNUM"] == "2") & (p["RACSOR"] == "1")

        two = Line(first + 6, "two_or_more", children=(
            Line(first + 7, "two_with_sor", _and(origin, two_with_sor)),
            Line(first + 8, "other_two_or_three_plus",
                 _and(origin, lambda p: (p["RAC1P"] == "9") & ~two_with_sor(p))),
        ))
        return (*alone, two)

    root = Line(1, "total", children=(
        Line(2, "not_hispanic", children=side(3, hispanic=False)),
        Line(12, "hispanic", children=side(13, hispanic=True)),
    ))
    return tree_table("B03002", "person", ("HISP", "RAC1P", "RACNUM", "RACSOR"),
                      geography, root, 21)


def means_of_transportation(geography: str = "block group") -> ConstraintTable:
    """``B08301``: commute mode with car, truck or van by occupancy. 21 cells.

    Occupancy is ``JWRIP``, the number of people in the vehicle (1 = drove
    alone; 10 is top-coded "10 or more"), defined exactly for ``JWTRNS`` 01.
    """
    def mode(code: str) -> Select:
        return lambda p: p["JWTRNS"] == code

    def riders(lo: int, hi: int | None) -> Select:
        return _and(mode("01"),
                    lambda p: _between(pd.to_numeric(p["JWRIP"], errors="coerce"), lo, hi))

    root = Line(1, "total", children=(
        Line(2, "car_truck_van", children=(
            Line(3, "drove_alone", riders(1, 2)),
            Line(4, "carpooled", children=(
                Line(5, "2_person", riders(2, 3)),
                Line(6, "3_person", riders(3, 4)),
                Line(7, "4_person", riders(4, 5)),
                Line(8, "5_6_person", riders(5, 7)),
                Line(9, "7plus_person", riders(7, None)),
            )),
        )),
        Line(10, "public_transport", children=(
            Line(11, "bus", mode("02")),
            Line(12, "subway_elevated", mode("03")),
            Line(13, "commuter_rail", mode("04")),
            Line(14, "light_rail", mode("05")),
            Line(15, "ferry", mode("06")),
        )),
        Line(16, "taxi", mode("07")),
        Line(17, "motorcycle", mode("08")),
        Line(18, "bicycle", mode("09")),
        Line(19, "walked", mode("10")),
        Line(20, "other_means", mode("12")),
        Line(21, "worked_from_home", mode("11")),
    ))
    return tree_table("B08301", "person", ("JWTRNS", "JWRIP"), geography, root, 21)


def household_income(geography: str = "block group") -> ConstraintTable:
    """``B19001``: all 16 published income brackets. 17 cells."""
    from pmedm_vb.assemble.constraints import adjusted_household_income

    leaves = tuple(
        Line(int(cell.rsplit("_", 1)[1]),
             f"lt_{hi}" if lo is None else f"ge_{lo}" if hi is None else f"{lo}_{hi}",
             lambda h, lo=lo, hi=hi: _between(adjusted_household_income(h), lo, hi))
        for cell, lo, hi in B19001_BRACKETS
    )
    return tree_table("B19001", "household", ("HINCP", "ADJINC"), geography,
                      Line(1, "total", children=leaves), 17)


def tenure(geography: str = "block group") -> ConstraintTable:
    """``B25003``: owner and renter (``TEN`` 4 with renters, as in the default set)."""
    root = Line(1, "total", children=(
        Line(2, "owner", lambda h: h["TEN"].isin(TEN_TO_TENURE["owner"])),
        Line(3, "renter", lambda h: h["TEN"].isin(TEN_TO_TENURE["renter"])),
    ))
    return tree_table("B25003", "household", ("TEN",), geography, root, 3)


#: ``C24010``'s detailed occupation lines as SOC code prefixes (``SOCP``, no
#: dash), with the C24010 major group each lies in (:data:`OCCP_TO_MAJOR_GROUP`).
#: A code takes the line of its longest matching prefix. Supervisors of
#: transportation and material moving workers (53-1000) are a transportation
#: occupation, as the 2018 Census Occupation Code List groups them (9005-9430).
#: Dental hygienists (29-1292) are a health technologist, with whom their
#: Census code (3310) sits, although SOC puts them among the practitioners:
#: Tennessee's PUMS-weighted totals match the published lines that way and not
#: the other (``experiments/check_occupation_lines.py``, ``HANDOFF_paper.md``).
#: Other healthcare practitioners and technical (29-9000, Census code 3550) go
#: with the practitioners, as the line's title says and the same check agrees.
SOC_LINES = {
    "management": (("11",), "management_business_science_arts"),
    "business_financial": (("13",), "management_business_science_arts"),
    "computer_math": (("15",), "management_business_science_arts"),
    "architecture_engineering": (("17",), "management_business_science_arts"),
    "life_physical_social_science": (("19",), "management_business_science_arts"),
    "community_social_service": (("21",), "management_business_science_arts"),
    "legal": (("23",), "management_business_science_arts"),
    "education_library": (("25",), "management_business_science_arts"),
    "arts_media": (("27",), "management_business_science_arts"),
    "health_practitioners": (("291", "299"), "management_business_science_arts"),
    "health_technologists": (("292", "291292"), "management_business_science_arts"),
    "healthcare_support": (("31",), "service"),
    "fire_other_protective": (("331021", "33109", "332", "339"), "service"),
    "law_enforcement": (("331011", "331012", "333"), "service"),
    "food_preparation": (("35",), "service"),
    "building_cleaning": (("37",), "service"),
    "personal_care": (("39",), "service"),
    "sales": (("41",), "sales_office"),
    "office_admin": (("43",), "sales_office"),
    "farming_fishing_forestry": (("45",), "natural_resources_construction_maintenance"),
    "construction_extraction": (("47",), "natural_resources_construction_maintenance"),
    "installation_repair": (("49",), "natural_resources_construction_maintenance"),
    "production": (("51",), "production_transportation_material_moving"),
    "transportation": (("531", "532", "533", "534", "535", "536"),
                       "production_transportation_material_moving"),
    "material_moving": (("537",), "production_transportation_material_moving"),
}

#: Outside C24010's civilian employed universe (and excluded by ``ESR`` too):
#: military occupations, and 999920, unemployed with no work in the last five
#: years.
SOC_EXCLUDED = ("55", "9999")


def soc_lines(code: str) -> list[str]:
    """The :data:`SOC_LINES` lines of ``code``'s longest matching prefix (one,
    unless the table is ambiguous; none for a code outside every line)."""
    lengths = {line: max((len(p) for p in prefixes if code.startswith(p)), default=0)
               for line, (prefixes, _) in SOC_LINES.items()}
    longest = max(lengths.values())
    return [line for line, n in lengths.items() if n == longest and n > 0]


def soc_crosswalk(area: StudyArea) -> pd.Series:
    """``SOCP`` code to its :data:`SOC_LINES` line, from the data dictionary.

    Raises if a code matches no line or two, or if a code's label prefix (the
    ``OCCP`` taxonomy, ``"MED-..."``) puts it in a different C24010 major group
    than its line -- a check of the SOC reading against the prefix one the
    default set uses.
    """
    labels = variable_labels(area, "SOCP")
    group_of_prefix = {prefix: group for group, members in OCCP_TO_MAJOR_GROUP.items()
                       for prefix in members}
    lines = {}
    for code, label in zip(labels["lo"], labels["label"]):
        code = str(code)
        if code.startswith(SOC_EXCLUDED) or not code[:2].isdigit():
            continue
        matches = soc_lines(code)
        if len(matches) != 1:
            raise ValueError(f"SOCP {code} ({label}) matches {matches or 'no'} C24010 line(s)")
        prefix_group = group_of_prefix.get(str(label)[:3])
        if prefix_group != SOC_LINES[matches[0]][1]:
            raise ValueError(f"SOCP {code} ({label}): line {matches[0]} is in "
                             f"{SOC_LINES[matches[0]][1]}, its label prefix in {prefix_group}")
        lines[code] = matches[0]
    return pd.Series(lines, name="line").rename_axis("socp")


def sex_by_occupation(area: StudyArea, geography: str = "block group") -> ConstraintTable:
    """``C24010``: every occupation line by sex, over the civilian employed. 73 cells."""
    crosswalk = soc_crosswalk(area)

    def occupation(sex: str, line: str) -> Select:
        codes = tuple(crosswalk.index[crosswalk == line])
        return lambda p: ((p["SEX"] == SEX_CODES[sex]) & p["ESR"].isin(ESR_CIVILIAN_EMPLOYED)
                          & p["SOCP"].isin(codes))

    def block(sex: str, first: int) -> tuple[Line, ...]:
        def leaf(offset: int, line: str) -> Line:
            return Line(first + offset, line, occupation(sex, line))
        return (
            Line(first + 1, "mbsa", children=(
                Line(first + 2, "management_business_financial", children=(
                    leaf(3, "management"), leaf(4, "business_financial"))),
                Line(first + 5, "computer_engineering_science", children=(
                    leaf(6, "computer_math"), leaf(7, "architecture_engineering"),
                    leaf(8, "life_physical_social_science"))),
                Line(first + 9, "education_legal_community_arts_media", children=(
                    leaf(10, "community_social_service"), leaf(11, "legal"),
                    leaf(12, "education_library"), leaf(13, "arts_media"))),
                Line(first + 14, "healthcare_practitioners_technical", children=(
                    leaf(15, "health_practitioners"), leaf(16, "health_technologists"))),
            )),
            Line(first + 17, "service", children=(
                leaf(18, "healthcare_support"),
                Line(first + 19, "protective_service", children=(
                    leaf(20, "fire_other_protective"), leaf(21, "law_enforcement"))),
                leaf(22, "food_preparation"), leaf(23, "building_cleaning"),
                leaf(24, "personal_care"),
            )),
            Line(first + 25, "sales_office", children=(
                leaf(26, "sales"), leaf(27, "office_admin"))),
            Line(first + 28, "natural_resources_construction_maintenance", children=(
                leaf(29, "farming_fishing_forestry"), leaf(30, "construction_extraction"),
                leaf(31, "installation_repair"))),
            Line(first + 32, "production_transportation_material_moving", children=(
                leaf(33, "production"), leaf(34, "transportation"), leaf(35, "material_moving"))),
        )

    root = Line(1, "total", children=(
        Line(2, "male", children=block("male", 2)),
        Line(38, "female", children=block("female", 38)),
    ))
    return tree_table("C24010", "person", ("ESR", "SEX", "SOCP"), geography, root, 73)


def group_quarters(geography: str = "tract") -> ConstraintTable:
    """``B26001``: the group-quarters population, one cell (see the default set's)."""
    root = Line(1, "total", lambda u: u["TYPEHUGQ"].isin(TYPEHUGQ_GROUP_QUARTERS))
    return tree_table("B26001", "household", ("TYPEHUGQ",), geography, root, 1)


def poverty_by_age(geography: str = "tract") -> ConstraintTable:
    """``B17024``: all 12 income-to-poverty bands in each of 10 age groups. 131 cells."""
    groups = []
    for first, age_lo, age_hi in B17024_AGE_GROUPS:
        bands = tuple(
            Line(first + offset, f"{lo}" if hi is None else f"{lo}_{hi}",
                 lambda p, alo=age_lo, ahi=age_hi, lo=lo, hi=hi: (
                     _between(_age(p), alo, ahi)
                     & _between(pd.to_numeric(p["POVPIP"], errors="coerce"), lo, hi)))
            for offset, lo, hi in B17024_RATIO_BANDS
        )
        label = f"age_{age_lo}" if age_hi is None else f"age_{age_lo}_{age_hi}"
        groups.append(Line(first, label, children=bands))
    return tree_table("B17024", "person", ("AGEP", "POVPIP"), geography,
                      Line(1, "total", children=tuple(groups)), 131)


def employment_status(geography: str = "tract") -> ConstraintTable:
    """``B23001``: employment status in every age group by sex. 173 cells.

    Under 65 a group is in labor force (armed forces; civilian: employed,
    unemployed) and not in labor force. At 65 and over the table has no armed
    forces line: in labor force is employed plus unemployed, civilians only,
    while the age group itself still counts the armed forces -- nationally the
    65-69 group exceeds the sum of its lines by 9 people
    (``experiments/table_trees.py``). So those groups, and the sex and table
    totals above them, select by age directly rather than as the union of
    their lines; the published subtotals are reproduced as published.
    """
    def esr(*codes: str) -> Select:
        return lambda p: p["ESR"].isin(codes)

    def in_ages(sex: str, lo: int, hi: int | None) -> Select:
        return _and(_sex(sex), lambda p: _between(_age(p), lo, hi), lambda p: p["ESR"].notna())

    def block(sex: str, offset: int) -> tuple[Line, ...]:
        groups = []
        for first, lo, hi, shape in B23001_AGE_GROUPS:
            g = first + offset
            ages = in_ages(sex, lo, hi)
            if shape == 7:
                lines = (
                    Line(g + 1, "in_labor_force", children=(
                        Line(g + 2, "armed_forces", _and(ages, esr("4", "5"))),
                        Line(g + 3, "civilian", children=(
                            Line(g + 4, "employed", _and(ages, esr("1", "2"))),
                            Line(g + 5, "unemployed", _and(ages, esr("3"))),
                        )),
                    )),
                    Line(g + 6, "not_in_labor_force", _and(ages, esr("6"))),
                )
                select = None
            else:
                lines = (
                    Line(g + 1, "in_labor_force", children=(
                        Line(g + 2, "employed", _and(ages, esr("1", "2"))),
                        Line(g + 3, "unemployed", _and(ages, esr("3"))),
                    )),
                    Line(g + 4, "not_in_labor_force", _and(ages, esr("6"))),
                )
                select = ages
            label = f"age_{lo}" if hi is None else f"age_{lo}_{hi}"
            groups.append(Line(g, label, select=select, children=lines))
        return tuple(groups)

    root = Line(1, "total", select=lambda p: _between(_age(p), 16, None) & p["ESR"].notna(),
                children=(
                    Line(2, "male", select=in_ages("male", 16, None), children=block("male", 0)),
                    Line(2 + B23001_FEMALE_OFFSET, "female", select=in_ages("female", 16, None),
                         children=block("female", B23001_FEMALE_OFFSET)),
                ))
    return tree_table("B23001", "person", ("SEX", "AGEP", "ESR"), geography, root, 173)


def core_tables(area: StudyArea, geography: str) -> list[ConstraintTable]:
    """The six tables published at both tract and block group, every cell."""
    return [
        age_sex(geography),
        race_ethnicity(geography),
        means_of_transportation(geography),
        household_income(geography),
        tenure(geography),
        sex_by_occupation(area, geography),
    ]


def all_tables(area: StudyArea) -> list[ConstraintTable]:
    """The all-cell constraint set: the default set's tables, every published cell."""
    return [
        *core_tables(area, "block group"),
        *core_tables(area, "tract"),
        group_quarters("tract"),
        poverty_by_age("tract"),
        employment_status("tract"),
    ]


# -- held-out tables -----------------------------------------------------------


def tenure_by_vehicles(geography: str = "block group") -> ConstraintTable:
    """``B25044``: tenure by vehicles available. 15 cells."""
    def side(tenure_label: str, first: int) -> Line:
        leaves = tuple(
            Line(first + offset, label,
                 lambda h, t=TEN_TO_TENURE[tenure_label], v=codes: h["TEN"].isin(t) & h["VEH"].isin(v))
            for label, (codes, offset) in B25044_VEHICLES.items()
        )
        return Line(first, tenure_label, children=leaves)

    root = Line(1, "total", children=(side("owner", 2), side("renter", 9)))
    return tree_table("B25044", "household", ("TEN", "VEH"), geography, root, 15)


def poverty_ratio(geography: str = "block group") -> ConstraintTable:
    """``C17002``: ratio of income to poverty level. 8 cells."""
    leaves = tuple(
        Line(int(cell.rsplit("_", 1)[1]), f"{lo}" if hi is None else f"{lo}_{hi}",
             lambda p, lo=lo, hi=hi: _between(pd.to_numeric(p["POVPIP"], errors="coerce"), lo, hi))
        for cell, lo, hi in C17002_BANDS
    )
    return tree_table("C17002", "person", ("POVPIP",), geography,
                      Line(1, "total", children=leaves), 8)


#: ``C24030``'s sectors as 2022 Census industry code ranges (``INDP``,
#: inclusive), from the 2022 Census Industry Code List, by male-block order.
C24030_SECTORS = {
    4: ("agriculture_forestry_fishing", ((170, 290),)),
    5: ("mining", ((370, 490),)),
    6: ("construction", ((770, 770),)),
    7: ("manufacturing", ((1070, 3990),)),
    8: ("wholesale", ((4070, 4590),)),
    9: ("retail", ((4670, 5791),)),
    11: ("transportation_warehousing", ((6070, 6390),)),
    12: ("utilities", ((570, 690),)),
    13: ("information", ((6471, 6781),)),
    15: ("finance_insurance", ((6871, 6992),)),
    16: ("real_estate_rental", ((7071, 7190),)),
    18: ("professional_scientific_technical", ((7270, 7490),)),
    19: ("management_of_companies", ((7570, 7570),)),
    20: ("administrative_waste", ((7580, 7790),)),
    22: ("educational_services", ((7860, 7890),)),
    23: ("health_social_assistance", ((7970, 8470),)),
    25: ("arts_entertainment_recreation", ((8561, 8590),)),
    26: ("accommodation_food", ((8660, 8690),)),
    27: ("other_services", ((8770, 9290),)),
    28: ("public_administration", ((9370, 9590),)),
}

#: ``C24030``'s two-sector subtotals, by male-block order, and their children.
C24030_SUBTOTALS = {
    3: ("agriculture_mining", (4, 5)),
    10: ("transportation_utilities", (11, 12)),
    14: ("finance_real_estate", (15, 16)),
    17: ("professional_management_administrative", (18, 19, 20)),
    21: ("education_health", (22, 23)),
    24: ("arts_accommodation_food", (25, 26)),
}

#: ``C24030``'s male-block lines in published order: a sector, or a subtotal.
C24030_ORDER = (3, 6, 7, 8, 9, 10, 13, 14, 17, 21, 24, 27, 28)


def industry_codes(area: StudyArea) -> dict[int, tuple[str, ...]]:
    """``INDP`` codes per :data:`C24030_SECTORS` sector, from the data dictionary.

    Raises if a civilian industry code falls in no sector or two, or if its
    label prefix (:data:`INDP_TO_C24030_CELL`, the default set's reading) puts
    it under a different published line.
    """
    labels = variable_labels(area, "INDP")
    parent = {child: sub for sub, (_, children) in C24030_SUBTOTALS.items() for child in children}
    codes: dict[int, list[str]] = {order: [] for order in C24030_SECTORS}
    for code, label in zip(labels["lo"], labels["label"]):
        prefix = str(label)[:3]
        if prefix not in INDP_TO_C24030_CELL:  # military, unemployed, N/A
            continue
        value = int(code)
        sectors = [order for order, (_, ranges) in C24030_SECTORS.items()
                   if any(lo <= value <= hi for lo, hi in ranges)]
        if len(sectors) != 1:
            raise ValueError(f"INDP {code} ({label}) falls in sector(s) {sectors}")
        order = sectors[0]
        if INDP_TO_C24030_CELL[prefix] not in (order, parent.get(order)):
            raise ValueError(f"INDP {code} ({label}): sector {order}, but its prefix maps to "
                             f"{INDP_TO_C24030_CELL[prefix]}")
        codes[order].append(str(code))
    return {order: tuple(c) for order, c in codes.items()}


def sex_by_industry(area: StudyArea, geography: str = "block group") -> ConstraintTable:
    """``C24030``: every industry line by sex, over the civilian employed. 55 cells."""
    codes = industry_codes(area)

    def sector(sex: str, order: int, offset: int) -> Line:
        def select(p, c=codes[order], code=SEX_CODES[sex]):
            return (p["SEX"] == code) & p["ESR"].isin(ESR_CIVILIAN_EMPLOYED) & p["INDP"].isin(c)
        return Line(order + offset, C24030_SECTORS[order][0], select)

    def block(sex: str, offset: int) -> tuple[Line, ...]:
        lines = []
        for order in C24030_ORDER:
            if order in C24030_SUBTOTALS:
                label, children = C24030_SUBTOTALS[order]
                lines.append(Line(order + offset, label,
                                  children=tuple(sector(sex, c, offset) for c in children)))
            else:
                lines.append(sector(sex, order, offset))
        return tuple(lines)

    root = Line(1, "total", children=(
        Line(2, "male", children=block("male", 0)),
        Line(29, "female", children=block("female", 27)),
    ))
    return tree_table("C24030", "person", ("ESR", "SEX", "INDP"), geography, root, 55)


def marital_status(geography: str = "block group") -> ConstraintTable:
    """``B12001``: sex by marital status, with the now-married subtotals. 19 cells."""
    def status(sex: str, key: str) -> Select:
        return lambda p: (p["SEX"] == SEX_CODES[sex]) & (p["MSP"] == B12001_STATUS[key][0])

    def block(sex: str, first: int) -> Line:
        return Line(first, sex, children=(
            Line(first + 1, "never_married", status(sex, "never_married")),
            Line(first + 2, "now_married", children=(
                Line(first + 3, "spouse_present", status(sex, "married_spouse_present")),
                Line(first + 4, "spouse_absent", children=(
                    Line(first + 5, "separated", status(sex, "separated")),
                    Line(first + 6, "other", status(sex, "married_spouse_absent_other")),
                )),
            )),
            Line(first + 7, "widowed", status(sex, "widowed")),
            Line(first + 8, "divorced", status(sex, "divorced")),
        ))

    root = Line(1, "total", children=(block("male", 2), block("female", 11)))
    return tree_table("B12001", "person", ("SEX", "MSP"), geography, root, 19)


def educational_attainment(geography: str = "block group") -> ConstraintTable:
    """``B15002``: sex by educational attainment, 25 and over. 35 cells."""
    def block(sex: str, first: int) -> Line:
        return Line(first, sex, children=tuple(
            Line(first + offset, label,
                 lambda p, c=codes, s=SEX_CODES[sex]: (
                     (_age(p) >= 25) & (p["SEX"] == s) & p["SCHL"].isin(c)))
            for label, (codes, offset) in B15002_ATTAINMENT.items()
        ))

    root = Line(1, "total", children=(block("male", 2), block("female", 19)))
    return tree_table("B15002", "person", ("AGEP", "SEX", "SCHL"), geography, root, 35)


def heating_fuel(geography: str = "block group") -> ConstraintTable:
    """``B25040``: house heating fuel. 10 cells."""
    leaves = tuple(
        Line(int(cell.rsplit("_", 1)[1]), label, lambda h, c=code: h["HFL"] == c)
        for label, (code, cell) in HFL_TO_CELL.items()
    )
    return tree_table("B25040", "household", ("HFL",), geography,
                      Line(1, "total", children=leaves), 10)


def all_heldout_tables(area: StudyArea, geography: str) -> list[ConstraintTable]:
    """The held-out tables at one geography, every cell (:data:`HELDOUT_RELATION`)."""
    tables = [
        tenure_by_vehicles(geography),
        poverty_ratio(geography),
        sex_by_industry(area, geography),
        marital_status(geography),
        educational_attainment(geography),
        heating_fuel(geography),
    ]
    assert {t.table for t in tables} == set(HELDOUT_RELATION)
    return tables


# -- choosing a set ------------------------------------------------------------


def constraint_tables(area: StudyArea, tables: str = "default") -> list[ConstraintTable]:
    """The constraint set named by ``tables`` (:data:`TABLE_SETS`)."""
    from pmedm_vb.assemble.constraints import default_tables

    if tables not in TABLE_SETS:
        raise ValueError(f"tables must be one of {TABLE_SETS}, got {tables!r}")
    return default_tables(area) if tables == "default" else all_tables(area)


def heldout_set(area: StudyArea, geography: str, tables: str = "default") -> list[ConstraintTable]:
    """The held-out set named by ``tables`` at one geography."""
    from pmedm_vb.assemble.constraints import heldout_tables

    if tables not in TABLE_SETS:
        raise ValueError(f"tables must be one of {TABLE_SETS}, got {tables!r}")
    return heldout_tables(area, geography) if tables == "default" else all_heldout_tables(
        area, geography)
