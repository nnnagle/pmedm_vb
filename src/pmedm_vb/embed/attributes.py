"""PUMS attributes the embedding predicts that no block-group or tract table
publishes.

Each is a :class:`~pmedm_vb.assemble.constraints.ConstraintTable` whose
``table`` is the PUMS variable name and whose categories have **no published
cells**: there is nothing to constrain or score at small-area level, so the
cell correspondence is empty by design, not unresolved. Representing them as
tables lets them share the unit frame, the column loading and the per-unit
counts with the published tables.

Codes are from the 2020-2024 PUMS data dictionary. The categories partition
occupied housing units: blanks among the units carried are exactly the group
quarters, and where a blank also means "not this kind of household" (``CPLT``)
that is a category of its own, tested with ``.isna()`` on occupied units.

* ``MULTG`` -- multigenerational household. ``B11017`` publishes it only for
  the nation and states (2024 Data Product List).
* ``CPLT`` -- couple type: opposite- or same-sex, married or unmarried
  partners, or no couple.
* ``HUPAC`` -- presence and age of children (under 6 only, 6 to 17 only, both,
  none).
* ``MV`` -- when the household moved in, as a duration band. Carried in place
  of ``B25038``, whose calendar-year bands the PUMS cannot reproduce
  (:mod:`pmedm_vb.embed.tables`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pmedm_vb.assemble.constraints import Category, ConstraintTable
from pmedm_vb.embed.tables import Universe, occupied

#: Code -> category name, per attribute, in dictionary order.
CODES = {
    "MULTG": {"1": "not_multigenerational", "2": "multigenerational"},
    "CPLT": {
        "1": "opposite_sex_spouse",
        "2": "same_sex_spouse",
        "3": "opposite_sex_unmarried_partner",
        "4": "same_sex_unmarried_partner",
    },
    "HUPAC": {
        "1": "children_under_6_only",
        "2": "children_6_to_17_only",
        "3": "children_under_6_and_6_to_17",
        "4": "no_children",
    },
    "MV": {
        "1": "12_months_or_less",
        "2": "13_to_23_months",
        "3": "2_to_4_years",
        "4": "5_to_9_years",
        "5": "10_to_19_years",
        "6": "20_to_29_years",
        "7": "30_years_or_more",
    },
}

#: Attributes whose blank, on an occupied unit, is itself a category: ``CPLT``
#: is blank for "non-couple households" as well as group quarters.
BLANK_CATEGORY = {"CPLT": "no_couple"}


@dataclass(frozen=True)
class AttributeSpec:
    build: Callable[[], ConstraintTable]
    universe: Universe
    universe_title: str


def attribute_table(variable: str) -> ConstraintTable:
    """The categories of one household attribute, over occupied units."""
    categories = [
        Category(
            name=name,
            select=(lambda c: lambda u: u[variable] == c)(code),
        )
        for code, name in CODES[variable].items()
    ]
    if variable in BLANK_CATEGORY:
        categories.append(Category(
            name=BLANK_CATEGORY[variable],
            select=lambda u: occupied(u) & u[variable].isna(),
        ))
    return ConstraintTable(
        table=variable,
        # TEN identifies occupied units for the blank category.
        variables=(variable, "TEN"),
        universe="household",
        geography="none",
        categories=tuple(categories),
    )


#: The attributes, by PUMS variable name.
ATTRIBUTES: dict[str, AttributeSpec] = {
    variable: AttributeSpec(
        build=(lambda v: lambda: attribute_table(v))(variable),
        universe=occupied,
        universe_title="Occupied housing units",
    )
    for variable in CODES
}


def build_attribute(variable: str) -> ConstraintTable:
    return ATTRIBUTES[variable].build()


def is_attribute(name: str) -> bool:
    return name in ATTRIBUTES

