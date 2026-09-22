"""The unit frame: households and group-quarters persons, with their weights.

A *unit* is the row of ``X`` and the thing a zone's weight lands on. It is a
household or a group-quarters person, and the two are not interchangeable:

* An occupied housing unit is weighted by ``WGTP`` and holds however many
  people its person records describe.
* A group-quarters record is weighted by ``PWGTP``, because ``WGTP`` is
  **exactly zero** on every one of them -- verified across Tennessee, 8,077
  institutional and 8,203 noninstitutional. Weighting units by ``WGTP`` alone
  would give every GQ resident zero design probability, so they could never
  receive weight in any zone, while the published total-population tables go on
  counting them.

That asymmetry is the whole reason this module exists. The published structure
makes it cheap to resolve: a GQ housing record corresponds to exactly one
person record -- 16,280 of each in Tennessee -- so a GQ unit *is* a one-person
household, and the same person-to-unit aggregation works for both kinds
without a special case beyond the weight.

Vacant units are dropped. They hold nobody, so they contribute nothing to a
person-level constraint, and they are outside the occupied-housing-units
universe of the household-level ones. They are identified by a null ``TEN``
within ``TYPEHUGQ == "1"`` -- null ``TEN`` alone will not do, since the
published code ``b`` means "group quarters *or* vacant".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pmedm_vb.assemble.constraints import (
    ConstraintTable,
    TYPEHUGQ_GROUP_QUARTERS,
    required_variables,
)
from pmedm_vb.config import StudyArea
from pmedm_vb.data.pums import WEIGHT_PREFIXES, load_pums

#: Loaded whatever the constraints declare. ``TYPEHUGQ`` separates households
#: from group quarters and ``TEN`` separates occupied from vacant; without both
#: the unit set cannot be formed at all.
STRUCTURAL_COLUMNS = ("TYPEHUGQ", "TEN")

#: An ordinary housing unit, against the group-quarters codes.
TYPEHUGQ_HOUSING_UNIT = "1"


@dataclass(frozen=True)
class Units:
    """One PUMA's units and the person records belonging to them.

    Attributes
    ----------
    units:
        One row per unit, indexed by ``SERIALNO``, carrying ``weight`` (from
        ``WGTP`` or ``PWGTP`` as appropriate), ``is_group_quarters``, and every
        household-universe variable the constraints declared.
    persons:
        Person records for those units only, with ``SERIALNO`` to aggregate on.
        Includes the single occupant of each GQ unit, so person-level
        constraints see group-quarters residents.
    puma:
        The PUMA these units lie in.
    """

    units: pd.DataFrame
    persons: pd.DataFrame
    puma: str
    #: Person records in the PUMA whose ``SERIALNO`` matched no unit, and so
    #: were dropped. Counted *before* the filter, because afterwards the
    #: question cannot be asked -- which is how an earlier version of this
    #: check managed to be vacuous.
    dropped_persons: pd.DataFrame

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_group_quarters(self) -> int:
        return int(self.units["is_group_quarters"].sum())

    def validate(self) -> None:
        """Raise if the frames do not describe one consistent unit set."""
        problems = []
        if not self.units.index.is_unique:
            problems.append("units index (SERIALNO) is not unique")
        if not np.all(self.units["weight"] >= 0):
            problems.append("negative weights")
        if self.units["weight"].sum() <= 0:
            problems.append("weights sum to zero; no unit can receive any weight")
        if len(self.dropped_persons):
            serials = self.dropped_persons["SERIALNO"]
            gq = int(serials.str.contains("GQ").sum())
            problems.append(
                f"{serials.nunique()} person record(s) in this PUMA belong to no "
                f"unit and were dropped ({gq} of them group quarters, by SERIALNO). "
                f"Every person should sit in an occupied housing unit or a "
                f"group-quarters record; any that do not are missing from the "
                f"model while the published tables still count them"
            )
        gq = self.units.index[self.units["is_group_quarters"]]
        if len(gq):
            occupants = self.persons[self.persons["SERIALNO"].isin(set(gq))]
            counts = occupants.groupby("SERIALNO").size()
            if len(counts) != len(gq) or not (counts == 1).all():
                problems.append(
                    "group-quarters units are not one person each, so the "
                    "PWGTP weight does not describe the unit"
                )
        if problems:
            raise ValueError("inconsistent unit frame:\n  " + "\n  ".join(problems))


def build_units(
    area: StudyArea,
    puma: str,
    tables: Sequence[ConstraintTable],
) -> Units:
    """Assemble one PUMA's unit frame from the PUMS files.

    Columns are taken from what the constraint specifications declare, plus
    :data:`STRUCTURAL_COLUMNS`. The PUMS replicate weights are not read: PMEDM
    takes its covariance from the published tables, so those 80 columns are
    pure cost here.

    Raises
    ------
    ValueError
        If the PUMA holds no units, or if a group-quarters record has anything
        other than exactly one person -- the latter would mean ``PWGTP`` is not
        the unit's weight and the whole GQ treatment is unsound.
    """
    needed = required_variables(tables)

    housing = load_pums(
        area,
        [*needed["household"], *STRUCTURAL_COLUMNS],
        record_type="housing",
        replicate_weights=False,
    )
    persons = load_pums(
        area,
        list(needed["person"]),
        record_type="person",
        replicate_weights=False,
    )

    housing = housing[housing["puma_geoid"] == puma]
    persons = persons[persons["puma_geoid"] == puma]

    is_gq = housing["TYPEHUGQ"].isin(TYPEHUGQ_GROUP_QUARTERS)
    # Occupied means a housing unit with a tenure. Testing TEN alone would keep
    # vacants out *and* group quarters, since the published "b" covers both.
    occupied = (housing["TYPEHUGQ"] == TYPEHUGQ_HOUSING_UNIT) & housing["TEN"].notna()

    units = housing[is_gq | occupied].copy()
    if units.empty:
        raise ValueError(f"PUMA {puma!r} has no occupied or group-quarters records")
    units = units.set_index("SERIALNO")
    units["is_group_quarters"] = units["TYPEHUGQ"].isin(TYPEHUGQ_GROUP_QUARTERS)

    belongs = persons["SERIALNO"].isin(set(units.index))
    dropped = persons[~belongs].copy()
    persons = persons[belongs].copy()

    # A GQ unit's weight is its occupant's. Checked rather than assumed: if a GQ
    # record ever held more than one person, PWGTP would not be the unit weight.
    gq_index = units.index[units["is_group_quarters"]]
    occupants = persons[persons["SERIALNO"].isin(set(gq_index))]
    counts = occupants.groupby("SERIALNO").size()
    if len(counts) != len(gq_index) or not (counts == 1).all():
        raise ValueError(
            f"{(counts != 1).sum()} group-quarters record(s) in PUMA {puma} do not "
            f"hold exactly one person, and {len(gq_index) - len(counts)} hold none; "
            f"PWGTP is then not the unit's weight"
        )

    gq_weight = occupants.set_index("SERIALNO")[WEIGHT_PREFIXES["person"]]
    units["weight"] = np.where(
        units["is_group_quarters"],
        units.index.map(gq_weight).to_numpy(),
        pd.to_numeric(units[WEIGHT_PREFIXES["housing"]], errors="coerce").to_numpy(),
    )

    frame = Units(
        units=units, persons=persons, puma=str(puma), dropped_persons=dropped
    )
    frame.validate()
    return frame


def person_counts(
    frame: Units,
    tables: Sequence[ConstraintTable],
) -> pd.DataFrame:
    """Per-unit counts for every person-universe category, and 0/1 for unit ones.

    The columns of ``X``, before they are made sparse. A person-universe
    category counts matching persons within the unit; a unit-universe one is
    the indicator, which is the same arithmetic with one row per unit.

    Column names are ``"{table}.{category}"``, matching
    :attr:`~pmedm_vb.assemble.targets.TargetBlock.names` so ``X`` and ``Y``
    line up by name rather than by position.
    """
    columns: dict[str, pd.Series] = {}
    index = frame.units.index

    for table in tables:
        source = frame.persons if table.universe == "person" else frame.units
        for category in table.categories:
            selected = category.select(source)
            if table.universe == "person":
                counts = (
                    frame.persons.loc[selected.to_numpy(), "SERIALNO"]
                    .value_counts()
                    .reindex(index, fill_value=0)
                )
            else:
                counts = selected.reindex(index, fill_value=False).astype(int)
            columns[f"{table.table}.{category.name}"] = counts.astype(float)

    return pd.DataFrame(columns, index=index)
