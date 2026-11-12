"""The embedding's configuration: which tables and attributes are targets,
which are inputs, and the fitting settings.

A configuration is a JSON file::

    {
      "name": "...",
      "area": {"state": "47", "year": 2024, "span": 5},
      "targets": {"tables": [...], "attributes": [...]},
      "inputs":  {"tables": [...], "attributes": [...]},
      "fit": {"design_weights": true, "gamma": [...], "beta": 1.0,
              "d": [...], "ridge": null, "encoder": "none"}
    }

**Targets** are what the position φ must predict through the decoder;
**inputs** are what an encoder reads to compute φ. Most items are both. The
encoder-free baseline has no encoder, so its inputs are unused.

Table IDs resolve to the builders that already define them, so a table means
the same categories here as wherever else the repo uses it:

* the PMEDM constraint set (:func:`~pmedm_vb.assemble.constraints.default_tables`),
  at block group where it is constrained at both levels;
* the held-out tables (:func:`~pmedm_vb.assemble.constraints.heldout_tables`,
  and ``B17010A``-``I`` from
  :func:`~pmedm_vb.assemble.constraints.race_poverty`, whose cells are found
  in the tract VRE files and so need those cached);
* the tables added for the embedding (:data:`pmedm_vb.embed.tables.TABLES`).

Attributes are PUMS variables with no published table
(:data:`pmedm_vb.embed.attributes.ATTRIBUTES`).

``fit`` carries the settings agreed for stage 1: design weights on; the
rare-category exponent γ (a list is a grid to compare); β, the weight on
targets outside the tables, 1; the dimensions d to try; the ridge penalty
values to try; and the encoder (``"none"`` for the baseline). They record the
agreed grid and are checked here; ``experiments/embed/fit_baseline.py`` takes
the values to fit explicitly on its command line
(``experiments/embed/submit_baseline_grid.sh`` passes this grid).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pmedm_vb.assemble.constraints import (
    RACE_POVERTY_TABLES,
    ConstraintTable,
    default_table_ids,
    default_tables,
    heldout_tables,
    race_poverty,
)
from pmedm_vb.config import StudyArea
from pmedm_vb.embed.attributes import ATTRIBUTES, build_attribute
from pmedm_vb.embed.tables import TABLES, build_table

#: Held-out tables of :func:`heldout_tables`, by ID.
HELDOUT_TABLE_IDS = ("B25044", "C17002", "C24030", "B12001", "B15002", "B25040")

#: Encoders a configuration may name.
ENCODERS = ("none", "deep_set")


def table_kind(table: str) -> str:
    """``"constraint"``, ``"heldout"``, ``"added"`` or ``"attribute"``."""
    if table in default_table_ids():
        return "constraint"
    if table in HELDOUT_TABLE_IDS or table in RACE_POVERTY_TABLES:
        return "heldout"
    if table in TABLES:
        return "added"
    if table in ATTRIBUTES:
        return "attribute"
    raise KeyError(
        f"unknown table or attribute {table!r}. Known: constraint "
        f"{list(default_table_ids())}, held-out "
        f"{[*HELDOUT_TABLE_IDS, *RACE_POVERTY_TABLES]}, added {sorted(TABLES)}, "
        f"attributes {sorted(ATTRIBUTES)}"
    )


@dataclass(frozen=True)
class Selection:
    """Tables and attributes in one role."""

    tables: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()

    @property
    def ids(self) -> tuple[str, ...]:
        return (*self.tables, *self.attributes)


@dataclass(frozen=True)
class FitSettings:
    design_weights: bool = True
    gamma: tuple[float, ...] = (0.5,)
    beta: float = 1.0
    d: tuple[int, ...] = (8,)
    ridge: tuple[float, ...] | None = None
    encoder: str = "none"


@dataclass(frozen=True)
class EmbedConfig:
    name: str
    area: StudyArea
    targets: Selection
    inputs: Selection
    fit: FitSettings = field(default_factory=FitSettings)
    path: Path | None = None

    @property
    def ids(self) -> tuple[str, ...]:
        """Every table and attribute in either role, targets first, once each."""
        return tuple(dict.fromkeys((*self.targets.ids, *self.inputs.ids)))

    @property
    def table_ids(self) -> tuple[str, ...]:
        """Published tables in either role (attributes excluded)."""
        return tuple(i for i in self.ids if table_kind(i) != "attribute")


def _as_tuple(value, kind, name):
    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    try:
        return tuple(kind(v) for v in values)
    except (TypeError, ValueError):
        raise ValueError(f"fit.{name}: expected {kind.__name__} or a list of them, "
                         f"got {value!r}") from None


def _selection(raw: dict, role: str, where: str) -> Selection:
    block = raw.get(role) or {}
    unknown = set(block) - {"tables", "attributes"}
    if unknown:
        raise ValueError(f"{where}: {role} has unknown keys {sorted(unknown)}")
    tables = tuple(block.get("tables", ()))
    attributes = tuple(block.get("attributes", ()))
    for label, items, allowed in (
        ("tables", tables, ("constraint", "heldout", "added")),
        ("attributes", attributes, ("attribute",)),
    ):
        duplicates = sorted({i for i in items if items.count(i) > 1})
        if duplicates:
            raise ValueError(f"{where}: {role}.{label} lists {duplicates} twice")
        for item in items:
            kind = table_kind(item)
            if kind not in allowed:
                raise ValueError(f"{where}: {item!r} is a {kind} entry, not "
                                 f"allowed under {role}.{label}")
    return Selection(tables=tables, attributes=attributes)


def load_config(path: str | Path) -> EmbedConfig:
    """Read and check an embedding configuration."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    where = str(path)
    unknown = set(raw) - {"name", "area", "targets", "inputs", "fit"}
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}")

    area_raw = raw.get("area") or {}
    area = StudyArea(
        name=str(area_raw.get("name", "state")),
        state=str(area_raw["state"]).zfill(2),
        year=int(area_raw["year"]),
        span=int(area_raw.get("span", 5)),
    )

    fit_raw = raw.get("fit") or {}
    unknown = set(fit_raw) - {"design_weights", "gamma", "beta", "d", "ridge", "encoder"}
    if unknown:
        raise ValueError(f"{where}: fit has unknown keys {sorted(unknown)}")
    fit = FitSettings(
        design_weights=bool(fit_raw.get("design_weights", True)),
        gamma=_as_tuple(fit_raw.get("gamma", 0.5), float, "gamma"),
        beta=float(fit_raw.get("beta", 1.0)),
        d=_as_tuple(fit_raw.get("d", 8), int, "d"),
        ridge=_as_tuple(fit_raw.get("ridge"), float, "ridge"),
        encoder=str(fit_raw.get("encoder", "none")),
    )
    if not all(0.0 <= g <= 1.0 for g in fit.gamma):
        raise ValueError(f"{where}: fit.gamma must lie in [0, 1], got {fit.gamma}")
    if fit.beta <= 0:
        raise ValueError(f"{where}: fit.beta must be positive, got {fit.beta}")
    if not all(d > 0 for d in fit.d):
        raise ValueError(f"{where}: fit.d must be positive, got {fit.d}")
    if fit.ridge is not None and not all(r >= 0 for r in fit.ridge):
        raise ValueError(f"{where}: fit.ridge must be non-negative, got {fit.ridge}")
    if fit.encoder not in ENCODERS:
        raise ValueError(f"{where}: fit.encoder must be one of {ENCODERS}, "
                         f"got {fit.encoder!r}")

    targets = _selection(raw, "targets", where)
    if not targets.ids:
        raise ValueError(f"{where}: no targets")
    return EmbedConfig(
        name=str(raw.get("name", path.stem)),
        area=area,
        targets=targets,
        inputs=_selection(raw, "inputs", where),
        fit=fit,
        path=path,
    )


def resolve(area: StudyArea, ids) -> dict[str, ConstraintTable]:
    """Build the :class:`ConstraintTable` of every table or attribute in ``ids``.

    Constraint tables come from :func:`default_tables`, taking the block-group
    version of the six constrained at both levels (the categories are the
    same at both); held-out tables from :func:`heldout_tables` at block group,
    and ``B17010A``-``I`` at tract, where alone they are published with
    replicates.
    """
    ids = tuple(dict.fromkeys(ids))
    built: dict[str, ConstraintTable] = {}

    constraint = [i for i in ids if table_kind(i) == "constraint"]
    if constraint:
        for table in default_tables(area, only=tuple(constraint)):
            built.setdefault(table.table, table)

    heldout = [i for i in ids if i in HELDOUT_TABLE_IDS]
    if heldout:
        for table in heldout_tables(area, "block group"):
            if table.table in heldout:
                built[table.table] = table

    for table_id in ids:
        if table_id in built:
            continue
        kind = table_kind(table_id)
        if table_id in RACE_POVERTY_TABLES:
            built[table_id] = race_poverty(area, table_id[-1], "tract")
        elif kind == "added":
            built[table_id] = build_table(table_id)
        elif kind == "attribute":
            built[table_id] = build_attribute(table_id)

    missing = [i for i in ids if i not in built]
    if missing:
        raise RuntimeError(f"no builder produced {missing}")
    return {i: built[i] for i in ids}
