"""The embedding's target arrays: every statewide unit's counts in every target
category, built once from a configuration and cached.

One row per **unit** (an occupied household or a group-quarters person, as
:func:`~pmedm_vb.assemble.households.build_units` defines them), one column per
target category. A person-universe table contributes the number of the unit's
members in each category; a household-universe table or an attribute
contributes the unit's 0/1 indicator. Because the categories of every target
partition its universe (checked here, as step 1 checked the added tables), a
unit's row sum within a table is its **universe count** -- the number of trials
of that table's multinomial -- and a unit outside the universe has a zero row,
which the loss then ignores. Household tables drop out for group-quarters units
that way, with no special case.

**Identical rows are merged.** The encoder-free baseline gives every row its
own free position, so two units with the same counts in every category and the
same unit type would get the same position anyway; fitting them once, with
their weights summed, gives the same answer for less work. The map from each
unit (``SERIALNO``) to its merged row is kept, so positions can be handed back
per unit.

The arrays are written under ``$PMEDM_VB_DATA/processed/embed/targets/``, in a
directory named after the configuration and a hash of what determines them
(area, vintage and the ordered target list), so a configuration that changes
its targets never reads a stale cache. Building reads only cached PUMS files,
so it runs offline (``PMEDM_VB_OFFLINE=1``) once the prefetch has run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.assemble.constraints import ConstraintTable
from pmedm_vb.assemble.households import build_units
from pmedm_vb.config import ensure_dir, processed_dir
from pmedm_vb.embed.config import EmbedConfig, resolve, table_kind
from pmedm_vb.progress import logger, save_npz, stage, write_text

#: ``TYPEHUGQ`` codes of the two group-quarters unit types, in the order of the
#: fixed indicator columns of φ.
GQ_TYPES = {"institutional": "2", "noninstitutional": "3"}

#: Format of the cached arrays; bump when their layout changes.
TARGETS_FORMAT = 1


@dataclass
class TargetArrays:
    """Merged target counts, weights, unit types and the unit map.

    Attributes
    ----------
    counts:
        ``(rows, K)`` category counts of each merged row (float32).
    weight:
        ``(rows,)`` summed design weights of the units merged into each row.
    n_units:
        ``(rows,)`` number of units merged into each row -- the weight of the
        unweighted diagnostic fit.
    gq:
        ``(rows, 2)`` 0/1 unit-type indicators (institutional,
        noninstitutional group quarters); zero for households.
    groups:
        ``(G, 2)`` start and end column of each target in ``counts``.
    units:
        One row per unit, indexed by ``SERIALNO``: ``row`` (its merged row),
        ``weight`` and ``gq_type``.
    meta:
        Names and order of targets and categories, universes, hash, sizes.
    """

    counts: np.ndarray
    weight: np.ndarray
    n_units: np.ndarray
    gq: np.ndarray
    groups: np.ndarray
    units: pd.DataFrame
    meta: dict

    @property
    def targets(self) -> list[str]:
        return [g["id"] for g in self.meta["targets"]]

    @property
    def categories(self) -> list[str]:
        """``"{target}.{category}"`` per column."""
        return [f"{g['id']}.{c}" for g in self.meta["targets"] for c in g["categories"]]


def targets_key(config: EmbedConfig) -> str:
    """Hash of everything that determines the arrays."""
    payload = {
        "format": TARGETS_FORMAT,
        "area": [config.area.state, config.area.year, config.area.span],
        "targets": list(config.targets.ids),
    }
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:12]


def targets_dir(config: EmbedConfig) -> Path:
    return processed_dir() / "embed" / "targets" / f"{config.name}_{targets_key(config)}"


def _table_counts(
    table: ConstraintTable,
    units: pd.DataFrame,
    persons: pd.DataFrame,
    person_row: np.ndarray,
) -> np.ndarray:
    """``(units, categories)`` counts of one target, checking that no record is
    claimed by two of its categories (which would double-count it in the
    multinomial)."""
    source = persons if table.universe == "person" else units
    hits = np.zeros(len(source), dtype=np.int16)
    out = np.zeros((len(units), len(table.categories)), dtype=np.float32)
    for j, category in enumerate(table.categories):
        mask = category.select(source)
        mask = mask.fillna(False).to_numpy(dtype=bool) if isinstance(mask, pd.Series) \
            else np.asarray(mask, dtype=bool)
        hits += mask
        if table.universe == "person":
            out[:, j] = np.bincount(person_row[mask], minlength=len(units))
        else:
            out[:, j] = mask
    several = int((hits > 1).sum())
    if several:
        raise ValueError(
            f"{table.table}: {several} record(s) fall in two or more categories; "
            f"its categories must partition its universe for the per-table "
            f"multinomial"
        )
    return out


def build_targets(config: EmbedConfig) -> TargetArrays:
    """Build the target arrays from the PUMS (no caching; see :func:`load_targets`)."""
    area = config.area
    with stage(f"resolve {len(config.targets.ids)} targets"):
        tables = resolve(area, config.targets.ids)
    with stage("build statewide units") as step:
        frame = build_units(area, None, list(tables.values()))
        step.detail = f"{frame.n_units:,} units, {frame.n_group_quarters:,} group quarters"
    units, persons = frame.units, frame.persons
    person_row = units.index.get_indexer(persons["SERIALNO"])
    if (person_row < 0).any():
        raise ValueError("person records whose SERIALNO is not a unit")

    blocks, meta_targets, start = [], [], 0
    with stage("count target categories") as step:
        for target_id, table in tables.items():
            block = _table_counts(table, units, persons, person_row)
            blocks.append(block)
            meta_targets.append({
                "id": target_id,
                "kind": table_kind(target_id),
                "universe": table.universe,
                "categories": [c.name for c in table.categories],
                "start": start,
                "end": start + block.shape[1],
            })
            start += block.shape[1]
        counts = np.concatenate(blocks, axis=1)
        step.detail = f"{counts.shape[1]} categories in {len(tables)} targets"

    gq_code = units["TYPEHUGQ"].astype("string")
    gq = np.column_stack([(gq_code == code).fillna(False).to_numpy(dtype=bool)
                          for code in GQ_TYPES.values()]).astype(np.float32)
    weight = units["weight"].to_numpy(dtype=np.float64)

    with stage("merge identical units") as step:
        if counts.max() > np.iinfo(np.int16).max:
            raise ValueError("a category count exceeds int16")
        key = np.concatenate([counts.astype(np.int16), gq.astype(np.int16)], axis=1)
        _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.ravel()
        rows = len(first)
        merged_weight = np.bincount(inverse, weights=weight, minlength=rows)
        n_units = np.bincount(inverse, minlength=rows).astype(np.float64)
        step.detail = f"{len(units):,} units -> {rows:,} rows"

    unit_frame = pd.DataFrame({
        "row": inverse.astype(np.int64),
        "weight": weight,
        "gq_type": np.select(
            [gq[:, 0] == 1, gq[:, 1] == 1], list(GQ_TYPES), default="household"),
    }, index=units.index)

    meta = {
        "format": TARGETS_FORMAT,
        "name": config.name,
        "key": targets_key(config),
        "area": {"state": area.state, "year": area.year, "span": area.span},
        "n_units": int(len(units)),
        "n_group_quarters": int(frame.n_group_quarters),
        "n_rows": int(rows),
        "n_categories": int(counts.shape[1]),
        "gq_types": list(GQ_TYPES),
        "targets": meta_targets,
    }
    return TargetArrays(
        counts=counts[first],
        weight=merged_weight,
        n_units=n_units,
        gq=gq[first],
        groups=np.array([[t["start"], t["end"]] for t in meta_targets], dtype=np.int64),
        units=unit_frame,
        meta=meta,
    )


def save_targets(arrays: TargetArrays, directory: Path) -> None:
    ensure_dir(directory)
    save_npz(directory / "arrays.npz", counts=arrays.counts, weight=arrays.weight,
             n_units=arrays.n_units, gq=arrays.gq, groups=arrays.groups)
    units = arrays.units.reset_index()
    write_text(directory / "units.csv", units.to_csv(index=False))
    # Metadata last: its presence marks a complete directory.
    write_text(directory / "meta.json", json.dumps(arrays.meta, indent=1))


def read_targets(directory: Path) -> TargetArrays:
    meta = json.loads((directory / "meta.json").read_text())
    with np.load(directory / "arrays.npz") as data:
        arrays = {k: data[k] for k in data.files}
    units = pd.read_csv(directory / "units.csv", dtype={"SERIALNO": str}).set_index("SERIALNO")
    return TargetArrays(meta=meta, units=units, **arrays)


def load_targets(config: EmbedConfig, *, rebuild: bool = False) -> TargetArrays:
    """The configuration's target arrays, from the cache or built and cached."""
    directory = targets_dir(config)
    if not rebuild and (directory / "meta.json").exists():
        with stage(f"read targets {directory.name}"):
            return read_targets(directory)
    arrays = build_targets(config)
    with stage(f"save targets {directory}"):
        save_targets(arrays, directory)
    logger.info("targets: %s units, %s rows, %s categories, %.0f MB",
                f"{arrays.meta['n_units']:,}", f"{arrays.meta['n_rows']:,}",
                arrays.meta["n_categories"], arrays.counts.nbytes / 2**20)
    return arrays
