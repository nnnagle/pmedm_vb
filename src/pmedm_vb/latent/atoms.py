"""Atoms and their contributions (step 1, pieces A1 and A2 of ``PLAN_latent.md``).

**Atoms (A1).** The statewide PUMS units ``i = 1..M`` (households and
group-quarters persons, as :func:`~pmedm_vb.assemble.households.build_units`
defines them), each with its stage 1 position ``z_i``, design weight ``w_i``
(``WGTP``, or the occupant's ``PWGTP`` for group quarters), its 80 replicate
weights, its PUMA and its unit type. The statewide weights are
``q_i = w_i / Σ_j w_j`` (``MODEL_latent.qmd``). The units, weights, replicates
and PUMAs come from the leave-one-PUMA-out check's loader
(:func:`~pmedm_vb.embed.puma_check.load_check_data`), the positions from the
stage 1 run (:func:`~pmedm_vb.embed.puma_check.load_phi`), aligned by
``SERIALNO``. The atoms' order is the check's unit order throughout step 1.

**Contributions (A2).** ``T`` (``M × J``, sparse) has row ``t(x_i)'``: the
unit's contributions to every published cell that
:func:`~pmedm_vb.assemble.build.build_puma` builds -- the constraint tables of
:func:`~pmedm_vb.assemble.allcells.constraint_tables` at tract and block group,
then the held-out tables of :func:`~pmedm_vb.assemble.allcells.heldout_set` at
tract and block group -- computed over the statewide units by the same code
``build_puma`` uses (:func:`~pmedm_vb.assemble.households.person_counts`,
:func:`~pmedm_vb.assemble.design.build_attribute_matrix`). A published table
appears once per geography it is published at, so a cell name is
``"<level>:<set>:<table>.<category>"`` (level ``tract`` or ``block group``,
set ``constraint`` or ``heldout``); the part after the second colon is
``build_puma``'s own column name. The columns are built table by table, which
gives the same numbers as building them all at once (each column depends on
its own category only) without a dense ``M × J`` frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.design import build_attribute_matrix
from pmedm_vb.assemble.households import build_units, person_counts
from pmedm_vb.config import StudyArea
from pmedm_vb.embed.puma_check import CheckData, load_phi
from pmedm_vb.embed.units import GQ_TYPES
from pmedm_vb.progress import save_npz, stage, write_text

#: Indicator columns of φ, in the order :func:`~pmedm_vb.embed.puma_check.load_phi` gives them.
GQ_COLUMNS = ("gq_institutional", "gq_noninstitutional")

#: Unit types by ``TYPEHUGQ`` code.
UNIT_TYPES = {"1": "household", **{code: name for name, code in GQ_TYPES.items()}}

#: Levels and sets of ``T``'s columns, in their stacked order.
T_BLOCKS = (("tract", "constraint"), ("block group", "constraint"),
            ("tract", "heldout"), ("block group", "heldout"))


@dataclass
class Atoms:
    """The statewide atoms ``(z_i, x_i)`` with their weights.

    Attributes
    ----------
    serial: ``SERIALNO`` of each atom (the check's unit order).
    z: ``(M, d)`` stage 1 positions (whitened coordinates ``z1..zd``).
    gq: ``(M, 2)`` institutional and noninstitutional group-quarters indicators.
    weight: ``(M,)`` design weights ``w_i``.
    replicates: ``(M, 80)`` replicate weights.
    puma: ``(M,)`` PUMA GEOID of each atom.
    unit_type: ``(M,)`` ``household``, ``institutional`` or ``noninstitutional``.
    z_columns: names of ``z``'s columns.
    """

    serial: pd.Index
    z: np.ndarray
    gq: np.ndarray
    weight: np.ndarray
    replicates: np.ndarray
    puma: np.ndarray
    unit_type: np.ndarray
    z_columns: list[str]

    @property
    def M(self) -> int:
        return len(self.serial)

    @property
    def q(self) -> np.ndarray:
        """Statewide normalized design weights ``q_i``."""
        return self.weight / self.weight.sum()

    @property
    def pumas(self) -> list[str]:
        return sorted(set(self.puma))

    def phi(self) -> tuple[np.ndarray, list[str], np.ndarray]:
        """``[gq, z]`` with column names and indicator flags, as
        :func:`~pmedm_vb.embed.puma_check.load_phi` returns them (the input of
        the decoder and quadratic feature maps)."""
        columns = [*GQ_COLUMNS, *self.z_columns]
        binary = np.array([True] * len(GQ_COLUMNS) + [False] * len(self.z_columns))
        return np.column_stack([self.gq, self.z]), columns, binary


def load_atoms(data: CheckData, run: Path) -> Atoms:
    """Atoms from the check's statewide units and the stage 1 run's positions."""
    phi, columns, binary = load_phi(Path(run), data.units.index)
    missing = [c for c in GQ_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f"{run}: positions lack the indicator columns {missing}")
    gq = phi[:, [columns.index(c) for c in GQ_COLUMNS]]
    z_idx = [j for j, c in enumerate(columns) if c not in GQ_COLUMNS]
    if binary[z_idx].any():
        raise ValueError(f"{run}: a z column is 0/1, which load_phi takes for an indicator")
    codes = data.units["TYPEHUGQ"].astype(str).to_numpy()
    unknown = sorted(set(codes) - set(UNIT_TYPES))
    if unknown:
        raise ValueError(f"unit type codes {unknown} not in {sorted(UNIT_TYPES)}")
    unit_type = np.array([UNIT_TYPES[c] for c in codes])
    expected = np.column_stack([unit_type == "institutional", unit_type == "noninstitutional"])
    if not np.array_equal(gq.astype(bool), expected):
        raise ValueError("the positions' group-quarters indicators disagree with TYPEHUGQ")
    return Atoms(serial=data.units.index, z=phi[:, z_idx], gq=gq, weight=data.weight,
                 replicates=data.replicates, puma=data.puma.astype(str), unit_type=unit_type,
                 z_columns=[columns[j] for j in z_idx])


def atom_checks(atoms: Atoms) -> dict:
    """Atom count; design-weighted mean and covariance of ``z`` (expected 0
    and ``I``); weight totals by unit type."""
    w = atoms.q
    mean = w @ atoms.z
    centred = atoms.z - mean
    cov = (centred * w[:, None]).T @ centred
    out = {
        "M": atoms.M,
        "d": int(atoms.z.shape[1]),
        "z_mean_max_abs": float(np.abs(mean).max()),
        "z_cov_max_abs_dev_from_I": float(np.abs(cov - np.eye(cov.shape[0])).max()),
        "weight_total": float(atoms.weight.sum()),
    }
    for kind in ("household", "institutional", "noninstitutional"):
        sel = atoms.unit_type == kind
        out[f"n_{kind}"] = int(sel.sum())
        out[f"weight_{kind}"] = float(atoms.weight[sel].sum())
    return out


# --------------------------------------------------------------------------
# Contributions T
# --------------------------------------------------------------------------


@dataclass
class Contributions:
    """``T`` (``M × J``, sparse), its cell names (see the module notes) and its
    rows' ``SERIALNO``."""

    T: sp.csr_matrix
    names: list[str]
    level: np.ndarray
    kind: np.ndarray
    serial: pd.Index | None = None

    @property
    def cell(self) -> list[str]:
        """``build_puma``'s column names, ``"{table}.{category}"``."""
        return [n.split(":", 2)[2] for n in self.names]

    def block(self, level: str, kind: str) -> np.ndarray:
        """Column positions of one (level, set) block."""
        return np.flatnonzero((self.level == level) & (self.kind == kind))

    def unique_cells(self) -> np.ndarray:
        """One column per distinct ``"{table}.{category}"`` (its first
        occurrence): a table published at both levels has identical columns
        at each, and PUMA-level scoring needs it once."""
        _, first = np.unique(np.array(self.cell), return_index=True)
        return np.sort(first)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_npz(directory / "T.npz", data=self.T.data, indices=self.T.indices,
                 indptr=self.T.indptr, shape=np.array(self.T.shape))
        write_text(directory / "T_cells.json", json.dumps(
            {"names": self.names, "level": self.level.tolist(), "kind": self.kind.tolist(),
             "serial": None if self.serial is None else [str(s) for s in self.serial]}))

    @classmethod
    def load(cls, directory: Path) -> Contributions:
        directory = Path(directory)
        a = np.load(directory / "T.npz")
        T = sp.csr_matrix((a["data"], a["indices"], a["indptr"]), shape=tuple(a["shape"]))
        meta = json.loads((directory / "T_cells.json").read_text())
        serial = meta.get("serial")
        return cls(T=T, names=meta["names"], level=np.array(meta["level"]),
                   kind=np.array(meta["kind"]),
                   serial=None if serial is None else pd.Index(serial, name="SERIALNO"))


def contribution_tables(area: StudyArea, tables="all") -> dict[tuple[str, str], list]:
    """The published-cell definitions ``build_puma`` and the held-out build use,
    by (level, set): ``tables`` is a :data:`~pmedm_vb.assemble.allcells.TABLE_SETS`
    name or table IDs, as ``run_map.py --tables`` takes it."""
    from pmedm_vb.assemble.allcells import constraint_tables, heldout_set

    constraint = constraint_tables(area, tables)
    out = {}
    for level, kind in T_BLOCKS:
        chosen = (constraint if kind == "constraint" else heldout_set(area, level, tables))
        out[(level, kind)] = [t for t in chosen if t.geography == level]
    return out


def build_contributions(area: StudyArea, serial: pd.Index, tables="all") -> Contributions:
    """``T`` over the statewide units, rows in ``serial``'s order."""
    by_block = contribution_tables(area, tables)
    every = [t for block in by_block.values() for t in block]
    with stage("statewide units for T") as step:
        frame = build_units(area, None, every)
        step.detail = f"{frame.n_units:,} units"
    rows = frame.units.index.get_indexer(serial)
    if (rows < 0).any() or frame.n_units != len(serial):
        raise ValueError(f"the units built for T ({frame.n_units:,}) are not the atoms "
                         f"({len(serial):,}; {int((rows < 0).sum())} not found)")
    blocks, names, level, kind = [], [], [], []
    with stage(f"T: {len(every)} tables") as step:
        for (lev, knd), block_tables in by_block.items():
            for table in block_tables:
                matrix, labels = build_attribute_matrix(person_counts(frame, [table]))
                blocks.append(matrix[rows])
                names += [f"{lev}:{knd}:{label}" for label in labels]
                level += [lev] * len(labels)
                kind += [knd] * len(labels)
        T = sp.hstack(blocks).tocsr()
        step.detail = f"{T.shape[0]:,} x {T.shape[1]:,}, {T.nnz:,} nonzeros"
    return Contributions(T=T, names=names, level=np.array(level), kind=np.array(kind),
                         serial=pd.Index(serial))
