"""Held-out tables: published cells a fit never sees, for scoring it.

A held-out table is built exactly as a constraint is -- the same
:class:`~pmedm_vb.assemble.constraints.ConstraintTable` specification, the same
per-unit counts and the same published targets and variances -- but it is kept
apart from :class:`~pmedm_vb.assemble.inputs.PMEDMInputs`, so it cannot enter
``f``. A fit's weights ``W`` predict it as ``A W X_H``, and the published
estimate and its standard error score the prediction: a truth that does not
depend on any reference posterior.

The rows of ``X_H`` follow the problem's own ``units`` order, which is checked
rather than assumed. Only the PUMA-only problem is supported: a statewide
support merges records into rows, and the held-out columns would have to be
merged the same way.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.constraints import HELDOUT_RELATION, ConstraintTable
from pmedm_vb.assemble.design import build_attribute_matrix
from pmedm_vb.assemble.households import build_units, person_counts
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.targets import build_targets
from pmedm_vb.config import StudyArea, ensure_dir

#: Directory under a problem's inputs directory where its held-out tables live.
HELDOUT_DIR = "heldout"

#: Levels, in the order they are stored.
LEVELS = ("tract", "block group")


@dataclass
class HeldOut:
    """One PUMA's held-out tables at both levels.

    Attributes
    ----------
    X:
        Per level, ``(n_units, k)`` sparse counts, rows in the problem's
        ``units`` order.
    Y:
        Per level, ``(n_areas, k)`` published estimates, rows in the problem's
        ``tracts`` or ``block_groups`` order.
    v:
        Per level, ``(n_areas, k)`` published variances, zero-cell policy
        applied as for the constraints.
    names:
        Per level, the ``k`` column names, ``"{table}.{category}"``.
    """

    X: dict[str, sp.csr_matrix]
    Y: dict[str, np.ndarray]
    v: dict[str, np.ndarray]
    names: dict[str, list[str]]

    def relation(self, level: str) -> list[str]:
        """``"related"`` or ``"less_related"`` per column of one level."""
        return [HELDOUT_RELATION[name.split(".")[0]] for name in self.names[level]]

    def save(self, path: Path | str) -> Path:
        """Write under ``<path>/heldout``: one ``.npz``/``.npy`` set per level."""
        root = ensure_dir(Path(path) / HELDOUT_DIR)
        for level in LEVELS:
            tag = level.replace(" ", "_")
            sp.save_npz(root / f"X_{tag}.npz", self.X[level].tocsr())
            np.save(root / f"Y_{tag}.npy", self.Y[level])
            np.save(root / f"v_{tag}.npy", self.v[level])
        (root / "names.json").write_text(json.dumps(self.names, indent=2) + "\n")
        return root

    @classmethod
    def load(cls, path: Path | str) -> "HeldOut":
        """Read back what :meth:`save` wrote, given the problem's directory."""
        root = Path(path) / HELDOUT_DIR
        X, Y, v = {}, {}, {}
        for level in LEVELS:
            tag = level.replace(" ", "_")
            X[level] = sp.load_npz(root / f"X_{tag}.npz").tocsr()
            Y[level] = np.load(root / f"Y_{tag}.npy")
            v[level] = np.load(root / f"v_{tag}.npy")
        names = json.loads((root / "names.json").read_text())
        return cls(X=X, Y=Y, v=v, names=names)

    @staticmethod
    def exists(path: Path | str) -> bool:
        return (Path(path) / HELDOUT_DIR / "names.json").exists()


def build_heldout(
    area: StudyArea,
    inputs: PMEDMInputs,
    tables: Sequence[ConstraintTable],
    *,
    policy: str = "model",
) -> HeldOut:
    """Build the held-out tables for an assembled PUMA-only problem.

    ``tables`` holds every held-out specification, at both levels.

    Raises
    ------
    ValueError
        If the problem has a statewide support, or if the unit set rebuilt from
        PUMS is not the problem's own.
    """
    if inputs.unit_members is not None:
        raise ValueError("held-out tables are only built for the PUMA-only problem")
    units = build_units(area, inputs.puma, tables)
    serials = inputs.units["SERIALNO"].to_numpy()
    if set(units.units.index) != set(serials) or len(serials) != units.n_units:
        raise ValueError(
            f"PUMA {inputs.puma}: the units rebuilt for the held-out tables are not "
            f"the problem's ({units.n_units:,} against {len(serials):,})"
        )

    geoids = {
        "tract": inputs.tracts.iloc[:, 0].astype(str).tolist(),
        "block group": inputs.block_groups.iloc[:, 0].astype(str).tolist(),
    }
    X, Y, v, names = {}, {}, {}, {}
    for level in LEVELS:
        level_tables = [t for t in tables if t.geography == level]
        block = build_targets(area, level_tables, geography=level, geoids=geoids[level],
                              policy=policy)
        if list(block.geoids) != geoids[level]:
            raise ValueError(f"{level}: held-out targets are not in the problem's area order")
        counts = person_counts(units, level_tables).reindex(pd.Index(serials))
        matrix, labels = build_attribute_matrix(counts)
        if labels != list(block.names):
            raise ValueError(f"{level}: held-out X and Y columns do not agree")
        X[level] = matrix
        Y[level] = block.Y
        v[level] = block.v.reshape(block.Y.shape, order="F")
        names[level] = list(block.names)
    return HeldOut(X=X, Y=Y, v=v, names=names)
