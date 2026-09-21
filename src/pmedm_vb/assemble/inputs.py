"""The container handed to a solver, and the on-disk format for it.

Notation follows ``pmedm_derivation.md``. The pycnophylactic constraints are

.. math::

    Y_T = A_T W X_T + e_T, \\qquad Y_B = A_B W X_B + e_B

where ``W`` is the (zones x individuals) matrix of population weights the
solver is after. Vectorising with the identity ``vec(A X B) = (B' kron A) vec(X)``
stacks these into ``Y = X_tilde' vec(W)`` with

.. math::

    X_tilde' = \\begin{bmatrix} X_T' \\otimes A_T \\\\ X_B' \\otimes A_B \\end{bmatrix}

**This class stores the four factors and never materialises that Kronecker
product.** It has (tract cells + block group cells) rows and
(zones x individuals) columns, while the factors are small and sparse, and
every use the solver makes of it -- ``X'p`` in the gradient and
``X' diag(p) X`` in the Hessian -- needs only products against a vector. Those
factor into two ordinary matrix multiplications:
:class:`~pmedm_vb.solvers.base.ConstraintOperator` owns that arithmetic.

Anywhere a ``vec`` appears it is **column-major** (``order="F"``), which is what
the identity above requires.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb import __version__
from pmedm_vb.config import ensure_dir

_SPARSE = ("X_T", "X_B", "A_T", "A_B")
_DENSE = ("q", "Y_T", "Y_B", "sigma")
_FRAMES = ("individuals", "zones", "tracts", "block_groups")


@dataclass
class PMEDMInputs:
    """Everything a solver needs for one study area.

    Attributes
    ----------
    q:
        Design probabilities, shape ``(n_zones, n_individuals)`` -- the same
        orientation as the solution ``W``. Zero marks an individual who cannot
        receive weight in a zone, which is how PUMA membership enters.
    X_T, X_B:
        Individual attribute matrices, ``(n_individuals, n_constraints)``,
        sparse. Typically indicator columns, one per constraint table cell.
    A_T, A_B:
        Aggregation from zones to constraint rows, ``(n_areas, n_zones)``,
        sparse. ``A_B`` is the identity when zones *are* block groups.
    Y_T, Y_B:
        Published estimates, ``(n_areas, n_constraints)``.
    sigma:
        Error covariance over the stacked constraint vector
        ``[vec(Y_T); vec(Y_B)]``. One-dimensional means a diagonal built from
        published margins of error; two-dimensional means a full covariance
        from variance replicates.
    n:
        Sample size -- the number of PUMS records, before weighting.
    N:
        Population total the weights must sum to.
    individuals, zones, tracts, block_groups:
        Identifier frames whose row order defines the axes above.
    tract_constraints, bg_constraints:
        Column labels for ``X_T``/``Y_T`` and ``X_B``/``Y_B``.
    """

    q: np.ndarray
    X_T: sp.spmatrix
    X_B: sp.spmatrix
    A_T: sp.spmatrix
    A_B: sp.spmatrix
    Y_T: np.ndarray
    Y_B: np.ndarray
    sigma: np.ndarray
    n: int
    N: float
    individuals: pd.DataFrame
    zones: pd.DataFrame
    tracts: pd.DataFrame
    block_groups: pd.DataFrame
    tract_constraints: list[str]
    bg_constraints: list[str]

    # -- shape accessors -------------------------------------------------

    @property
    def n_individuals(self) -> int:
        return self.q.shape[1]

    @property
    def n_zones(self) -> int:
        return self.q.shape[0]

    @property
    def n_constraints(self) -> int:
        """Rows of the stacked constraint vector."""
        return self.Y_T.size + self.Y_B.size

    @property
    def sigma_is_diagonal(self) -> bool:
        return self.sigma.ndim == 1

    # -- consistency -----------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ValueError` if the pieces do not line up.

        Cheap to run and worth running: a transposed factor or an off-by-one in
        the constraint list produces a solver that converges to something
        meaningless rather than an error.
        """
        checks = [
            (self.X_T.shape[0] == self.n_individuals,
             f"X_T has {self.X_T.shape[0]} rows, expected {self.n_individuals} individuals"),
            (self.X_B.shape[0] == self.n_individuals,
             f"X_B has {self.X_B.shape[0]} rows, expected {self.n_individuals} individuals"),
            (self.A_T.shape[1] == self.n_zones,
             f"A_T has {self.A_T.shape[1]} columns, expected {self.n_zones} zones"),
            (self.A_B.shape[1] == self.n_zones,
             f"A_B has {self.A_B.shape[1]} columns, expected {self.n_zones} zones"),
            (self.Y_T.shape == (self.A_T.shape[0], self.X_T.shape[1]),
             f"Y_T is {self.Y_T.shape}, expected {(self.A_T.shape[0], self.X_T.shape[1])}"),
            (self.Y_B.shape == (self.A_B.shape[0], self.X_B.shape[1]),
             f"Y_B is {self.Y_B.shape}, expected {(self.A_B.shape[0], self.X_B.shape[1])}"),
            (len(self.tract_constraints) == self.X_T.shape[1],
             "tract_constraints does not match the width of X_T"),
            (len(self.bg_constraints) == self.X_B.shape[1],
             "bg_constraints does not match the width of X_B"),
            (len(self.individuals) == self.n_individuals,
             "individuals frame does not match the width of q"),
            (len(self.zones) == self.n_zones,
             "zones frame does not match the height of q"),
        ]
        expected_sigma = (
            (self.n_constraints,)
            if self.sigma.ndim == 1
            else (self.n_constraints, self.n_constraints)
        )
        checks.append(
            (self.sigma.shape == expected_sigma,
             f"sigma is {self.sigma.shape}, expected {expected_sigma}")
        )

        problems = [message for ok, message in checks if not ok]
        if problems:
            raise ValueError("inconsistent PMEDM inputs:\n  " + "\n  ".join(problems))

    # -- persistence -----------------------------------------------------

    def save(self, path: Path | str) -> Path:
        """Write to a directory: Parquet for frames, ``.npy``/``.npz`` for arrays."""
        path = ensure_dir(Path(path))

        for name in _DENSE:
            np.save(path / f"{name}.npy", np.asarray(getattr(self, name)))
        for name in _SPARSE:
            sp.save_npz(path / f"{name}.npz", getattr(self, name).tocsr())
        for name in _FRAMES:
            getattr(self, name).to_parquet(path / f"{name}.parquet")

        manifest = {
            "pmedm_vb_version": __version__,
            "n": int(self.n),
            "N": float(self.N),
            "tract_constraints": list(self.tract_constraints),
            "bg_constraints": list(self.bg_constraints),
            "sigma_is_diagonal": self.sigma_is_diagonal,
            "shapes": {
                "n_individuals": self.n_individuals,
                "n_zones": self.n_zones,
                "n_tracts": int(self.A_T.shape[0]),
                "n_block_groups": int(self.A_B.shape[0]),
                "n_constraints": int(self.n_constraints),
            },
        }
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "PMEDMInputs":
        """Read back a directory written by :meth:`save`."""
        path = Path(path)
        manifest = json.loads((path / "manifest.json").read_text())

        fields = {name: np.load(path / f"{name}.npy") for name in _DENSE}
        fields.update({name: sp.load_npz(path / f"{name}.npz") for name in _SPARSE})
        fields.update({name: pd.read_parquet(path / f"{name}.parquet") for name in _FRAMES})

        return cls(
            n=manifest["n"],
            N=manifest["N"],
            tract_constraints=manifest["tract_constraints"],
            bg_constraints=manifest["bg_constraints"],
            **fields,
        )
