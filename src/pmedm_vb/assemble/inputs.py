"""The container handed to a solver, and the on-disk format for it.

Notation follows ``pmedm_derivation.md``. The pycnophylactic constraints are

.. math::

    Y_T = A_T W X_T + e_T, \\qquad Y_B = A_B W X_B + e_B

where ``W`` is the (zones x units) matrix of population weights the
solver is after. Vectorising with the identity ``vec(A X B) = (B' kron A) vec(X)``
stacks these into ``Y = X_tilde' vec(W)`` with

.. math::

    X_tilde' = \\begin{bmatrix} X_T' \\otimes A_T \\\\ X_B' \\otimes A_B \\end{bmatrix}

**This class stores the four factors and never materialises that Kronecker
product.** It has (tract cells + block group cells) rows and
(zones x units) columns, while the factors are small and sparse, and
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
from pmedm_vb.assemble.sigma import Sigma
from pmedm_vb.config import ensure_dir
from pmedm_vb.data.variance import N_REPLICATES, SDR_FACTOR

_SPARSE = ("X_T", "X_B", "A_T", "A_B")
_DENSE = ("q", "Y_T", "Y_B", "sigma_v")
#: Written only when present; ``None`` is the exactly-diagonal case.
_OPTIONAL_DENSE = ("sigma_l",)
_FRAMES = ("units", "zones", "tracts", "block_groups")


@dataclass
class PMEDMInputs:
    """Everything a solver needs for one PUMA.

    The PUMA is the unit of a problem, not the study area. Design weights are
    zero outside a record's own PUMA, so the joint problem is block diagonal
    by PUMA and each block is an independent PMEDM.

    Attributes
    ----------
    q:
        Design probabilities, shape ``(n_zones, n_units)`` -- the same
        orientation as the solution ``W``. Zero marks a unit that cannot
        receive weight in a zone. Within a single PUMA's problem that is rare
        rather than structural: PUMA membership is what selected these units in
        the first place.
    X_T, X_B:
        Unit attribute matrices, ``(n_units, n_constraints)``, sparse. Entries
        are **counts, not indicators**: a household contributes the number of
        its members matching a person-level cell, and 0/1 is only the special
        case of a household-level one.
    A_T, A_B:
        Aggregation from zones to constraint rows, ``(n_areas, n_zones)``,
        sparse. ``A_B`` is the identity when zones *are* block groups.
    Y_T, Y_B:
        Published estimates, ``(n_areas, n_constraints)``.
    sigma_v, sigma_l:
        Covariance factors over the stacked constraint vector
        ``[vec(Y_T); vec(Y_B)]``, column-major. ``sigma_v`` is the published
        per-cell variance; ``sigma_l`` the ``(n, 80)`` replicate deviations, or
        ``None`` for an exactly diagonal covariance. Held raw so that the
        shrinkage ``alpha`` stays a parameter of the run -- see
        :meth:`sigma`.
    puma:
        The PUMA this problem covers. One PMEDM per PUMA.
    n:
        Sample size -- the number of units, before weighting.
    N:
        Total the weights must sum to, in units: occupied housing units plus
        group-quarters residents in this PUMA.
    units, zones, tracts, block_groups:
        Identifier frames whose row order defines the axes above. A *unit* is a
        household or a group-quarters person: GQ records carry no ``WGTP``, so
        they enter as one-person units weighted by ``PWGTP``, which is how the
        total-population tables stay matchable.
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
    sigma_v: np.ndarray
    sigma_l: np.ndarray | None
    puma: str
    n: int
    N: float
    units: pd.DataFrame
    zones: pd.DataFrame
    tracts: pd.DataFrame
    block_groups: pd.DataFrame
    tract_constraints: list[str]
    bg_constraints: list[str]

    # -- shape accessors -------------------------------------------------

    @property
    def n_units(self) -> int:
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
        """Whether no replicate structure is carried at all.

        Distinct from ``alpha = 1``, which also yields a diagonal ``Sigma`` but
        keeps ``L`` so the rest of the sweep stays available.
        """
        return self.sigma_l is None

    def sigma(self, alpha: float) -> Sigma:
        """The covariance at this shrinkage, as a :class:`Sigma` view.

        ``alpha`` is a parameter of the *run*, not of the assembled problem, so
        it is supplied here rather than stored: ``v`` and ``L`` are kept raw and
        one assembled problem serves an entire sweep.
        """
        return Sigma(v=self.sigma_v, l=self.sigma_l, alpha=alpha)

    # -- consistency -----------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ValueError` if the pieces do not line up.

        Cheap to run and worth running: a transposed factor or an off-by-one in
        the constraint list produces a solver that converges to something
        meaningless rather than an error.
        """
        checks = [
            (self.X_T.shape[0] == self.n_units,
             f"X_T has {self.X_T.shape[0]} rows, expected {self.n_units} units"),
            (self.X_B.shape[0] == self.n_units,
             f"X_B has {self.X_B.shape[0]} rows, expected {self.n_units} units"),
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
            (len(self.units) == self.n_units,
             "units frame does not match the width of q"),
            (len(self.zones) == self.n_zones,
             "zones frame does not match the height of q"),
        ]
        checks.append(
            (self.sigma_v.shape == (self.n_constraints,),
             f"sigma_v is {self.sigma_v.shape}, expected {(self.n_constraints,)}")
        )
        if self.sigma_l is not None:
            expected = (self.n_constraints, N_REPLICATES)
            checks.append(
                (self.sigma_l.shape == expected,
                 f"sigma_l is {self.sigma_l.shape}, expected {expected}")
            )
        # A permutation of sigma_v against sigma_l is undetectable downstream
        # and yields a converged solver with wrong uncertainties, so the one
        # cheap consequence of it is checked here: the residual diagonal must
        # stay positive, which it cannot if the two describe different cells.
        if self.sigma_l is not None and self.sigma_v.shape == (self.n_constraints,):
            residual = self.sigma_v - SDR_FACTOR * np.square(self.sigma_l).sum(axis=1)
            checks.append(
                (bool(np.all(residual >= 0)),
                 f"{int((residual < 0).sum())} cell(s) have sigma_v below their "
                 f"own replicate variance, so sigma_v and sigma_l disagree")
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
        for name in _OPTIONAL_DENSE:
            value = getattr(self, name)
            if value is not None:
                np.save(path / f"{name}.npy", np.asarray(value))
        for name in _SPARSE:
            sp.save_npz(path / f"{name}.npz", getattr(self, name).tocsr())
        for name in _FRAMES:
            getattr(self, name).to_parquet(path / f"{name}.parquet")

        manifest = {
            "pmedm_vb_version": __version__,
            "puma": str(self.puma),
            "n": int(self.n),
            "N": float(self.N),
            "tract_constraints": list(self.tract_constraints),
            "bg_constraints": list(self.bg_constraints),
            "sigma_is_diagonal": self.sigma_is_diagonal,
            "shapes": {
                "n_units": self.n_units,
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
        fields.update(
            {
                name: (
                    np.load(path / f"{name}.npy")
                    if (path / f"{name}.npy").exists()
                    else None
                )
                for name in _OPTIONAL_DENSE
            }
        )
        fields.update({name: sp.load_npz(path / f"{name}.npz") for name in _SPARSE})
        fields.update({name: pd.read_parquet(path / f"{name}.parquet") for name in _FRAMES})

        return cls(
            puma=manifest["puma"],
            n=manifest["n"],
            N=manifest["N"],
            tract_constraints=manifest["tract_constraints"],
            bg_constraints=manifest["bg_constraints"],
            **fields,
        )
