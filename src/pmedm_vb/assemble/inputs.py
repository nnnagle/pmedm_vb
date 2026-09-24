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
#: Frames written only when present: the statewide support's bookkeeping.
_OPTIONAL_FRAMES = ("unit_members", "support_columns")


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
    unit_members, support, support_columns:
        ``None`` for a problem over the PUMA's own records. With a statewide
        support (``build_puma(..., epsilon=...)``) a unit is a merged row of
        identical records, and ``unit_members`` has one row per record:
        ``SERIALNO``, ``row`` (its unit), ``weight`` (its design weight) and
        ``in_puma``. ``n`` and ``N`` still count the PUMA's own records.
        ``support`` holds ``epsilon`` and the row counts, and
        ``support_columns`` per-constraint counts of the rows carrying each
        category; see :func:`~pmedm_vb.assemble.build.support_report`.
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
    unit_members: pd.DataFrame | None = None
    support: dict | None = None
    support_columns: pd.DataFrame | None = None

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

    def sigma(
        self,
        alpha: float,
        taper: str | None = "tract",
        variance_floor: str | float | None = None,
    ) -> Sigma:
        """The covariance at this shrinkage, as a :class:`Sigma` view.

        ``alpha`` is a parameter of the *run*, not of the assembled problem, so
        it is supplied here rather than stored: ``v`` and ``L`` are kept raw and
        one assembled problem serves an entire sweep. ``taper`` is too:
        ``"tract"`` keeps replicate covariance only between cells in the same
        tract, ``None`` keeps all of it. Why tract is the default is under
        *Tapering* in :mod:`pmedm_vb.assemble.sigma`. So is ``variance_floor``;
        see :meth:`floored_variances`.
        """
        if taper not in ("tract", None):
            raise ValueError(f"taper must be 'tract' or None, got {taper!r}")
        groups = self.constraint_tracts() if taper == "tract" else None
        v = self.floored_variances(variance_floor)
        return Sigma(v=v, l=self.sigma_l, alpha=alpha, groups=groups)

    def floored_variances(self, variance_floor: str | float | None) -> np.ndarray:
        """``sigma_v`` with no cell more certain than a floor.

        ``None`` leaves the published variances alone. ``"zero"`` floors each
        cell at the zero-count variance ``w * k`` of its own area and level
        (:meth:`zero_cell_variances`); a number floors every cell at that value.

        **Why.** A published zero is given the modelled variance ``w * k`` --
        72 at Knox block groups -- while a published count of 1 or 2 carries
        its replicate variance, often 2 or 3. So "1 +- 1.4" is a far tighter
        constraint than "0 +- 8.5", although both are estimates of a count at or
        near zero. On the Knox fits those tight small-count cells
        were where the posterior over the multipliers was furthest from
        Gaussian: steep on one side, flat on the other, with Laplace draws
        landing tens of thousands of nats below it
        (``experiments/laplace_diagnostic.py``). Flooring at the zero-count
        variance says no estimate is more certain than a zero in the same
        place. It binds only where ``v`` is below the floor, which in practice
        is small counts: at Knox block groups the median variance of counts 1
        to 12 was 68.6 against a floor of 72.

        Raising ``v`` raises the residual diagonal ``D`` and leaves the
        replicate factor alone, so a floored cell's correlations with the rest
        shrink -- the added variance is noise the replicates did not see.
        """
        if variance_floor is None:
            return self.sigma_v
        if isinstance(variance_floor, str):
            if variance_floor != "zero":
                raise ValueError(
                    f"variance_floor must be None, 'zero' or a number, got {variance_floor!r}"
                )
            floor = self.zero_cell_variances()
        else:
            if variance_floor <= 0:
                raise ValueError(f"a numeric variance_floor must be positive, got {variance_floor}")
            floor = np.full(self.n_constraints, float(variance_floor))
        return np.maximum(self.sigma_v, floor)

    def zero_cell_variances(self) -> np.ndarray:
        """``(n_constraints,)`` the zero-count variance of each row's area and level.

        Read off the data rather than recomputed: a cell published as zero,
        with no replicate spread, carries exactly the modelled ``w * k`` of its
        area (the ``"model"`` zero-cell policy), and ``k`` depends on the area's
        population, so it differs between areas. An area with no zero cell of
        its own takes the median over the areas at its level that have one.

        Raises
        ------
        ValueError
            If a level has no zero cells at all, so there is nothing to read.
        """
        m = self.n_constraints
        replicate = (
            np.zeros(m) if self.sigma_l is None
            else SDR_FACTOR * np.square(self.sigma_l).sum(axis=1)
        )
        zero = (self.targets() == 0) & (replicate == 0)
        split = self.Y_T.size
        levels = [
            (slice(0, split), np.tile(np.arange(self.Y_T.shape[0]), self.Y_T.shape[1]), "tract"),
            (slice(split, m), np.tile(np.arange(self.Y_B.shape[0]), self.Y_B.shape[1]), "block group"),
        ]
        out = np.empty(m)
        for rows, areas, name in levels:
            v, is_zero = self.sigma_v[rows], zero[rows]
            if not is_zero.any():
                raise ValueError(
                    f"no {name} cell is a published zero, so there is no zero-count "
                    f"variance to floor at; pass a number instead"
                )
            per_area = pd.Series(v[is_zero]).groupby(areas[is_zero]).median()
            out[rows] = per_area.reindex(areas).fillna(per_area.median()).to_numpy()
        return out

    def targets(self) -> np.ndarray:
        """The stacked constraint vector ``[vec(Y_T); vec(Y_B)]``, column-major."""
        return np.concatenate([self.Y_T.ravel(order="F"), self.Y_B.ravel(order="F")])

    def zone_tracts(self) -> np.ndarray:
        """``(n_zones,)`` index into :attr:`tracts` of the tract holding each zone.

        Read off ``A_T``, which must have exactly one nonzero per column: a
        zone in no tract, or in two, means the aggregation is not a nesting.
        """
        a_t = sp.csc_matrix(self.A_T)
        counts = np.diff(a_t.indptr)
        if not np.all(counts == 1):
            raise ValueError(
                f"A_T must place each zone in exactly one tract; "
                f"{int((counts != 1).sum())} zone(s) do not"
            )
        return a_t.indices.copy()

    def constraint_tracts(self) -> np.ndarray:
        """``(n_constraints,)`` tract index of each row of the stacked vector.

        Tract row ``k * n_tracts + t`` belongs to tract ``t``; block-group row
        ``k * n_block_groups + b`` to the tract holding block group ``b``. Both
        follow from the column-major stacking. Assumes ``A_B`` is the identity,
        i.e. that the zones *are* the block groups, which is how
        :func:`~pmedm_vb.assemble.build.build_puma` builds it.
        """
        if self.A_B.shape[0] != self.n_zones or (
            sp.csr_matrix(self.A_B) != sp.identity(self.n_zones)
        ).nnz:
            raise ValueError("constraint_tracts assumes A_B is the identity")
        n_tracts = self.A_T.shape[0]
        tract_rows = np.tile(np.arange(n_tracts), self.Y_T.shape[1])
        bg_rows = np.tile(self.zone_tracts(), self.Y_B.shape[1])
        return np.concatenate([tract_rows, bg_rows])

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
        if self.unit_members is not None:
            members = self.unit_members
            kept = members[members["row"] >= 0]
            summed = kept.groupby("row")["weight"].sum().reindex(range(self.n_units))
            checks += [
                (int(members["in_puma"].sum()) == self.n,
                 f"unit_members has {int(members['in_puma'].sum())} PUMA records, "
                 f"expected n = {self.n}"),
                (bool(np.allclose(summed.to_numpy(), self.units["weight"].to_numpy())),
                 "unit_members' weights do not sum to each unit's weight"),
                (bool(np.isclose(self.units["weight"].sum(), self.N)),
                 f"unit weights sum to {self.units['weight'].sum():,.1f}, expected N = {self.N:,.1f}"),
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
        for name in _OPTIONAL_FRAMES:
            value = getattr(self, name)
            if value is not None:
                value.to_parquet(path / f"{name}.parquet")

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
        if self.support is not None:
            manifest["support"] = self.support
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
        fields.update(
            {
                name: pd.read_parquet(path / f"{name}.parquet")
                for name in _OPTIONAL_FRAMES
                if (path / f"{name}.parquet").exists()
            }
        )

        return cls(
            puma=manifest["puma"],
            n=manifest["n"],
            N=manifest["N"],
            tract_constraints=manifest["tract_constraints"],
            bg_constraints=manifest["bg_constraints"],
            support=manifest.get("support"),
            **fields,
        )
