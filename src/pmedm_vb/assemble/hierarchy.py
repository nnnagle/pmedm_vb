"""Hierarchical constraints: each total measured once.

Block group estimates sum exactly to their tract's, and so do their replicates,
so a tract's total for a category is measured twice -- by its tract row and by
the sum of its block group rows -- and at ``alpha = 1`` the two are treated as
independent. ``level`` removes the duplicate by constraining multipliers to sum
to zero within each hierarchy level:

- ``"none"``: today's model.
- ``"tract"``: for each category constrained at both levels, the block group
  multipliers of each tract sum to zero, so a block group's tilt is
  ``lambda_t + lambda_b`` with the tract row carrying the tract total and the
  block group rows only its split. Restricting ``lambda`` to that subspace is
  exactly dropping the constraint combination that duplicates the tract row.
- ``"puma"``: as ``"tract"``, plus one PUMA row per tract category -- the tract
  estimates summed, with the SDR variance of the summed tract replicates -- and
  the tract multipliers of each category summing to zero across the PUMA. A
  block group's tilt is then ``lambda_P + lambda_t + lambda_b``, and the PUMA
  total is measured once, at the survey's own variance, which the tract taper
  otherwise discards.

**Coordinates.** A solver works in ``xi = (theta, lambda_P)``: ``theta`` in the
stacked tract/block group layout, ``lambda_P`` the PUMA rows (none below
``"puma"``). With ``M`` the projection removing each sum-to-zero group's mean,

- the multipliers the data see, in today's layout, are
  ``lambda_data = M theta + E lambda_P``, ``E`` adding ``lambda_P[k]`` to every
  tract row of category ``k`` (:meth:`Hierarchy.lambda_data`);
- the multipliers of the constraints are ``zeta = (M theta, lambda_P)``
  (:meth:`Hierarchy.zeta`), which pair with ``Y_ext = (Y, Y_P)`` and
  ``Sigma_ext`` (:meth:`Hierarchy.sigma`);
- the objective is

  .. math::

      \\tilde f(\\xi) = Y_{ext}'\\zeta / N + \\log q' e^{-X\\lambda_{data}}
                      + \\tfrac{c}{2}\\zeta'\\Sigma_{ext}\\zeta
                      + \\tfrac{\\kappa}{2}\\|(I - M)\\theta\\|^2 .

The last term only makes the dropped directions proper: they appear in no other
term, so under ``exp(-n f~)`` they are exactly ``N(0, 1/(n kappa))`` and
independent of the rest, and ``lambda_data`` -- hence ``p`` and ``W`` -- does
not depend on them. ``kappa = 1/n`` gives them unit scale.

``Sigma_ext`` is the run's ``Sigma`` with the PUMA rows appended as their own
taper group: their replicate covariance with each other is kept, their
covariance with tract rows dropped (only the tract rows' contrasts survive the
projection, and their covariance with the total is set to zero). A PUMA row
whose summed replicates have no spread -- a published zero in every tract --
takes the sum of its tract rows' variances instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.sigma import Sigma
from pmedm_vb.data.variance import SDR_FACTOR

LEVELS = ("none", "tract", "puma")


def _group_means(x: np.ndarray, group: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-row mean of ``x`` over its group (0 for ungrouped rows); ``x`` is
    ``(rows,)`` or ``(rows, k)``."""
    rows = group >= 0
    k = x.shape[1:] if x.ndim > 1 else ()
    sums = np.zeros((counts.size,) + k)
    np.add.at(sums, group[rows], x[rows])
    out = np.zeros_like(x, dtype=float)
    out[rows] = sums[group[rows]] / counts[group[rows]].reshape((-1,) + (1,) * len(k))
    return out


@dataclass
class Hierarchy:
    """The sum-to-zero groups, the PUMA rows and the maps between coordinates.

    Build with :meth:`build`. ``group`` labels every stacked row with its
    sum-to-zero group (``-1`` for none); block group groups and tract groups
    are disjoint. ``bg_group`` is the block group part alone, which stays
    inside one tract. ``puma_of_row`` gives, for each tract row, the PUMA row
    its category adds to (``-1`` elsewhere, and everywhere below ``"puma"``).
    """

    level: str
    m: int
    group: np.ndarray
    bg_group: np.ndarray
    puma_of_row: np.ndarray
    y_ext: np.ndarray
    Y_P: np.ndarray
    v_P: np.ndarray
    l_P: np.ndarray | None
    kappa: float
    n_tracts: int
    puma_names: list[str]

    @classmethod
    def build(cls, inputs: PMEDMInputs, level: str = "none") -> "Hierarchy":
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        n_tracts, c_t = inputs.Y_T.shape
        n_zones, c_b = inputs.Y_B.shape
        split, m = inputs.Y_T.size, inputs.n_constraints
        group = np.full(m, -1)
        next_id = 0
        if level in ("tract", "puma"):
            inputs.constraint_tracts()  # raises unless A_B is the identity
            zone_tract = inputs.zone_tracts()
            tract_index = {name: k for k, name in enumerate(inputs.tract_constraints)}
            for kb, name in enumerate(inputs.bg_constraints):
                if name not in tract_index:
                    continue  # constrained at block group only: nothing to duplicate
                rows = split + kb * n_zones + np.arange(n_zones)
                group[rows] = next_id + zone_tract
                next_id += n_tracts
        bg_group = group.copy()
        puma_of_row = np.full(m, -1)
        Y_P = np.zeros(0)
        v_P = np.zeros(0)
        l_P = None
        if level == "puma":
            if inputs.sigma_l is None:
                raise ValueError("PUMA rows need the replicate deviations, and sigma_l is None")
            for kt in range(c_t):
                rows = kt * n_tracts + np.arange(n_tracts)
                group[rows] = next_id
                next_id += 1
                puma_of_row[rows] = kt
            Y_P = inputs.Y_T.sum(axis=0).astype(float)
            l_P = np.stack([inputs.sigma_l[kt * n_tracts + np.arange(n_tracts)].sum(axis=0)
                            for kt in range(c_t)])
            v_P = SDR_FACTOR * np.square(l_P).sum(axis=1)
            tract_v = inputs.sigma_v[:split].reshape(inputs.Y_T.shape, order="F").sum(axis=0)
            v_P = np.where(v_P > 0, v_P, tract_v)
        # Re-number groups densely so counts index them.
        used = np.unique(group[group >= 0])
        remap = np.full(max(next_id, 1), -1)
        remap[used] = np.arange(used.size)
        group = np.where(group >= 0, remap[np.maximum(group, 0)], -1)
        bg_group = np.where(bg_group >= 0, remap[np.maximum(bg_group, 0)], -1)
        y_ext = np.concatenate([inputs.targets(), Y_P]) / inputs.N
        return cls(level=level, m=m, group=group, bg_group=bg_group, puma_of_row=puma_of_row,
                   y_ext=y_ext, Y_P=Y_P, v_P=v_P, l_P=l_P, kappa=1.0 / inputs.n,
                   n_tracts=n_tracts, puma_names=list(inputs.tract_constraints) if level == "puma" else [])

    # -- sizes -----------------------------------------------------------

    @property
    def n_puma(self) -> int:
        return self.Y_P.size

    @property
    def size(self) -> int:
        """Length of ``xi``: the stacked rows plus the PUMA rows."""
        return self.m + self.n_puma

    @property
    def is_trivial(self) -> bool:
        return self.level == "none"

    def _counts(self, group: np.ndarray) -> np.ndarray:
        return np.bincount(group[group >= 0], minlength=int(group.max()) + 1 if group.max() >= 0 else 0)

    # -- numpy maps (vectors are (rows,) or (rows, k)) -----------------------

    def project(self, theta: np.ndarray, which: str = "all") -> np.ndarray:
        """``M theta``: subtract each group's mean. ``which="bg"`` uses only the
        block group groups (the part of ``M`` local to a tract)."""
        group = self.group if which == "all" else self.bg_group
        if group.max() < 0:
            return np.array(theta, dtype=float)
        return theta - _group_means(theta, group, self._counts(group))

    def null(self, theta: np.ndarray) -> np.ndarray:
        """``(I - M) theta``, the dropped directions."""
        return theta - self.project(theta)

    def puma_sum(self, g: np.ndarray) -> np.ndarray:
        """``E' g``: for each PUMA row, the sum of ``g`` over its tract rows."""
        rows = self.puma_of_row >= 0
        k = g.shape[1:] if g.ndim > 1 else ()
        out = np.zeros((self.n_puma,) + k)
        np.add.at(out, self.puma_of_row[rows], g[: self.m][rows])
        return out

    def lambda_data(self, xi: np.ndarray) -> np.ndarray:
        """``M theta + E lambda_P``, the multipliers in today's layout."""
        lam = self.project(xi[: self.m])
        if self.n_puma:
            rows = self.puma_of_row >= 0
            lam[rows] += xi[self.m:][self.puma_of_row[rows]]
        return lam

    def lambda_data_T(self, g: np.ndarray) -> np.ndarray:
        """Adjoint of :meth:`lambda_data`: ``(M g, E' g)``."""
        return np.concatenate([self.project(g), self.puma_sum(g)], axis=0)

    def zeta(self, xi: np.ndarray) -> np.ndarray:
        """``(M theta, lambda_P)``, the multipliers of the extended constraints."""
        return np.concatenate([self.project(xi[: self.m]), xi[self.m:]], axis=0)

    zeta_T = zeta  # M is symmetric, so the map is its own adjoint

    def ridge_gradient(self, xi: np.ndarray) -> np.ndarray:
        """Gradient of ``(kappa/2) ||(I - M) theta||^2`` in ``xi``."""
        out = np.zeros_like(xi, dtype=float)
        out[: self.m] = self.kappa * self.null(xi[: self.m])
        return out

    def extend(self, stacked: np.ndarray) -> np.ndarray:
        """A stacked-row vector with its PUMA totals appended, e.g. ``N X'p``."""
        return np.concatenate([stacked, self.puma_sum(stacked)], axis=0)

    def tract_group_basis(self) -> np.ndarray:
        """``(size, n_puma)``: column ``k`` the unit-norm indicator of category
        ``k``'s tract rows -- the part of ``I - M`` that couples tracts."""
        U = np.zeros((self.size, self.n_puma))
        rows = np.flatnonzero(self.puma_of_row >= 0)
        U[rows, self.puma_of_row[rows]] = 1.0 / np.sqrt(self.n_tracts)
        return U

    def local_projector(self, rows: np.ndarray) -> np.ndarray:
        """``I - M_B`` restricted to ``rows`` (one tract's block): the block group
        groups there average to zero within the block."""
        g = self.bg_group[rows]
        Q = np.zeros((rows.size, rows.size))
        for label in np.unique(g[g >= 0]):
            members = np.flatnonzero(g == label)
            Q[np.ix_(members, members)] = 1.0 / members.size
        return Q

    # -- Sigma -----------------------------------------------------------

    def sigma(self, inputs: PMEDMInputs, alpha: float, taper: str | None,
              variance_floor: str | float | None) -> Sigma:
        """``Sigma_ext``: the run's ``Sigma`` with the PUMA rows appended as their
        own taper group. The same as ``inputs.sigma(...)`` below ``"puma"``."""
        base = inputs.sigma(alpha, taper, variance_floor)
        if not self.n_puma:
            return base
        v = np.concatenate([base.v, self.v_P])
        l = np.vstack([base.l, self.l_P])
        groups = None
        if base.groups is not None:
            groups = np.concatenate([base.groups, np.full(self.n_puma, base.groups.max() + 1)])
        return Sigma(v=v, l=l, alpha=alpha, groups=groups)

    # -- torch maps (tensors are (batch, rows)) ------------------------------

    def torch_maps(self, device: str):
        """The same maps on ``(batch, size)`` tensors, for the VB and HMC target."""
        return _TorchHierarchy(self, device)


class _TorchHierarchy:
    def __init__(self, h: Hierarchy, device: str) -> None:
        import torch

        self.torch = torch
        self.m, self.n_puma, self.kappa = h.m, h.n_puma, h.kappa
        grouped = np.flatnonzero(h.group >= 0)
        self.grouped = torch.as_tensor(grouped, device=device)
        self.group = torch.as_tensor(h.group[grouped], device=device)
        counts = h._counts(h.group) if grouped.size else np.zeros(0)
        self.counts = torch.as_tensor(counts, dtype=torch.float64, device=device)
        tract = np.flatnonzero(h.puma_of_row >= 0)
        self.tract_rows = torch.as_tensor(tract, device=device)
        self.tract_puma = torch.as_tensor(h.puma_of_row[tract], device=device)

    def project(self, theta):
        if not self.grouped.numel():
            return theta
        sums = theta.new_zeros(theta.shape[0], self.counts.numel())
        sums = sums.index_add(1, self.group, theta[:, self.grouped])
        means = sums / self.counts
        out = theta.clone()
        out[:, self.grouped] = theta[:, self.grouped] - means[:, self.group]
        return out

    def lambda_data(self, xi):
        lam = self.project(xi[:, : self.m])
        if self.n_puma:
            lam = lam.index_add(1, self.tract_rows, xi[:, self.m:][:, self.tract_puma])
        return lam

    def zeta(self, xi):
        return self.torch.cat([self.project(xi[:, : self.m]), xi[:, self.m:]], dim=1)

    def ridge(self, xi):
        theta = xi[:, : self.m]
        null = theta - self.project(theta)
        return 0.5 * self.kappa * (null * null).sum(1)
