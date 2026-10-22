"""A Tweedie error for published zeros: the ``+tweedie<w>`` level suffix.

In the dual (``pmedm_derivation.md``) a row's Gaussian error enters as
``y zeta + (c / 2) D zeta^2`` with ``c = n / N^2``: ``(1/n) K(n zeta / N)`` for
``K(t) = Y t + D t^2 / 2``, the cumulant generating function of ``N(Y, D)``.
Taking the dual as an exponential-family model in its own right, any
cumulant generating function can stand in its place; this one replaces it, on
the rows published as zero, by a Tweedie's with power ``p = 1.5`` -- a
compound Poisson sum of exponential amounts, as a survey count is a random
number of sampled people times their weights.

With the amounts' mean the average weight ``w`` (Census's zero-count variance
is ``w k``) and the variance the row's ``v0 = w k``, the mean is
``mu0 = v0 / (2 w) = k / 2`` and

.. math::

    K(t) = \\frac{\\mu_0 t}{1 - w t}, \\qquad t < 1 / w,

with ``K'(0) = mu0``, ``K''(0) = 2 mu0 w = v0`` and ``P(0) = exp(-mu0 / w)``.
Its conjugate, the penalty on an allocated count ``m``, is
``(sqrt(m) - sqrt(mu0))^2 / w``: Gaussian on the square-root scale.

Only rows published as zero with no replicate spread are changed, where the
whole of ``Sigma``'s diagonal is ``v0`` (for a nonzero row at ``alpha = 0.01``
the diagonal part is ``alpha v``, too little to matter). A prior's ``tau^2``
stays a Gaussian prior on the multipliers. Beyond ``t = 1 / w`` the target is
infinite: HMC rejects any trajectory that reaches it, and the MAP's line search
backs off. A Gaussian VB fit cannot have an infinite target anywhere, so with
``extend`` the term is continued past ``EXTEND / w`` by its second-order Taylor
expansion there -- convex and finite, for VB only (it whitens HMC).

It travels with the hierarchy level: ``"puma-rpruned+prior3+tweedie18"``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

SEPARATOR = "+tweedie"
#: VB's continuation starts at this share of the bound ``1 / w``. At 0.99 the
#: term's curvature there, ``(1 - EXTEND)^-3``, made the ELBO estimate noisy on
#: the synthetic problem (Laplace ELBO standard error 15,869 against 101 without
#: the term); at 0.9 it was 106.
EXTEND = 0.9


def split_tweedie(level: str) -> tuple[str, float | None]:
    """``"puma+prior3+tweedie18"`` -> ``("puma+prior3", 18.0)``; none -> ``(level, None)``."""
    if SEPARATOR not in level:
        return level, None
    head, _, rest = level.partition(SEPARATOR)
    spec, plus, tail = rest.partition("+")
    try:
        w = float(spec)
    except ValueError:
        raise ValueError(f"a Tweedie zero term is +tweedie<w>, e.g. +tweedie18; got {level!r}") from None
    if not w > 0:
        raise ValueError(f"the Tweedie weight must be positive, got {level!r}")
    return head + (plus + tail if plus else ""), w


def tweedie_level(level: str, w: float | None) -> str:
    """The level string with the Tweedie zero term appended, for ``--tweedie-weight``."""
    if w is None:
        return level
    if not w > 0:
        raise ValueError(f"--tweedie-weight must be positive, got {w}")
    return f"{level}{SEPARATOR}{w:g}"


@dataclass
class TweedieZeros:
    """The Tweedie term on the multiplier rows (``zeta``'s) published as zero.

    ``rows`` index ``zeta``; ``mu0`` and ``v0`` are per row; ``w`` the amounts'
    mean; ``n``, ``N`` the problem's. Every method returns the *change* from the
    Gaussian term it replaces, so the rest of the dual is untouched.
    """

    rows: np.ndarray
    mu0: np.ndarray
    v0: np.ndarray
    w: float
    n: int
    N: float

    @classmethod
    def build(cls, inputs, hierarchy, w: float, variance_floor=None) -> "TweedieZeros":
        """The rows of ``hierarchy``'s ``zeta`` published as zero with no replicate
        spread, from its ``Sigma`` (whose ``v`` and ``l`` do not depend on alpha)."""
        sigma = hierarchy.sigma(inputs, 1.0, None, variance_floor)
        spread = (np.zeros(sigma.v.size) if sigma.l is None
                  else np.square(sigma.l).sum(axis=1))
        rows = np.flatnonzero((hierarchy.targets_ext == 0) & (spread == 0))
        v0 = np.asarray(sigma.v[rows], dtype=float)
        return cls(rows=rows, mu0=v0 / (2 * w), v0=v0, w=float(w), n=int(inputs.n),
                   N=float(inputs.N))

    @property
    def c(self) -> float:
        return self.n / self.N**2

    def bound(self) -> float:
        """The largest ``zeta`` on these rows: ``t = n zeta / N < 1 / w``."""
        return self.N / (self.n * self.w)

    # -- numpy, for the MAP ----------------------------------------------

    def _t(self, zeta: np.ndarray) -> np.ndarray:
        return self.n * zeta[self.rows] / self.N

    def feasible(self, zeta: np.ndarray) -> bool:
        return bool(np.all(self.w * self._t(zeta) < 1))

    def value(self, zeta: np.ndarray) -> float:
        """``sum (1/n) K(t) - (c/2) v0 zeta^2`` over the rows; ``inf`` past the bound."""
        if not self.rows.size:
            return 0.0
        t = self._t(zeta)
        gap = 1 - self.w * t
        if np.any(gap <= 0):
            return np.inf
        z = zeta[self.rows]
        return float(np.sum(self.mu0 * t / gap) / self.n - 0.5 * self.c * np.sum(self.v0 * z * z))

    def gradient(self, zeta: np.ndarray) -> np.ndarray:
        """Its gradient in ``zeta`` (full length): ``K'(t) / N - c v0 zeta`` on the rows."""
        out = np.zeros(zeta.size)
        if self.rows.size:
            gap = 1 - self.w * self._t(zeta)
            out[self.rows] = self.mu0 / gap**2 / self.N - self.c * self.v0 * zeta[self.rows]
        return out

    def curvature(self, zeta: np.ndarray) -> np.ndarray:
        """``K''(t) = 2 mu0 w / (1 - w t)^3`` per row: the variance in place of ``v0``."""
        gap = 1 - self.w * self._t(zeta)
        return 2 * self.mu0 * self.w / gap**3

    def sigma_at(self, sigma, zeta: np.ndarray):
        """``sigma`` with these rows' ``v`` replaced by ``K''(t)``: the dual's
        Hessian at ``zeta`` is then the Gaussian one's built from it (the rows
        have no low-rank part, so ``D`` there is ``v``)."""
        if not self.rows.size:
            return sigma
        v = np.array(sigma.v, dtype=float)
        v[self.rows] = v[self.rows] - self.v0 + self.curvature(zeta)
        return replace(sigma, v=v)

    # -- torch, for VB and HMC --------------------------------------------

    def torch_term(self, device: str, dtype) -> "TorchTweedieZeros":
        import torch

        def tensor(x):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

        return TorchTweedieZeros(rows=torch.as_tensor(self.rows, device=device),
                                 mu0=tensor(self.mu0), v0=tensor(self.v0), w=self.w,
                                 n=self.n, N=self.N)


@dataclass
class TorchTweedieZeros:
    """:class:`TweedieZeros` on ``(batch, rows)`` tensors of ``zeta``."""

    rows: object
    mu0: object
    v0: object
    w: float
    n: int
    N: float

    def feasible(self, zeta):
        import torch

        if not self.rows.numel():
            return torch.ones(zeta.shape[0], dtype=torch.bool, device=zeta.device)
        return (self.w * self.n * zeta[:, self.rows] / self.N < 1).all(1)

    def __call__(self, zeta, extend: bool = False):
        """Per draw, the change from the Gaussian term: ``inf`` past the bound,
        or with ``extend`` the quadratic continuation past ``EXTEND / w``."""
        import torch

        if not self.rows.numel():
            return torch.zeros(zeta.shape[0], dtype=zeta.dtype, device=zeta.device)
        z = zeta[:, self.rows]
        t = self.n * z / self.N
        c = self.n / self.N**2
        gaussian = 0.5 * c * (self.v0 * z * z).sum(1)
        if extend:
            t_e = EXTEND / self.w
            tc = torch.clamp(t, max=t_e)
            gap = 1 - self.w * tc
            k = self.mu0 * tc / gap
            over = torch.clamp(t - t_e, min=0.0)
            gap_e = 1 - EXTEND
            k1 = self.mu0 / gap_e**2
            k2 = 2 * self.mu0 * self.w / gap_e**3
            k = k + k1 * over + 0.5 * k2 * over * over
            return k.sum(1) / self.n - gaussian
        gap = 1 - self.w * t
        safe = torch.clamp(gap, min=torch.finfo(zeta.dtype).tiny)
        k = torch.where(gap > 0, self.mu0 * t / safe, torch.full_like(t, float("inf")))
        return k.sum(1) / self.n - gaussian
