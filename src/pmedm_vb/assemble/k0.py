"""A count-shaped error on every row with Sigma's correlations: the ``+k0w<w>``
level suffix.

In the dual (``pmedm_derivation.md``) the rows' Gaussian error enters as
``(1/n) K(n zeta / N)`` with ``K(t) = Y't + t' Sigma t / 2``, the cumulant
generating function of ``N(Y, Sigma)``. Taking the dual as a model in its own
right (as :mod:`pmedm_vb.assemble.tweedie` does for the published zeros), this
replaces it on every row by

.. math::

    K_0(t) = a' T (I - 2 S T)^{-1} a, \\qquad T = \\mathrm{diag}(t),
    \\qquad S^{-1} - 2T \\succ 0,

with ``a = sqrt(Y)`` (on rows published as zero, ``a^2 = mu0 = v / (2 w)`` for
the average weight ``w``, as the Tweedie zero term has it) and
``S = A^-1 Sigma A^-1 / 4``, ``A = diag(a)``. Then ``K0'(0) = a^2``,
``K0''(0) = 4 A S A = Sigma`` exactly, each row's marginal is a Tweedie with
``p = 1.5`` -- ``Y t / (1 - w_i t)``, ``w_i = 2 S_ii = v_i / (2 Y_i)``, the
Tweedie zero term on a zero row without replicate spread -- and for large counts
``K0 -> Y't + t' Sigma t / 2``. ``K0`` is convex; it is the zero-degrees-of-
freedom member of ``m = sum_r u_r o u_r``, ``u_r ~ N(a_r, S)``, and the
conjugate of the square-root-scale Gaussian penalty, so its primal penalty is
that penalty's convex envelope (``OPEN_QUESTIONS.md``, #5).

With ``S = D_S + B_S B_S'`` (``Sigma``'s diagonal plus its rank-80 factor, per
taper group), ``f = 1 - 2 d_S t`` and ``R = I - 2 B_S' diag(t / f) B_S`` (80 x 80),
the domain is ``f > 0`` and ``R`` positive definite, and

.. math::

    g = (I - 2ST)^{-1} a = a / f + 2 F^{-1} B_S R^{-1} B_S' (t a / f),
    \\quad K_0 = \\sum_i g_i t_i a_i, \\quad \\nabla K_0 = g \\circ g,
    \\quad \\nabla^2 K_0 = 4 G (S^{-1} - 2T)^{-1} G,

``(S^-1 - 2T)^-1 = diag(d_S / f) + (F^-1 B_S) R^-1 (F^-1 B_S)'``: the Hessian is
again a diagonal plus rank 80 per group, which the MAP's Newton solver takes as
it takes ``Sigma``. Every evaluation costs ``O(rows * 80^2)``.

The prior's ``tau^2`` stays a Gaussian prior on the multipliers: ``S`` is built
from ``Sigma`` without it. Past the domain the target is infinite: HMC rejects
any trajectory that reaches it and the MAP's line search backs off. There is no
continuation for VB (the boundary is joint, not per row), so VB with this term
raises.

It travels with the hierarchy level, in the slot the Tweedie zero term uses
(``Hierarchy.tweedie``): ``"puma-rpruned+prior3+k0w18"``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from pmedm_vb.assemble.sigma import Sigma

SEPARATOR = "+k0w"


def split_k0(level: str) -> tuple[str, float | None]:
    """``"puma+prior3+k0w18"`` -> ``("puma+prior3", 18.0)``; none -> ``(level, None)``."""
    if SEPARATOR not in level:
        return level, None
    head, _, rest = level.partition(SEPARATOR)
    spec, plus, tail = rest.partition("+")
    try:
        w = float(spec)
    except ValueError:
        raise ValueError(f"a K0 term is +k0w<w>, e.g. +k0w18; got {level!r}") from None
    if not w > 0:
        raise ValueError(f"the K0 zero-row weight must be positive, got {level!r}")
    return head + (plus + tail if plus else ""), w


def k0_level(level: str, w: float | None) -> str:
    """The level string with the K0 term appended, for ``--k0-weight``."""
    if w is None:
        return level
    if not w > 0:
        raise ValueError(f"--k0-weight must be positive, got {w}")
    return f"{level}{SEPARATOR}{w:g}"


@dataclass
class K0Term:
    """``K0`` on every row of ``zeta``, as the change from the Gaussian term.

    ``a`` per row; ``d_s`` and ``b_s`` the diagonal and low-rank factor of ``S``
    (``b_s`` ``None`` for a diagonal ``Sigma``); ``members`` the taper groups'
    rows; ``sigma0`` the ``Sigma`` it replaces, without the prior; ``y`` the
    targets over ``N``. Every method returns the change from the Gaussian term
    ``y' zeta + (c / 2) zeta' Sigma0 zeta``, so the rest of the dual is untouched.
    """

    a: np.ndarray
    d_s: np.ndarray
    b_s: np.ndarray | None
    members: list
    sigma0: object
    y: np.ndarray
    w: float
    n: int
    N: float

    @classmethod
    def build(cls, inputs, hierarchy, w: float, alpha, taper, variance_floor=None) -> "K0Term":
        if alpha is None:
            raise ValueError("a K0 term needs alpha when the hierarchy is built: S is "
                             "built from Sigma at that alpha")
        sigma0 = replace(hierarchy.sigma(inputs, alpha, taper, variance_floor), prior_var=0.0)
        y = np.asarray(hierarchy.targets_ext, dtype=float)
        mu = np.where(y > 0, y, sigma0.v / (2 * w))
        a = np.sqrt(mu)
        b = sigma0.b
        return cls(a=a, d_s=sigma0.d / (4 * mu), b_s=None if b is None else b / (2 * a[:, None]),
                   members=sigma0.members, sigma0=sigma0, y=y / inputs.N, w=float(w),
                   n=int(inputs.n), N=float(inputs.N))

    @property
    def c(self) -> float:
        return self.n / self.N**2

    # -- numpy, for the MAP ----------------------------------------------

    def _solve(self, zeta: np.ndarray):
        """``(t, f, g, factors)`` at ``zeta``, or ``None`` outside the domain;
        ``factors`` the Cholesky factor of each group's ``R``."""
        t = self.n * zeta / self.N
        f = 1 - 2 * self.d_s * t
        if np.any(f <= 0):
            return None
        g = self.a / f
        factors = []
        if self.b_s is not None:
            for rows in self.members:
                bf = self.b_s[rows] / f[rows, None]
                r = np.eye(bf.shape[1]) - 2 * self.b_s[rows].T @ (bf * t[rows, None])
                try:
                    factor = cho_factor(r)
                except np.linalg.LinAlgError:
                    return None
                factors.append(factor)
                g[rows] += 2 * bf @ cho_solve(factor, bf.T @ (t[rows] * self.a[rows]))
        return t, f, g, factors

    def feasible(self, zeta: np.ndarray) -> bool:
        return self._solve(zeta) is not None

    def value(self, zeta: np.ndarray) -> float:
        """``(1/n) K0(t)`` less the Gaussian term; ``inf`` outside the domain."""
        solved = self._solve(zeta)
        if solved is None:
            return np.inf
        t, _, g, _ = solved
        gaussian = self.y @ zeta + 0.5 * self.c * (zeta @ self.sigma0.matvec(zeta))
        return float(np.sum(g * t * self.a) / self.n - gaussian)

    def gradient(self, zeta: np.ndarray) -> np.ndarray:
        """Its gradient in ``zeta``: ``g o g / N - y - c Sigma0 zeta``; NaN outside
        the domain (the line search never accepts such a point)."""
        solved = self._solve(zeta)
        if solved is None:
            return np.full(zeta.size, np.nan)
        g = solved[2]
        return g * g / self.N - self.y - self.c * self.sigma0.matvec(zeta)

    def sigma_at(self, sigma, zeta: np.ndarray):
        """``sigma`` with ``Sigma0`` replaced by ``K0''(t)``, keeping its prior:
        the dual's Hessian at ``zeta`` is then the Gaussian one's built from it."""
        t, f, g, factors = self._solve(zeta)
        diag = 4 * g * g * self.d_s / f
        if self.b_s is None:
            return _Factored(v=diag + sigma.prior_var, l=None, alpha=sigma.alpha,
                             groups=sigma.groups, prior_var=sigma.prior_var, diag=diag,
                             factor=None)
        factor = np.zeros_like(self.b_s)
        for rows, (chol, lower) in zip(self.members, factors):
            upper = chol if not lower else chol.T  # R = U'U
            # V = 2 G F^-1 B_S U^-1, so that V V' = 4 G F^-1 B_S R^-1 B_S' F^-1 G
            scaled = 2 * (g[rows] / f[rows])[:, None] * self.b_s[rows]
            factor[rows] = np.linalg.solve(np.triu(upper).T, scaled.T).T
        return _Factored(v=diag + np.square(factor).sum(axis=1) + sigma.prior_var, l=None,
                         alpha=sigma.alpha, groups=sigma.groups, prior_var=sigma.prior_var,
                         diag=diag, factor=factor)

    # -- torch, for HMC --------------------------------------------------

    def torch_term(self, device: str, dtype) -> "TorchK0Term":
        import torch

        def tensor(x):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

        return TorchK0Term(
            a=tensor(self.a), d_s=tensor(self.d_s),
            b_s=None if self.b_s is None else tensor(self.b_s),
            members=[torch.as_tensor(rows, device=device) for rows in self.members],
            d0=tensor(self.sigma0.d),
            b0=None if self.sigma0.b is None else tensor(self.sigma0.b),
            y=tensor(self.y), n=self.n, N=self.N)


@dataclass(frozen=True)
class _Factored(Sigma):
    """A ``Sigma`` given directly as ``diag + factor factor'`` (plus the prior):
    ``K0''`` at a point. Round-tripping it through ``v`` and ``l`` cancels
    catastrophically where ``K0''`` is large, near the boundary."""

    diag: np.ndarray = None
    factor: np.ndarray | None = None

    def __post_init__(self) -> None:
        pass

    @property
    def d(self) -> np.ndarray:
        return self.diag + self.prior_var

    @property
    def b(self) -> np.ndarray | None:
        return self.factor


@dataclass
class TorchK0Term:
    """:class:`K0Term` on ``(batch, rows)`` tensors of ``zeta``."""

    a: object
    d_s: object
    b_s: object
    members: list
    d0: object
    b0: object
    y: object
    n: int
    N: float

    def _k0(self, zeta):
        """Per draw, ``K0(t)`` and whether ``zeta`` is inside the domain."""
        import torch

        t = self.n * zeta / self.N
        f = 1 - 2 * self.d_s * t
        ok = (f > 0).all(1)
        f = torch.where(f > 0, f, torch.ones_like(f))
        g = self.a / f
        if self.b_s is not None:
            eye = torch.eye(self.b_s.shape[1], dtype=zeta.dtype, device=zeta.device)
            parts = []
            for rows in self.members:
                b, fr, tr = self.b_s[rows], f[:, rows], t[:, rows]
                r = eye - 2 * (b.T[None] * (tr / fr)[:, None, :]) @ b
                chol, info = torch.linalg.cholesky_ex(r)
                ok = ok & (info == 0)
                chol = torch.where((info == 0)[:, None, None], chol, eye)
                rhs = ((tr * self.a[rows] / fr) @ b)[..., None]
                x = torch.cholesky_solve(rhs, chol)[..., 0]
                parts.append((rows, 2 * (x @ b.T) / fr))
            g = g.clone()
            for rows, part in parts:
                g[:, rows] = g[:, rows] + part
        return (g * t * self.a).sum(1), ok

    def feasible(self, zeta):
        import torch

        with torch.no_grad():
            return self._k0(zeta)[1]

    def __call__(self, zeta, extend: bool = False):
        """Per draw, the change from the Gaussian term; ``inf`` outside the domain."""
        import torch

        if extend:
            raise NotImplementedError("the K0 term has no VB continuation: its boundary is "
                                      "joint; start HMC from the MAP instead")
        k0, ok = self._k0(zeta)
        sigma0_zeta = self.d0 * zeta
        if self.b0 is not None:
            for rows in self.members:
                b = self.b0[rows]
                sigma0_zeta = sigma0_zeta.index_add(
                    1, rows, (zeta[:, rows] @ b) @ b.T)
        c = self.n / self.N**2
        gaussian = zeta @ self.y + 0.5 * c * (zeta * sigma0_zeta).sum(1)
        value = k0 / self.n - gaussian
        return torch.where(ok, value, torch.full_like(value, float("inf")))
