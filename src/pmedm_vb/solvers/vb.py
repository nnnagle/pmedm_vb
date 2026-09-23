"""Variational Bayes solver -- the method the paper is about.

**Target.** The dual objective read as a log posterior over the multipliers,

.. math::

    \\pi(\\lambda) \\propto \\exp(-n f(\\lambda)), \\qquad
    f(\\lambda) = Y'\\lambda/N + \\log q'e^{-X\\lambda}
                  + \\tfrac12 c\\, \\lambda'\\Sigma\\lambda .

Up to constants, ``-n f`` is the log posterior of the natural parameter of
the MaxEnt family ``p(lambda) = q e^{-X lambda} / Z`` given ``n`` observations
whose average statistic is ``Y/N``, under the prior
``lambda ~ N(0, (N/n)^2 Sigma^{-1})``: the Gaussian slack on the constraints in
the primal is a Gaussian prior on the multipliers in the dual. Its mode is the
MAP solution of :mod:`~pmedm_vb.solvers.map_dual`, and its Laplace
approximation is ``N(lambda*, (n H)^{-1})`` with ``H`` the converged dual
Hessian. That makes the comparison with Laplace one of approximations to the
*same* posterior. It is not the derivation's generative model (multinomial
sample from ``q``, noisy ``Y``), which shares the MAP but not the uncertainty;
the README lists which Laplace baseline the paper uses as an open question.

**Family.** A Gaussian whose precision has the shape of ``n H``:

.. math::

    q(\\lambda) = N(\\mu, (G G')^{-1}), \\qquad
    G = \\mathrm{blockdiag}_t(L_t)\\, (I + W V')

with ``L_t`` lower triangular over tract ``t``'s rows (the same blocks the MAP
Hessian factorises) and ``W, V`` of rank ``r``. ``n H = n K + n w J w'`` --
per-tract blocks plus a rank-1 downdate, or rank 81 untapered -- is exactly a
member: ``L_t`` the Cholesky factors of ``n K_t`` and ``I + W V'`` the symmetric
square root of ``I + Z J Z'`` for ``Z = L^{-1} w sqrt(n)``. The fit starts
there, so the Laplace approximation is the initial point and the ELBO can only
improve on it.

**Skewed family.** On Knox the Gaussian leaves a tail: about 1 percent of its
draws run thousands of nats into a wall. ``experiments/laplace_diagnostic.py``
traced the wall to multipliers of cells with small fitted counts, where ``-n f``
behaves like a Poisson log-likelihood in a log-rate -- exponential on one side,
nearly linear on the other -- which no symmetric family can follow. With
``family="skewed"`` each coordinate is then passed through a monotone
sinh-arcsinh map (Jones & Pewsey 2009, *Biometrika* 96),

.. math::

    \\lambda_k = \\mu_k + s_k T_k(x_k / s_k), \\qquad
    T(z) = \\sinh\\big((\\operatorname{asinh} z + a_k) / b_k\\big)
           - \\sinh(a_k / b_k)

with ``x = G'^{-1} eps`` the Gaussian part, ``a_k`` a skew, ``b_k > 0`` a tail
weight and ``s_k`` the Gaussian marginal sd, fixed so ``T`` sees a unit-scale
argument. ``T(0) = 0`` keeps the median at ``mu_k``; ``a = 0, b = 1`` is the
identity, so the Gaussian is a member and the skewed fit starts from it. The
map is monotone and coordinatewise, so the entropy is the Gaussian's plus
``E[sum_k log T_k'(x_k / s_k)]``, estimated from the same draws as the rest of
the ELBO. It is fitted in a second stage, after the Gaussian has converged,
so each run reports Laplace, Gaussian and skewed ELBOs. Sinh-arcsinh bends
both tails but is not the log-gamma shape the Poisson argument predicts; how
much of the tail it removes is for the diagnostic to say.

Draws are ``lambda = mu + G'^{-1} eps``: an ``r x r`` Woodbury solve for
``(I + V W')^{-1}``, then one triangular solve per tract. The entropy is
``m/2 log(2 pi e) - sum log diag L_t - log|det(I + V'W)|``.

**Objective.** ``ELBO = E_q[-n f(lambda)] + H[q]``, a lower bound on
``log integral exp(-n f)``. ``E_q[log q'e^{-X lambda}]`` has no closed form, so
the expectation is estimated from reparameterised draws and differentiated by
torch, as in automatic-differentiation VI (Kucukelbir et al. 2017, *JMLR*
18). The estimate is unbiased and the optimisation is stochastic (Adam).

**Parameterisation.** ``mu = lambda* + L_0'^{-1} delta`` and
``L_t = L_{0,t} T_t`` with ``T_t`` lower triangular and a log-parameterised
diagonal, all starting at the Laplace factor ``L_0`` (``delta = 0``,
``T = I``). A unit step in ``delta`` or ``T`` is then about one posterior
standard deviation in every direction, which is what makes a single Adam
learning rate workable across multipliers whose scales differ by orders of
magnitude.

**Stopping.** A single draw's ``-n f`` varies by roughly ``sqrt(m / 2)`` about
its mean -- about 65 at ``m = 8,000`` -- so a relative tolerance on the raw
ELBO cannot be met. Instead the ELBO is averaged over consecutive windows, and
a window that improves on the one before by less than ``tol`` standard errors
of the difference counts as a stall. Each stall halves the learning rate, and
the fit stops after ``patience`` of them. The reported fit is the average of
the parameters over the last window (Polyak-Ruppert averaging), not the last
iterate. Both answer the same problem: with a constant step Adam's iterates
keep a fixed amount of jitter about the optimum, and on a synthetic problem
where the Laplace start was already nearly optimal the last iterate's ELBO
fell below that start. A single-window stopping rule, tried first, also
stopped with two thirds of the gain a long run reaches.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.sigma import Sigma
from pmedm_vb.progress import logger
from pmedm_vb.solvers.base import dual_state, weights_from_lambda
from pmedm_vb.solvers.map_dual import DualHessian, MAPResult, solve_map

DTYPE = torch.float64

#: Draws used to fix the skewed stage's per-coordinate scale ``s``. Any fixed
#: positive ``s`` gives a valid family; it only needs to be about right.
SCALE_DRAWS = 2000


def sinh_arcsinh(z, skew, log_tail):
    """``T(z) = sinh((asinh z + a) / b) - sinh(a / b)`` and ``log T'(z)``.

    Works on numpy arrays and torch tensors alike, broadcasting ``skew`` and
    ``log_tail`` (``b = exp(log_tail)``) over trailing draws.
    """
    lib = torch if isinstance(z, torch.Tensor) else np
    tail = lib.exp(log_tail)
    inner = (lib.arcsinh(z) + skew) / tail
    value = lib.sinh(inner) - lib.sinh(skew / tail)
    log_derivative = lib.log(lib.cosh(inner)) - log_tail - 0.5 * lib.log1p(z * z)
    return value, log_derivative


def sinh_arcsinh_inverse(y, skew, log_tail):
    """``T^{-1}``: ``sinh(b asinh(y + sinh(a / b)) - a)``."""
    tail = np.exp(log_tail)
    return np.sinh(tail * np.arcsinh(y + np.sinh(skew / tail)) - skew)


@dataclass
class StructuredGaussian:
    """``N(mean, (G G')^{-1})`` with ``G = blockdiag(blocks) (I + W V')``.

    Attributes
    ----------
    mean:
        ``(m,)``.
    rows:
        Stacked-row indices of each block, one array per tract.
    blocks:
        Lower-triangular factors, one per tract, aligned with ``rows``.
    W, V:
        ``(m, r)`` low-rank factors.
    skew, log_tail, scale:
        ``(m,)`` sinh-arcsinh parameters and the fixed scale ``s``, or ``None``
        for the plain Gaussian. See *Skewed family* in the module docstring;
        with them, ``mean`` is the median of each coordinate, not its mean.
    """

    mean: np.ndarray
    rows: list[np.ndarray]
    blocks: list[np.ndarray]
    W: np.ndarray
    V: np.ndarray
    skew: np.ndarray | None = None
    log_tail: np.ndarray | None = None
    scale: np.ndarray | None = None

    @property
    def is_skewed(self) -> bool:
        return self.skew is not None

    @property
    def size(self) -> int:
        return self.mean.size

    def _block_apply(self, x: np.ndarray, transpose: bool) -> np.ndarray:
        out = np.empty_like(x)
        for rows, block in zip(self.rows, self.blocks):
            out[rows] = (block.T if transpose else block) @ x[rows]
        return out

    def _block_solve_transpose(self, x: np.ndarray) -> np.ndarray:
        from scipy.linalg import solve_triangular

        out = np.empty_like(x)
        for rows, block in zip(self.rows, self.blocks):
            out[rows] = solve_triangular(block, x[rows], trans="T", lower=True)
        return out

    def precision_matvec(self, x: np.ndarray) -> np.ndarray:
        """``G G' x``."""
        y = self._block_apply(x, transpose=True)          # L' x
        y = y + self.V @ (self.W.T @ y)                   # (I + V W') L' x = G' x
        y = y + self.W @ (self.V.T @ y)                   # (I + W V') ...
        return self._block_apply(y, transpose=False)      # L ...

    def gaussian_part(self, eps: np.ndarray) -> np.ndarray:
        """``G'^{-1} eps``, the zero-mean Gaussian offsets before any skewing."""
        r = self.W.shape[1]
        small = np.eye(r) + self.W.T @ self.V
        y = eps - self.V @ np.linalg.solve(small, self.W.T @ eps)
        return self._block_solve_transpose(y)

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """``(m, size)`` draws: ``mean + G'^{-1} eps``, skewed if the family is."""
        x = self.gaussian_part(rng.standard_normal((self.size, size)))
        if self.is_skewed:
            s = self.scale[:, None]
            x = s * sinh_arcsinh(x / s, self.skew[:, None], self.log_tail[:, None])[0]
        return self.mean[:, None] + x

    def log_density(self, lam: np.ndarray) -> np.ndarray:
        """``log q`` at each column of ``lam`` (m x k)."""
        x = lam - self.mean[:, None]
        log_jacobian = 0.0
        if self.is_skewed:
            s = self.scale[:, None]
            z = sinh_arcsinh_inverse(x / s, self.skew[:, None], self.log_tail[:, None])
            log_jacobian = sinh_arcsinh(z, self.skew[:, None], self.log_tail[:, None])[1].sum(0)
            x = s * z
        quadratic = np.einsum("id,id->d", x, self.precision_matvec(x))
        gaussian = -0.5 * self.size * math.log(2 * math.pi) + 0.5 * self.logdet_precision() - 0.5 * quadratic
        return gaussian - log_jacobian

    def logdet_precision(self) -> float:
        """``log det(G G')``."""
        blocks = sum(2.0 * np.log(np.diag(b)).sum() for b in self.blocks)
        r = self.W.shape[1]
        _, low_rank = np.linalg.slogdet(np.eye(r) + self.V.T @ self.W)
        return float(blocks + 2.0 * low_rank)

    def entropy(self) -> float:
        """Entropy of the Gaussian part; the skew adds ``E[sum log T']``."""
        return 0.5 * self.size * math.log(2 * math.pi * math.e) - 0.5 * self.logdet_precision()

    @classmethod
    def laplace(cls, inputs: PMEDMInputs, result: MAPResult) -> "StructuredGaussian":
        """``N(lambda*, (n H)^{-1})`` written in this family."""
        sigma = inputs.sigma(result.alpha, result.taper, result.variance_floor)
        hessian = DualHessian(inputs, dual_state(inputs, result.lam, sigma), sigma)
        n = inputs.n
        blocks = [np.linalg.cholesky(n * block) for block in hessian.blocks]
        family = cls(result.lam.copy(), hessian.rows, blocks,
                     np.zeros((result.lam.size, 0)), np.zeros((result.lam.size, 0)))
        # I + Z J Z' with Z = L^{-1} w sqrt(n); its symmetric square root is
        # I + Q E diag(sqrt(1 + Lambda) - 1) E' Q' for Z = Q R, R J R' = E Lambda E'.
        z = np.empty_like(hessian.w)
        for rows, block in zip(hessian.rows, blocks):
            z[rows] = np.linalg.solve(block, np.sqrt(n) * hessian.w[rows])
        q_basis, r_factor = np.linalg.qr(z)
        eigenvalues, vectors = np.linalg.eigh(r_factor @ np.diag(hessian.signs) @ r_factor.T)
        if eigenvalues.min() <= -1:
            raise ValueError("n H is not positive definite at the MAP solution")
        family.W = q_basis @ vectors
        family.V = family.W * (np.sqrt(1.0 + eigenvalues) - 1.0)
        return family


@dataclass
class VBResult:
    """Converged variational fit.

    Attributes
    ----------
    q:
        The fitted :class:`StructuredGaussian` over ``lambda``.
    params:
        ``mean``, ``W`` and ``V`` as arrays, for saving; the per-tract blocks
        are on ``q``.
    elbo:
        ELBO of ``q``, estimated with ``final_draws`` draws.
    elbo_se:
        Monte Carlo standard error of ``elbo``.
    laplace_elbo:
        The same estimate for the Laplace approximation the fit started from,
        with its standard error in ``laplace_elbo_se``. ``elbo`` above it by
        more than a few standard errors means VB found a better Gaussian.
    elbo_trace:
        The single-batch ELBO estimate at each iteration -- noisy by design;
        see *Stopping* in the module docstring.
    n_iter, converged:
        As for :class:`~pmedm_vb.solvers.map_dual.MAPResult`.
    alpha, taper, variance_floor:
        The ``Sigma`` the posterior is under.
    family:
        ``"gaussian"`` or ``"skewed"``.
    gaussian_elbo, gaussian_elbo_se:
        The ELBO of the Gaussian fit the skewed stage started from; for the
        Gaussian family the same as ``elbo``. ``elbo`` above it by more than a
        few standard errors means the skew found a closer approximation.
    """

    q: StructuredGaussian
    params: dict[str, np.ndarray]
    elbo: float
    elbo_se: float
    laplace_elbo: float
    laplace_elbo_se: float
    elbo_trace: np.ndarray
    n_iter: int
    converged: bool
    alpha: float
    taper: str | None
    map_result: MAPResult | None = field(default=None, repr=False)
    variance_floor: str | float | None = None
    family: str = "gaussian"
    gaussian_elbo: float | None = None
    gaussian_elbo_se: float | None = None


class _DualTarget:
    """``f(lambda)`` for a batch of multipliers, in torch."""

    def __init__(self, inputs: PMEDMInputs, sigma: Sigma, device: str) -> None:
        def tensor(x):
            return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=device)

        a_b = inputs.A_B.toarray() if hasattr(inputs.A_B, "toarray") else np.asarray(inputs.A_B)
        if not np.array_equal(a_b, np.eye(inputs.n_zones)):
            raise ValueError("the VB target assumes A_B is the identity")
        with np.errstate(divide="ignore"):
            self.log_q = tensor(np.log(inputs.q))
        self.X_T = tensor(inputs.X_T.toarray())
        self.X_B = tensor(inputs.X_B.toarray())
        self.A_T = tensor(inputs.A_T.toarray())
        self.y = tensor(inputs.targets() / inputs.N)
        self.c = inputs.n / inputs.N**2
        self.split = inputs.Y_T.size
        self.shape_T = inputs.Y_T.shape
        self.shape_B = inputs.Y_B.shape
        self.d = tensor(sigma.d)
        self.b = None if sigma.b is None else tensor(sigma.b)
        groups = np.zeros(inputs.n_constraints, int) if sigma.groups is None else sigma.groups
        _, inverse = np.unique(groups, return_inverse=True)
        self.groups = torch.as_tensor(inverse, device=device)
        self.n_groups = int(inverse.max()) + 1

    def adjoint(self, lam: torch.Tensor) -> torch.Tensor:
        """``X lambda`` as ``(batch, n_zones, n_units)``, column-major like the numpy one."""
        batch = lam.shape[0]
        (n_t, c_t), (n_b, c_b) = self.shape_T, self.shape_B
        v_t = lam[:, : self.split].reshape(batch, c_t, n_t).transpose(1, 2)
        v_b = lam[:, self.split :].reshape(batch, c_b, n_b).transpose(1, 2)
        return self.A_T.T @ (v_t @ self.X_T.T) + v_b @ self.X_B.T

    def sigma_matvec(self, lam: torch.Tensor) -> torch.Tensor:
        out = self.d * lam
        if self.b is None:
            return out
        batch = lam.shape[0]
        sums = torch.zeros(batch, self.n_groups, self.b.shape[1], dtype=DTYPE, device=lam.device)
        sums = sums.index_add(1, self.groups, self.b[None] * lam[:, :, None])
        return out + (self.b[None] * sums[:, self.groups, :]).sum(-1)

    def __call__(self, lam: torch.Tensor) -> torch.Tensor:
        logits = self.log_q[None] - self.adjoint(lam)
        log_z = torch.logsumexp(logits.reshape(lam.shape[0], -1), dim=1)
        return lam @ self.y + log_z + 0.5 * self.c * (lam * self.sigma_matvec(lam)).sum(1)


class _Variational(torch.nn.Module):
    """The family in whitened coordinates about a starting ``StructuredGaussian``."""

    def __init__(self, start: StructuredGaussian, device: str, skewed: bool = False) -> None:
        super().__init__()

        def tensor(x):
            return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=device)

        self.m = start.size
        self.rows = [torch.as_tensor(r, device=device) for r in start.rows]
        self.mean0 = tensor(start.mean)
        self.blocks0 = [tensor(b) for b in start.blocks]
        self.delta = torch.nn.Parameter(torch.zeros(self.m, dtype=DTYPE, device=device))
        self.lower = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros_like(b)) for b in self.blocks0]
        )
        self.log_diag = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(b.shape[0], dtype=DTYPE, device=device))
             for b in self.blocks0]
        )
        self.W = torch.nn.Parameter(tensor(start.W))
        self.V = torch.nn.Parameter(tensor(start.V))
        self.skewed = skewed
        if skewed:
            # Identity at the start: the skewed family begins exactly at ``start``.
            self.skew = torch.nn.Parameter(torch.zeros(self.m, dtype=DTYPE, device=device))
            self.log_tail = torch.nn.Parameter(torch.zeros(self.m, dtype=DTYPE, device=device))
            if start.scale is None:
                raise ValueError("a skewed family needs start.scale, the Gaussian marginal sd")
            self.scale = tensor(start.scale)

    def blocks(self) -> list[torch.Tensor]:
        return [
            b0 @ (torch.tril(a, -1) + torch.diag(torch.exp(s)))
            for b0, a, s in zip(self.blocks0, self.lower, self.log_diag)
        ]

    def _solve_transpose(self, blocks, x: torch.Tensor) -> torch.Tensor:
        """``L'^{-1} x`` blockwise, for ``x`` of shape ``(m, batch)``."""
        out = torch.empty_like(x)
        for rows, block in zip(self.rows, blocks):
            out[rows] = torch.linalg.solve_triangular(block.T, x[rows], upper=True)
        return out

    def mean(self) -> torch.Tensor:
        return self.mean0 + self._solve_transpose(self.blocks0, self.delta[:, None])[:, 0]

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(count, m)`` reparameterised draws and each draw's ``sum log T'``.

        The log-Jacobian is zero for the Gaussian family; skewed, it is the
        term that turns the Gaussian part's entropy into the family's.
        """
        blocks = self.blocks()
        eps = torch.randn(self.m, count, dtype=DTYPE, device=self.delta.device)
        r = self.W.shape[1]
        if r:
            small = torch.eye(r, dtype=DTYPE, device=eps.device) + self.W.T @ self.V
            eps = eps - self.V @ torch.linalg.solve(small, self.W.T @ eps)
        x = self._solve_transpose(blocks, eps)
        log_jacobian = torch.zeros(count, dtype=DTYPE, device=eps.device)
        if self.skewed:
            s = self.scale[:, None]
            value, log_derivative = sinh_arcsinh(x / s, self.skew[:, None], self.log_tail[:, None])
            x = s * value
            log_jacobian = log_derivative.sum(0)
        return (self.mean()[:, None] + x).T, log_jacobian

    def entropy(self) -> torch.Tensor:
        logdet = sum(
            torch.log(torch.diagonal(b0)).sum() + s.sum()
            for b0, s in zip(self.blocks0, self.log_diag)
        )
        r = self.W.shape[1]
        if r:
            small = torch.eye(r, dtype=DTYPE, device=self.W.device) + self.V.T @ self.W
            logdet = logdet + torch.linalg.slogdet(small)[1]
        return 0.5 * self.m * math.log(2 * math.pi * math.e) - logdet

    def export(self) -> StructuredGaussian:
        with torch.no_grad():
            skewed = {}
            if self.skewed:
                skewed = dict(
                    skew=self.skew.detach().cpu().numpy(),
                    log_tail=self.log_tail.detach().cpu().numpy(),
                    scale=self.scale.cpu().numpy(),
                )
            return StructuredGaussian(
                mean=self.mean().cpu().numpy(),
                rows=[r.cpu().numpy() for r in self.rows],
                blocks=[b.cpu().numpy() for b in self.blocks()],
                W=self.W.detach().cpu().numpy(),
                V=self.V.detach().cpu().numpy(),
                **skewed,
            )


def _parameter_groups(model: "_Variational", learning_rate: float) -> list[dict]:
    """Adam steps scaled to each parameter's share of the whole.

    Adam moves every coordinate by about the learning rate whatever its
    gradient, so a group of ``k`` noise-dominated coordinates takes a random
    step of norm about ``lr * sqrt(k)``. For the mean and the log-diagonal that
    is a per-coordinate change in posterior standard deviations, which is what
    ``lr`` is meant to be. A block's off-diagonal entries and the low-rank
    columns act on a vector through a whole row or column, so their steps are
    divided by the row length -- otherwise a million block entries, each moving
    by ``lr``, perturb the covariance far more than any single coordinate.
    """
    groups = [
        {"params": [model.delta], "lr": learning_rate},
        {"params": list(model.log_diag), "lr": learning_rate},
        {"params": [model.W, model.V], "lr": learning_rate / math.sqrt(model.m)},
    ]
    for block in model.lower:
        groups.append({"params": [block], "lr": learning_rate / block.shape[0]})
    if model.skewed:
        # One pair per coordinate, each acting on a unit-scale argument.
        groups.append({"params": [model.skew, model.log_tail], "lr": learning_rate})
    return groups


def _elbo_terms(model: _Variational, target: _DualTarget, n: int, count: int) -> torch.Tensor:
    """Per-draw ``-n f(lambda) + H[q]``, whose mean is the ELBO.

    ``H[q]`` is the Gaussian part's entropy plus, skewed, the draw's
    ``sum log T'``, whose expectation is the rest of it.
    """
    lam, log_jacobian = model.sample(count)
    return -n * target(lam) + model.entropy() + log_jacobian


def _estimate(model, target, n, draws, batch=16) -> tuple[float, float]:
    """ELBO and its Monte Carlo standard error from ``draws`` draws."""
    with torch.no_grad():
        values = torch.cat([
            _elbo_terms(model, target, n, min(batch, draws - i)) for i in range(0, draws, batch)
        ]).cpu().numpy()
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def solve_vb(
    inputs: PMEDMInputs,
    *,
    alpha: float,
    taper: str | None = "tract",
    init: MAPResult | None = None,
    variance_floor: str | float | None = None,
    family: str = "gaussian",
    max_iter: int = 1000,
    tol: float = 2.0,
    draws: int = 8,
    learning_rate: float = 0.02,
    window: int = 50,
    patience: int = 4,
    final_draws: int = 400,
    seed: int = 0,
    device: str = "cpu",
) -> VBResult:
    """Fit the variational posterior.

    Parameters
    ----------
    alpha, taper, variance_floor:
        Select ``Sigma``, as for :func:`~pmedm_vb.solvers.map_dual.solve_map`.
    init:
        MAP fit to start from; solved here when ``None``. Its ``alpha``,
        ``taper`` and ``variance_floor`` must match.
    tol, patience:
        A window whose mean ELBO beats the previous window's by less than
        ``tol`` standard errors of the difference halves the learning rate;
        the fit stops after ``patience`` halvings.
    draws:
        Monte Carlo draws per gradient step.
    learning_rate:
        Adam step, in posterior standard deviations (see *Parameterisation*).
    window:
        Iterations per ELBO window for the stopping rule.
    family:
        ``"gaussian"``, or ``"skewed"`` to add a second stage fitting a
        sinh-arcsinh skew per coordinate on top of the converged Gaussian (see
        *Skewed family*). ``max_iter`` applies to each stage.
    final_draws:
        Draws for the reported ELBOs of the fit, the Gaussian stage and the
        Laplace start.
    device:
        A torch device, e.g. ``"cuda"``.
    """
    if family not in ("gaussian", "skewed"):
        raise ValueError(f"family must be 'gaussian' or 'skewed', got {family!r}")
    if init is None:
        init = solve_map(inputs, alpha=alpha, taper=taper, variance_floor=variance_floor)
    if (init.alpha, init.taper, init.variance_floor) != (alpha, taper, variance_floor):
        raise ValueError(
            f"init was fitted at alpha={init.alpha}, taper={init.taper}, "
            f"variance_floor={init.variance_floor}, not alpha={alpha}, "
            f"taper={taper}, variance_floor={variance_floor}"
        )
    torch.manual_seed(seed)
    started = time.perf_counter()
    target = _DualTarget(inputs, inputs.sigma(alpha, taper, variance_floor), device)
    model = _Variational(StructuredGaussian.laplace(inputs, init), device)
    n = inputs.n

    laplace_elbo, laplace_se = _estimate(model, target, n, final_draws)
    logger.info(
        "solve_vb PUMA %s: %s multipliers, alpha=%s, taper=%s, family=%s, "
        "Laplace ELBO %.2f +- %.2f",
        inputs.puma, f"{model.m:,}", alpha, taper, family, laplace_elbo, laplace_se,
    )
    settings = dict(learning_rate=learning_rate, draws=draws, max_iter=max_iter,
                    tol=tol, window=window, patience=patience, started=started)
    trace, n_iter, converged = _optimise(model, target, n, **settings)
    gaussian_elbo, gaussian_se = _estimate(model, target, n, final_draws)
    elbo_value, elbo_se = gaussian_elbo, gaussian_se
    logger.info(
        "  Gaussian stage: %s after %d iterations, ELBO %.2f +- %.2f",
        "converged" if converged else "NOT converged", n_iter, gaussian_elbo, gaussian_se,
    )

    if family == "skewed":
        start = model.export()
        rng = np.random.default_rng(seed)
        start.scale = start.gaussian_part(rng.standard_normal((model.m, SCALE_DRAWS))).std(
            axis=1, ddof=1
        )
        model = _Variational(start, device, skewed=True)
        skew_trace, skew_iter, skew_converged = _optimise(model, target, n, **settings)
        trace += skew_trace
        n_iter += skew_iter
        converged = converged and skew_converged
        elbo_value, elbo_se = _estimate(model, target, n, final_draws)
        logger.info(
            "  skewed stage: %s after %d iterations, ELBO %.2f +- %.2f",
            "converged" if skew_converged else "NOT converged", skew_iter, elbo_value, elbo_se,
        )

    q = model.export()
    logger.info(
        "solve_vb PUMA %s: %s after %d iterations, ELBO %.2f +- %.2f "
        "(Gaussian %.2f +- %.2f, Laplace %.2f +- %.2f), %.1fs",
        inputs.puma, "converged" if converged else "NOT converged", n_iter,
        elbo_value, elbo_se, gaussian_elbo, gaussian_se, laplace_elbo, laplace_se,
        time.perf_counter() - started,
    )
    params = {"mean": q.mean, "W": q.W, "V": q.V}
    if q.is_skewed:
        params.update(skew=q.skew, log_tail=q.log_tail, scale=q.scale)
    return VBResult(
        q=q,
        params=params,
        elbo=elbo_value,
        elbo_se=elbo_se,
        laplace_elbo=laplace_elbo,
        laplace_elbo_se=laplace_se,
        elbo_trace=np.asarray(trace),
        n_iter=n_iter,
        converged=converged,
        alpha=alpha,
        taper=taper,
        map_result=init,
        variance_floor=variance_floor,
        family=family,
        gaussian_elbo=gaussian_elbo,
        gaussian_elbo_se=gaussian_se,
    )


def _optimise(
    model: _Variational,
    target: _DualTarget,
    n: int,
    *,
    learning_rate: float,
    draws: int,
    max_iter: int,
    tol: float,
    window: int,
    patience: int,
    started: float,
) -> tuple[list[float], int, bool]:
    """Adam with window stalls, learning-rate halving and parameter averaging.

    Leaves ``model`` at its parameters averaged over the last window (see
    *Stopping*) and returns the per-iteration ELBO trace, the iterations taken
    and whether it converged before ``max_iter``.
    """
    parameters = list(model.parameters())
    optimiser = torch.optim.Adam(_parameter_groups(model, learning_rate))
    running = [torch.zeros_like(p) for p in parameters]
    averaged = [p.detach().clone() for p in parameters]
    in_window = 0
    stalled = 0
    trace: list[float] = []
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        optimiser.zero_grad()
        elbo = _elbo_terms(model, target, n, draws).mean()
        (-elbo).backward()
        optimiser.step()
        trace.append(float(elbo.detach()))
        with torch.no_grad():
            for total, parameter in zip(running, parameters):
                total += parameter
        in_window += 1
        if n_iter % window == 0:
            averaged = [total / in_window for total in running]
            running = [torch.zeros_like(p) for p in parameters]
            in_window = 0
        if n_iter % window == 0 and n_iter >= 2 * window:
            last = np.asarray(trace[-window:])
            before = np.asarray(trace[-2 * window : -window])
            gain = last.mean() - before.mean()
            se = np.sqrt(last.var(ddof=1) / window + before.var(ddof=1) / window)
            logger.info(
                "  iter %5d  window ELBO %.2f  gain %.2f  (%.1f se)  %.1fs",
                n_iter, last.mean(), gain, gain / se, time.perf_counter() - started,
            )
            if gain < tol * se:
                stalled += 1
                if stalled >= patience:
                    converged = True
                    break
                for group in optimiser.param_groups:
                    group["lr"] *= 0.5

    with torch.no_grad():
        final = [total / in_window for total in running] if in_window else averaged
        for parameter, value in zip(parameters, final):
            parameter.copy_(value)
    return trace, n_iter, converged


def posterior_weights(
    inputs: PMEDMInputs,
    result: VBResult,
    *,
    n_draws: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw population weight matrices from the fitted variational posterior.

    Each draw is ``lambda ~ q`` mapped through the MaxEnt family,
    ``W = N q e^{-X lambda} / Z``.

    Returns
    -------
    numpy.ndarray
        ``(n_draws, n_zones, n_units)``, each summing to ``N``.
    """
    rng = rng or np.random.default_rng()
    lams = result.q.sample(rng, n_draws)
    return np.stack([inputs.N * weights_from_lambda(inputs, lam) for lam in lams.T])
