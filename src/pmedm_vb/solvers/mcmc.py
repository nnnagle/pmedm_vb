"""Hamiltonian Monte Carlo on the dual posterior -- a reference for VB and Laplace.

**Target.** The same ``pi(lambda) ~ exp(-n f(lambda))`` as
:mod:`~pmedm_vb.solvers.vb`, evaluated by its batched torch ``f``. VB and
Laplace both approximate it; this samples it, so that their draws can be
checked against draws from the posterior itself. In particular: does the
posterior ever put a large share of ``N`` on one (block group, record) cell,
as 6-17% of VB draws do (``solvers_synopsis.md``)?

**Whitening.** Sampling is in coordinates ``x`` with

.. math::

    \\lambda = \\mu + A x, \\qquad A = G'^{-1},

``mu`` and ``G`` the mean and precision factor of a Gaussian -- in practice the
Gaussian part of a fitted VB family, which is tighter than Laplace exactly
along the walls. If that Gaussian were the posterior, ``x`` would be standard
normal, so one step size and trajectory length serve every coordinate. Using a
variational fit to straighten the geometry for HMC is the idea of NeuTra
(Hoffman et al. 2019, arXiv:1903.03704), there with a nonlinear flow; here the
map is linear, so its Jacobian is constant and drops out. ``A`` is formed
densely once -- ``m^2`` doubles, 225 MB at ``m = 5,304`` -- so each gradient
is two dense products on top of ``f``.

**Sampler.** Plain HMC with a leapfrog integrator and an identity mass matrix
in ``x``, many chains advanced together as one batch so that each ``f``
evaluation serves them all. The trajectory length is fixed and jittered
uniformly over ``[0.5, 1.5]`` times its nominal value each iteration, shared by
all chains, and the step size is shared too. During warmup the step size is
adapted towards a target mean acceptance probability by the dual averaging of
Hoffman & Gelman (2014, *JMLR* 15, algorithm 5, with their constants
``gamma = 0.05, t0 = 10, kappa = 0.75``); after warmup it is fixed at the
averaged value, so the chains are then a valid (time-homogeneous) Markov
chain.

**Learned diagonal.** The whitening is the mass matrix up to a diagonal: with
:meth:`HMC.set_scale`, ``lambda = mu + A diag(s) x``. Warmup can learn ``s`` in
windows, as Stan does (:func:`stan_windows`): over a window the sampler
accumulates each coordinate of ``u = s * x`` (the whitening's own
coordinates) across every chain and iteration, and at its end ``s`` becomes
their sd, shrunk towards 1 (the whitening) as Stan shrinks towards its
default, and the step size's dual averaging restarts. Pooling over chains
counts any spread between chains still in it as width, which errs wide.

**Many short chains.** :func:`nested_rhat` is the nested ``R-hat`` of
Margossian et al. (2024, *Bayesian Analysis*, definition 2.2) for ``K``
superchains of ``M`` chains that start at one point each; at one draw per
chain (``N = 1``) the threshold is ``sqrt(1 + 1/M + tau)`` (their eq. 29),
``tau`` the tolerated nonstationary variance relative to the posterior's.
:func:`nested_tau` sets ``tau`` at the level the diagnostic can resolve with
``K`` superchains, from its distribution at stationarity.

A transition whose energy error exceeds ``divergence`` nats (or is not finite)
is recorded as divergent, following Stan's convention; divergences are where
the integrator could not follow the posterior, and here are expected at the
walls.

**Delayed rejection** (``retry_factor = a > 1``). A wall is a few household-zone
cells whose weight jumps from nothing to something along some direction; a
step size small enough for every wall wastes most of each trajectory in the
bulk. So a divergent transition is retried, from the same position and
momentum, with step ``eps / a`` and the same number of steps, and the retry
accepted with the delayed-rejection probability (Tierney & Mira 1999; for HMC,
Modi, Barnett & Carpenter 2024, *Bayesian Analysis* 19, whose reference code
does the same with every rejection retried):

.. math::

    \\alpha_2 = \\min\\Big(1, e^{H_0 - H_2}\\,
        \\frac{\\rho(z_2')\\,(1 - \\alpha_1(z_2'))}{\\rho(z_0)\\,(1 - \\alpha_1(z_0))}\\Big),

``z_0 = (x_0, p_0)`` the start, ``z_2 = (x_2, p_2)`` the retry's end and
``z_2' = (x_2, -p_2)`` its reversal, ``alpha_1(z)`` the first stage's
acceptance from ``z`` (its "ghost" trajectory, step ``eps``, the same number of
steps) and ``rho(z)`` the probability of retrying from ``z``: 1 if that first
stage diverges, else 0. Retrying only divergences is a state-dependent retry
rule, so the reversal's must hold too: a retry whose ghost does not diverge is
rejected. Both maps are reversible and volume-preserving, so this keeps
``exp(-H)`` invariant. Only the retried chains are integrated again, as a
smaller batch; ``extra_leapfrog`` counts those steps (the retry and its ghost)
so that cost per gradient stays honest. During warmup the step size is adapted
to the acceptance of either stage, so walls do not drag it down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from pmedm_vb.solvers.vb import DTYPE, StructuredGaussian, _DualTarget

#: Dual averaging constants from Hoffman & Gelman (2014), section 3.2.
DA_GAMMA = 0.05
DA_T0 = 10.0
DA_KAPPA = 0.75


def largest_share(target: _DualTarget, lam: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row of ``lam`` (``(k, m)``), the largest (zone, unit) cell's share of
    ``p(lambda)`` and its flat index into the ``(n_zones, n_units)`` matrix."""
    with torch.no_grad():
        logits = target.logits(lam)
        flat = logits.reshape(lam.shape[0], -1)
        top, index = flat.max(1)
        return torch.exp(top - torch.logsumexp(flat, 1)), index


def starting_points(
    target: _DualTarget,
    mean: np.ndarray,
    A: np.ndarray,
    chains: int,
    rng: np.random.Generator,
    max_share: float = 0.01,
    device: str = "cpu",
    tries: int = 20,
    inflate: float = 1.0,
) -> np.ndarray:
    """``(chains, m)`` whitened starting points: standard normal draws (times
    ``inflate``, to start overdispersed), keeping only those whose largest cell
    holds at most ``max_share`` of ``p`` and where the target is finite (a
    Tweedie zero term's bound: a chain started past it could never move).

    A draw from the whitening Gaussian can land on a wall -- as VB draws do --
    and a chain started there diverges on every trajectory at a step size that
    suits the bulk, so it never moves. Screening the start does not stop a chain
    reaching such a region by itself; it only keeps warmup out of one.
    """
    keep = []
    mean_t = torch.as_tensor(mean, dtype=DTYPE, device=device)
    A_t = torch.as_tensor(A, dtype=DTYPE, device=device)
    for _ in range(tries):
        x = inflate * rng.standard_normal((chains, mean.size))
        lam = mean_t + torch.as_tensor(x, dtype=DTYPE, device=device) @ A_t.T
        share, _ = largest_share(target, lam)
        keep.extend(x[((share <= max_share) & target.feasible(lam)).cpu().numpy()])
        if len(keep) >= chains:
            return np.asarray(keep[:chains])
    raise RuntimeError(
        f"only {len(keep)} of {chains * tries} starting draws had a largest cell under "
        f"{max_share:.1%} of p; the whitening Gaussian is far from the posterior"
    )


def overdispersed_starts(
    target: _DualTarget,
    mean: np.ndarray,
    A: np.ndarray,
    starts: int,
    rng: np.random.Generator,
    max_share: float = 0.01,
    device: str = "cpu",
    inflate: float = 1.5,
    shrink: float = 0.95,
    batch: int | None = None,
    floor: float = 1.0,
) -> tuple[np.ndarray, float, float]:
    """``starts`` whitened starting points as overdispersed as the walls allow.

    Draws ``batch`` (default ``4 starts``) standard normals times ``f``, from
    ``f = inflate`` down by ``shrink`` each time (never below ``floor``), and keeps
    the first ``f`` at which at least ``starts`` of them hold at most ``max_share``
    of ``p`` on one cell -- the screen of :func:`starting_points`. Inflating every
    coordinate inflates every tilt, so in thousands of dimensions a large ``f``
    puts some cell over the screen in every draw; this finds the largest ``f`` that
    does not. At ``f = floor`` it falls back to :func:`starting_points`. A
    ``floor`` below 1 starts closer to the whitening's mean than its own draws:
    for a whitening far wider than the posterior (a Laplace approximation),
    whose draws fail the screen. Returns the starts, the ``f`` used and the share
    of that batch that passed.
    """
    batch = batch or 4 * starts
    mean_t = torch.as_tensor(mean, dtype=DTYPE, device=device)
    A_t = torch.as_tensor(A, dtype=DTYPE, device=device)
    f = max(floor, inflate)
    while f > floor:
        x = f * rng.standard_normal((batch, mean.size))
        lam = mean_t + torch.as_tensor(x, dtype=DTYPE, device=device) @ A_t.T
        share, _ = largest_share(target, lam)
        ok = ((share <= max_share) & target.feasible(lam)).cpu().numpy()
        if ok.sum() >= starts:
            return x[ok][:starts], f, float(ok.mean())
        f = max(floor, f * shrink)
    x = starting_points(target, mean, A, starts, rng, max_share, device, inflate=floor)
    return x, floor, float("nan")


def whitening_matrix(q: StructuredGaussian) -> np.ndarray:
    """``A = G'^{-1}`` as a dense ``(m, m)`` array, so that ``mean + A eps`` is a
    draw from the Gaussian part of ``q`` for ``eps ~ N(0, I)``."""
    return q.gaussian_part(np.eye(q.size))


def stan_windows(warmup: int, init: int = 75, term: int = 50, base: int = 25
                 ) -> list[tuple[int, int]]:
    """Stan's metric-adaptation windows for ``warmup`` iterations: ``(start,
    end)`` iteration ranges (0-based, end exclusive), after an initial buffer
    of ``init`` step-size-only iterations and before a final one of ``term``;
    ``base, 2 base, 4 base, ...``, the last stretched to the final buffer. Too
    short a warmup for the buffers gives no windows."""
    windows, start, size = [], init, base
    end_of_windows = warmup - term
    while start + size <= end_of_windows:
        end = start + size
        if end + 2 * size > end_of_windows:
            end = end_of_windows
        windows.append((start, end))
        start, size = end, 2 * size
    return windows


def nested_rhat(values: np.ndarray, superchains: int) -> np.ndarray:
    """Nested ``R-hat`` (Margossian et al. 2024, definition 2.2).

    ``values`` is ``(draws, chains)`` or ``(draws, chains, q)`` for ``q``
    quantities, chain ``c`` in superchain ``c // M``, ``M = chains /
    superchains``. The between-superchain variance ``B`` of the superchain
    means, over the within-superchain variance ``W`` (the variance between
    its chains' means plus, with more than one draw, the mean variance within
    them), as ``sqrt(1 + B / W)``. Returns ``(q,)`` (a 0-d array for 2-d input).
    """
    v = np.asarray(values, dtype=float)
    squeeze = v.ndim == 2
    if squeeze:
        v = v[:, :, None]
    n, chains, q = v.shape
    k = superchains
    if chains % k:
        raise ValueError(f"{chains} chains do not split into {k} superchains")
    m = chains // k
    v = v.reshape(n, k, m, q)                       # draw, superchain, chain, quantity
    chain_mean = v.mean(0)                          # (k, m, q)
    super_mean = chain_mean.mean(1)                 # (k, q)
    b = super_mean.var(0, ddof=1)
    within = np.zeros((k, q))
    if m > 1:
        within += chain_mean.var(1, ddof=1)
    if n > 1:
        within += v.var(0, ddof=1).mean(1)
    w = within.mean(0)
    out = np.sqrt(1.0 + b / w)
    return out[0] if squeeze else out


def nested_tau(superchains: int, chains_per: int, alpha: float = 0.05, tests: int = 1) -> float:
    """The tolerance ``tau`` at which the ``N = 1`` threshold
    ``sqrt(1 + 1/M + tau)`` is exceeded with probability ``alpha / tests`` by
    chains already at stationarity.

    Then every draw is independent, so ``M B / W`` follows an ``F`` distribution
    on ``K - 1`` and ``K (M - 1)`` degrees of freedom, and ``tau = (F_q - 1) / M``
    for ``F_q`` its ``1 - alpha / tests`` quantile: the smallest nonstationary
    variance (relative to the posterior's) that ``K`` superchains can tell from
    noise at that rate. Margossian et al. set ``tau`` from the error they
    tolerate (``1e-4`` for an ESS of 2,000 with 2,048 chains); with fewer
    chains, a ``tau`` below this level fails stationary chains often."""
    from scipy.stats import f

    k, m = superchains, chains_per
    quantile = float(f.ppf(1.0 - alpha / tests, k - 1, k * (m - 1)))
    return (quantile - 1.0) / m


def nested_threshold(chains_per: int, tau: float) -> float:
    """``sqrt(1 + 1/M + tau)``, the ``N = 1`` threshold (Margossian et al., eq. 29)."""
    return math.sqrt(1.0 + 1.0 / chains_per + tau)


@dataclass
class HMCSettings:
    """Tuning of :class:`HMC`.

    Attributes
    ----------
    step_size:
        Initial leapfrog step, in whitened units. About ``m^{-1/4}`` suits a
        standard normal in ``m`` dimensions; warmup adapts it.
    trajectory:
        Nominal integration time; each iteration draws one uniformly from
        ``[0.5, 1.5]`` times this.
    target_accept:
        Mean acceptance probability the warmup adapts towards.
    max_leapfrog:
        Cap on leapfrog steps per iteration, so a collapsing step size cannot
        stall a job.
    divergence:
        Energy error, in nats, above which a transition counts as divergent.
    locate:
        For each divergent transition, report this many whitened coordinates
        with the largest ``|gradient|`` at the trajectory's end -- where the
        potential is steepest, so where the integrator lost the posterior
        (0: none).
    retry_factor:
        Delayed rejection: a divergent transition is retried with step
        ``eps / retry_factor`` (see the module docstring); 0 or 1 turns it off.
    """

    step_size: float = 0.1
    trajectory: float = 3.0
    target_accept: float = 0.8
    max_leapfrog: int = 1000
    divergence: float = 1000.0
    locate: int = 0
    retry_factor: float = 0.0


class HMC:
    """Batched HMC over ``chains`` chains in whitened coordinates.

    ``potential(x)`` is ``n f(mean + A x)``; ``step()`` advances every chain by
    one iteration and returns its statistics. The whole state, random stream
    included, is in :meth:`state_dict`, so a job can checkpoint and resume.
    """

    def __init__(
        self,
        target: _DualTarget,
        n: int,
        mean: np.ndarray,
        A: np.ndarray,
        x0: np.ndarray,
        settings: HMCSettings,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.target = target
        self.n = n
        self.device = device
        self.settings = settings
        self.mean = torch.as_tensor(mean, dtype=DTYPE, device=device)
        self.A = torch.as_tensor(A, dtype=DTYPE, device=device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.x = torch.as_tensor(x0, dtype=DTYPE, device=device).clone()
        self.U, self.grad = self.potential_and_gradient(self.x)
        self.step_size = settings.step_size
        self.iteration = 0
        # Dual averaging state (Hoffman & Gelman 2014, algorithm 5).
        self.da_mu = math.log(10 * settings.step_size)
        self.da_hbar = 0.0
        self.da_log_bar = 0.0
        self.da_count = 0
        # The learned diagonal: lambda = mean + A x with A = A0 diag(scale).
        m = self.x.shape[1]
        self.scale = torch.ones(m, dtype=DTYPE, device=device)
        self.reset_accumulator()

    @property
    def chains(self) -> int:
        return self.x.shape[0]

    def lam(self, x: torch.Tensor) -> torch.Tensor:
        """``(chains, m)`` multipliers for ``(chains, m)`` whitened points."""
        return self.mean + x @ self.A.T

    def potential_and_gradient(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``n f(lambda(x))`` per chain and its gradient in ``x``."""
        x = x.detach().requires_grad_(True)
        U = self.n * self.target(self.lam(x))
        (grad,) = torch.autograd.grad(U.sum(), x)
        return U.detach(), grad

    def _uniform(self, *shape) -> torch.Tensor:
        return torch.rand(*shape, generator=self.generator, dtype=DTYPE, device=self.device)

    def _trajectory(self, x: torch.Tensor, p: torch.Tensor, grad: torch.Tensor, eps: float,
                    n_leap: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``n_leap`` leapfrog steps of size ``eps`` from ``(x, p)``, whose
        potential gradient is ``grad``: the end ``x``, ``p``, ``U`` and gradient."""
        p = p - 0.5 * eps * grad
        U = None
        for i in range(n_leap):
            x = x + eps * p
            U, grad = self.potential_and_gradient(x)
            p = p - (eps if i < n_leap - 1 else 0.5 * eps) * grad
        return x, p, U, grad

    @staticmethod
    def _energy_error(H1: torch.Tensor, H0: torch.Tensor) -> torch.Tensor:
        """``H1 - H0``, infinite where it is not finite."""
        error = H1 - H0
        return torch.where(torch.isfinite(error), error, torch.full_like(error, math.inf))

    def step(self, adapt: bool) -> dict[str, np.ndarray | float | int]:
        """One HMC iteration for every chain; adapts the step size if ``adapt``.

        Returns per-chain ``accept_prob`` (of either stage, with delayed
        rejection), ``accepted``, ``divergent`` (the first stage),
        ``energy`` (the Hamiltonian at the retained state, as in Stan's
        ``energy__``) and ``log_pi`` (``-n f`` there), plus the shared
        ``step_size`` and ``n_leapfrog`` used. With delayed rejection, also
        per chain ``retried`` and ``accepted2`` (the retry was accepted), and
        ``extra_leapfrog``, the steps the retries and their ghosts took.
        """
        s = self.settings
        eps = self.step_size
        length = s.trajectory * (0.5 + float(self._uniform(1)))
        n_leap = int(min(max(1, math.ceil(length / eps)), s.max_leapfrog))

        p0 = torch.randn(self.x.shape, generator=self.generator, dtype=DTYPE, device=self.device)
        H0 = self.U + 0.5 * (p0 * p0).sum(1)
        x, p, U, grad = self._trajectory(self.x, p0, self.grad, eps, n_leap)
        H1 = U + 0.5 * (p * p).sum(1)

        error = self._energy_error(H1, H0)
        alpha1 = torch.exp(torch.clamp(-error, max=0.0))
        accepted = self._uniform(self.chains) < alpha1
        divergent = error > s.divergence
        accept_prob = alpha1

        new_x, new_grad = torch.where(accepted[:, None], x, self.x), torch.where(accepted[:, None], grad, self.grad)
        new_U, energy = torch.where(accepted, U, self.U), torch.where(accepted, H1, H0)

        retried = torch.zeros_like(accepted)
        accepted2 = torch.zeros_like(accepted)
        extra = 0
        if s.retry_factor > 1 and bool((divergent & ~accepted).any()):
            retried = divergent & ~accepted
            r = torch.nonzero(retried).ravel()
            eps2 = eps / s.retry_factor
            x2, p2, U2, g2 = self._trajectory(self.x[r], p0[r], self.grad[r], eps2, n_leap)
            H2 = U2 + 0.5 * (p2 * p2).sum(1)
            # The ghost: the first stage from the retry's reversal.
            _, pg, Ug, _ = self._trajectory(x2, -p2, g2, eps, n_leap)
            Hg = Ug + 0.5 * (pg * pg).sum(1)
            error_g = self._energy_error(Hg, H2)
            alpha1_g = torch.exp(torch.clamp(-error_g, max=0.0))
            ghost_retries = error_g > s.divergence
            log_ratio = (-self._energy_error(H2, H0[r]) + torch.log1p(-alpha1_g)
                         - torch.log1p(-alpha1[r]))
            log_ratio = torch.where(ghost_retries & torch.isfinite(log_ratio), log_ratio,
                                    torch.full_like(log_ratio, -math.inf))
            alpha2 = torch.exp(torch.clamp(log_ratio, max=0.0))
            take = self._uniform(r.numel()) < alpha2
            accepted2[r] = take
            t = r[take]
            new_x[t], new_grad[t] = x2[take], g2[take]
            new_U[t], energy[t] = U2[take], H2[take]
            accept_prob = alpha1.clone()
            accept_prob[r] = alpha1[r] + (1 - alpha1[r]) * alpha2
            extra = 2 * n_leap

        self.x, self.grad, self.U = new_x, new_grad, new_U
        accepted = accepted | accepted2
        self.iteration += 1
        located = {}
        if s.locate and bool(divergent.any()):
            g = torch.nan_to_num(grad[divergent].abs(), nan=math.inf, posinf=math.inf)
            top = torch.topk(g, min(s.locate, g.shape[1]), dim=1)
            located = {"div_chain": torch.nonzero(divergent).ravel().cpu().numpy(),
                       "div_coords": top.indices.cpu().numpy(),
                       "div_grad": top.values.cpu().numpy()}

        mean_accept = float(accept_prob.mean())
        if adapt:
            self._adapt(mean_accept)
        return {
            "accept_prob": accept_prob.cpu().numpy(),
            "accepted": accepted.cpu().numpy(),
            "divergent": divergent.cpu().numpy(),
            "energy": energy.cpu().numpy(),
            "log_pi": (-self.U).cpu().numpy(),
            "step_size": eps,
            "n_leapfrog": n_leap,
            "retried": retried.cpu().numpy(),
            "accepted2": accepted2.cpu().numpy(),
            "extra_leapfrog": extra,
            **located,
        }

    def _adapt(self, accept: float) -> None:
        self.da_count += 1
        m = self.da_count
        w = 1.0 / (m + DA_T0)
        self.da_hbar = (1 - w) * self.da_hbar + w * (self.settings.target_accept - accept)
        log_eps = self.da_mu - math.sqrt(m) / DA_GAMMA * self.da_hbar
        weight = m ** -DA_KAPPA
        self.da_log_bar = weight * log_eps + (1 - weight) * self.da_log_bar
        self.step_size = math.exp(log_eps)

    def end_warmup(self) -> None:
        """Fix the step size at its dual-averaged value."""
        if self.da_count:
            self.step_size = math.exp(self.da_log_bar)

    def restart_step_size(self) -> None:
        """Restart dual averaging from the current step size, as Stan does
        after each metric update."""
        self.da_mu = math.log(10 * self.step_size)
        self.da_hbar = 0.0
        self.da_log_bar = 0.0
        self.da_count = 0

    # -- the learned diagonal ----------------------------------------------

    def set_scale(self, scale) -> None:
        """Make ``lambda = mean + A0 diag(scale) x``, keeping every chain where it is."""
        new = torch.as_tensor(np.asarray(scale), dtype=DTYPE, device=self.device)
        ratio = new / self.scale
        self.A = self.A * ratio[None, :]
        self.x = self.x / ratio
        self.scale = new
        self.U, self.grad = self.potential_and_gradient(self.x)

    def reset_accumulator(self) -> None:
        m = self.x.shape[1]
        self.acc_draws = 0
        self.acc_sum = torch.zeros(m, dtype=DTYPE, device=self.device)
        self.acc_sq = torch.zeros(m, dtype=DTYPE, device=self.device)

    def accumulate(self) -> None:
        """Add the current state of every chain, in the whitening's own
        coordinates ``u = scale * x``, to the window's sums."""
        u = self.x * self.scale
        self.acc_draws += u.shape[0]
        self.acc_sum += u.sum(0)
        self.acc_sq += (u * u).sum(0)

    def adapted_scale(self, shrink: float = 5.0) -> np.ndarray:
        """Each coordinate's sd over the window, pooled over chains and
        iterations, its variance shrunk towards 1 as Stan shrinks towards its
        default: ``(n var + shrink) / (n + shrink)`` for ``n`` the window's
        iterations."""
        draws = self.acc_draws
        mean = self.acc_sum / draws
        var = torch.clamp(self.acc_sq / draws - mean * mean, min=0.0)
        n = draws / self.chains
        var = (n * var + shrink) / (n + shrink)
        return torch.sqrt(torch.clamp(var, min=1e-8)).cpu().numpy()

    def adapt_scale(self, shrink: float = 5.0) -> np.ndarray:
        """End a metric window: set the scale from it, empty the sums and
        restart the step size's adaptation. Returns the new scale."""
        scale = self.adapted_scale(shrink)
        self.set_scale(scale)
        self.reset_accumulator()
        self.restart_step_size()
        return scale

    def largest_cell(self) -> tuple[np.ndarray, np.ndarray]:
        """Per chain, the largest cell's share of ``p(lambda)`` and its flat
        ``(zone, unit)`` index into the ``(n_zones, n_units)`` weight matrix."""
        share, index = largest_share(self.target, self.lam(self.x))
        return share.cpu().numpy(), index.cpu().numpy()

    def state_dict(self) -> dict:
        return {
            "x": self.x.cpu().numpy(),
            "step_size": self.step_size,
            "iteration": self.iteration,
            "da": np.array([self.da_mu, self.da_hbar, self.da_log_bar, self.da_count]),
            "generator": self.generator.get_state().cpu().numpy(),
            "scale": self.scale.cpu().numpy(),
            "acc": np.array([self.acc_draws], dtype=float),
            "acc_sum": self.acc_sum.cpu().numpy(),
            "acc_sq": self.acc_sq.cpu().numpy(),
        }

    def load_state_dict(self, state: dict) -> None:
        if "scale" in state:
            self.set_scale(state["scale"])
            self.acc_draws = int(state["acc"][0])
            self.acc_sum = torch.as_tensor(state["acc_sum"], dtype=DTYPE, device=self.device)
            self.acc_sq = torch.as_tensor(state["acc_sq"], dtype=DTYPE, device=self.device)
        self.x = torch.as_tensor(state["x"], dtype=DTYPE, device=self.device)
        self.U, self.grad = self.potential_and_gradient(self.x)
        self.step_size = float(state["step_size"])
        self.iteration = int(state["iteration"])
        self.da_mu, self.da_hbar, self.da_log_bar, count = (float(v) for v in state["da"])
        self.da_count = int(count)
        self.generator.set_state(torch.as_tensor(state["generator"], dtype=torch.uint8))
