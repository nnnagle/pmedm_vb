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
chain. The mass matrix is not adapted: the whitening plays that part.

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
) -> np.ndarray:
    """``(chains, m)`` whitened starting points: standard normal draws, keeping
    only those whose largest cell holds at most ``max_share`` of ``p``.

    A draw from the whitening Gaussian can land on a wall -- as VB draws do --
    and a chain started there diverges on every trajectory at a step size that
    suits the bulk, so it never moves. Screening the start does not stop a chain
    reaching such a region by itself; it only keeps warmup out of one.
    """
    keep = []
    mean_t = torch.as_tensor(mean, dtype=DTYPE, device=device)
    A_t = torch.as_tensor(A, dtype=DTYPE, device=device)
    for _ in range(tries):
        x = rng.standard_normal((chains, mean.size))
        lam = mean_t + torch.as_tensor(x, dtype=DTYPE, device=device) @ A_t.T
        share, _ = largest_share(target, lam)
        keep.extend(x[(share <= max_share).cpu().numpy()])
        if len(keep) >= chains:
            return np.asarray(keep[:chains])
    raise RuntimeError(
        f"only {len(keep)} of {chains * tries} starting draws had a largest cell under "
        f"{max_share:.1%} of p; the whitening Gaussian is far from the posterior"
    )


def whitening_matrix(q: StructuredGaussian) -> np.ndarray:
    """``A = G'^{-1}`` as a dense ``(m, m)`` array, so that ``mean + A eps`` is a
    draw from the Gaussian part of ``q`` for ``eps ~ N(0, I)``."""
    return q.gaussian_part(np.eye(q.size))


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
        }

    def load_state_dict(self, state: dict) -> None:
        self.x = torch.as_tensor(state["x"], dtype=DTYPE, device=self.device)
        self.U, self.grad = self.potential_and_gradient(self.x)
        self.step_size = float(state["step_size"])
        self.iteration = int(state["iteration"])
        self.da_mu, self.da_hbar, self.da_log_bar, count = (float(v) for v in state["da"])
        self.da_count = int(count)
        self.generator.set_state(torch.as_tensor(state["generator"], dtype=torch.uint8))
