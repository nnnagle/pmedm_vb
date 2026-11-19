"""The conditional PMEDM: the realized compositions' MAP given the zones'
superpopulations (step 3, piece C1 of ``PLAN_latent.md``;
``PLAN_latent_step3.md`` version 1, §4.4; ``MODEL_latent.qmd`` version 2,
stage 1 step 2).

Jointly over the region's zones ``b``, with ``r_b`` on the simplex over the
atoms,

    min_{r}  P(r) = Σ_b κ_b KL(r_b ‖ p_b) + ½ e'Σ⁻¹e,      e = Σ_b N_b D_b T' r_b - Y,

with ``κ_b = N_b`` (the model's entropy weight; ``latent_n`` uses
``κ_b = n_A N_b / N_A``, the standard PMEDM's total weight ``n_A`` split by
the zones' shares). ``D_b`` reaches zone ``b``'s own rows: the stacked vector
is cell-major with the zone fastest, so zone ``b``'s multipliers are
``λ_b = λ.reshape(C, Z)[:, b]``.

**Dual** (derived here, as in the model with a general ``κ_b``). Minimizing
the Lagrangian over ``e`` and each ``r_b``:

    f(λ) = λ'Y + Σ_b κ_b log Σ_i p_bi exp{-(N_b/κ_b) t_i'λ_b} + ½ λ'Σλ,
    r_bi ∝ p_bi exp{-(N_b/κ_b) t_i'λ_b},

``min P = -min f`` (strong duality; ``f`` is strictly convex since ``Σ`` is
positive definite). Gradient ``Y - Σ_b N_b D_b T'r_b + Σλ = Σλ - e``, Hessian
``Σ + Σ_b (N_b²/κ_b) D_b Cov_{r_b}(t) D_b'``.

**Solver** (:func:`solve_conditional`). A dedicated dual Newton method, the
same method as :func:`~pmedm_vb.solvers.map_dual.solve_map` (Newton with an
Armijo backtracking line search, stopped on the Newton decrement
``½ g'H⁻¹g``), on the dense Hessian: a region has at most ``Z C`` rows (a few
hundred to about a thousand), so the Hessian is factorized by Cholesky. The
existing solver is not used because it leaves the zone shares free (parent
plan, finding 3: it solves this problem plus ``N KL(π ‖ π⁰)`` over the shares
``π``), whereas here ``N_b`` is known and the shares are fixed.

**Checks** (in :class:`ConditionalResult`): the primal ``P`` at ``r(λ̂)``, the
dual value ``-f(λ̂)``, the duality gap ``P + f`` (``≥ 0``, ``0`` at the
optimum), the stationarity (KKT) residual ``max |e - Σλ|`` (alone and relative
to ``max |e|``), and the largest deviation of ``Σ_i r_bi`` from 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve
from scipy.special import logsumexp

from pmedm_vb.assemble.sigma import Sigma

ARMIJO = 0.25
BACKTRACK = 0.5
MAX_HALVINGS = 50


@dataclass
class ConditionalResult:
    lam: np.ndarray            # (C, Z)
    r: np.ndarray              # (Z, M)
    fitted: np.ndarray         # Σ_b N_b D_b T' r_b, stacked
    objective: float           # f(λ̂), the dual objective (minimized)
    primal: float              # P(r(λ̂))
    gap: float                 # P + f
    kkt_max: float             # max |e - Σλ|
    kkt_relative: float        # kkt_max / max |e|
    simplex_max: float         # max_b |Σ_i r_bi - 1|
    decrement: float
    iterations: int
    converged: bool
    seconds: float
    trace: dict = field(default_factory=dict)

    def info(self) -> dict:
        return {"objective": self.objective, "primal": self.primal, "gap": self.gap,
                "relative_gap": self.gap / max(1.0, abs(self.primal)),
                "kkt_max": self.kkt_max, "kkt_relative": self.kkt_relative,
                "simplex_max": self.simplex_max, "decrement": self.decrement,
                "iterations": self.iterations, "converged": self.converged,
                "seconds": self.seconds}


class ConditionalProblem:
    """The pieces of the dual (module notes)."""

    def __init__(self, T: sp.spmatrix, base: np.ndarray, N: np.ndarray, Y: np.ndarray,
                 sigma: Sigma | np.ndarray, kappa: np.ndarray | None = None):
        self.T = sp.csr_matrix(T, dtype=float)
        self.Tt = self.T.T.tocsr()
        base = np.asarray(base, float)
        self.Z, self.M = base.shape
        self.C = self.T.shape[1]
        if self.T.shape[0] != self.M:
            raise ValueError("T and the base disagree on the atoms")
        with np.errstate(divide="ignore"):
            self.logp = np.log(base)
        self.N = np.asarray(N, float)
        self.kappa = self.N.copy() if kappa is None else np.asarray(kappa, float)
        self.ratio = self.N / self.kappa
        self.Y = np.asarray(Y, float)
        if self.Y.size != self.C * self.Z:
            raise ValueError("Y is not cells x zones")
        self.Sigma = sigma.to_dense() if isinstance(sigma, Sigma) else np.asarray(sigma, float)
        self.rows = [np.arange(self.C) * self.Z + b for b in range(self.Z)]

    def state(self, lam: np.ndarray) -> dict:
        L = lam.reshape(self.C, self.Z)
        s = -(self.T @ L) * self.ratio[None, :]               # (M, Z)
        logits = self.logp.T + s
        logZ = logsumexp(logits, axis=0)
        r = np.exp(logits - logZ[None, :]).T                  # (Z, M)
        fitted = ((self.Tt @ r.T) * self.N[None, :]).ravel()  # (C, Z) -> stacked
        sig_lam = self.Sigma @ lam
        value = float(lam @ self.Y + self.kappa @ logZ + 0.5 * lam @ sig_lam)
        grad = self.Y - fitted + sig_lam
        return {"lam": lam, "r": r, "s": s, "logZ": logZ, "fitted": fitted,
                "sig_lam": sig_lam, "value": value, "grad": grad}

    def hessian(self, st: dict) -> np.ndarray:
        H = self.Sigma.copy()
        for b in range(self.Z):
            r = st["r"][b]
            mu = self.Tt @ r
            second = (self.Tt @ self.T.multiply(r[:, None])).toarray()
            cov = second - np.outer(mu, mu)
            rows = self.rows[b]
            H[np.ix_(rows, rows)] += (self.N[b] ** 2 / self.kappa[b]) * cov
        return H

    def primal(self, st: dict) -> float:
        """``P(r(λ))``: KL from the exponents, the misfit by a solve with Σ."""
        r, s, logZ = st["r"], st["s"], st["logZ"]
        kl = np.array([float(r[b] @ np.where(r[b] > 0, s[:, b], 0.0)) - logZ[b]
                       for b in range(self.Z)])
        e = st["fitted"] - self.Y
        quad = float(e @ cho_solve(cho_factor(self.Sigma, lower=True), e))
        return float(self.kappa @ kl + 0.5 * quad)


def solve_conditional(T: sp.spmatrix, base: np.ndarray, N: np.ndarray, Y: np.ndarray,
                      sigma: Sigma | np.ndarray, *, kappa: np.ndarray | None = None,
                      init: np.ndarray | None = None, tol: float = 1e-9,
                      max_iter: int = 200) -> ConditionalResult:
    """The conditional PMEDM by dual Newton (module notes). ``T`` is atoms ×
    cells, ``base`` zones × atoms (each row ``p_b``, summing to 1), ``N`` the
    zones' units, ``Y`` the stacked targets."""
    start = time.perf_counter()
    prob = ConditionalProblem(T, base, N, Y, sigma, kappa)
    lam = np.zeros(prob.C * prob.Z) if init is None else np.asarray(init, float).ravel()
    st = prob.state(lam)
    trace = {"objective": [], "decrement": [], "step": []}
    converged, decrement, it = False, np.inf, 0
    for it in range(max_iter + 1):
        H = prob.hessian(st)
        direction = -cho_solve(cho_factor(H, lower=True), st["grad"])
        slope = float(st["grad"] @ direction)
        decrement = -0.5 * slope
        trace["objective"].append(st["value"])
        trace["decrement"].append(decrement)
        if decrement <= tol:
            converged = True
            trace["step"].append(0.0)
            break
        if it == max_iter:
            trace["step"].append(0.0)
            break
        step = 1.0
        for _ in range(MAX_HALVINGS):
            cand = prob.state(st["lam"] + step * direction)
            if cand["value"] <= st["value"] + ARMIJO * step * slope:
                break
            step *= BACKTRACK
        else:
            # The objective no longer resolves the step: the decrement is at
            # the floating-point floor of f.
            trace["step"].append(0.0)
            break
        trace["step"].append(step)
        st = cand
    primal = prob.primal(st)
    e = st["fitted"] - prob.Y
    kkt = float(np.abs(e - st["sig_lam"]).max())
    return ConditionalResult(
        lam=st["lam"].reshape(prob.C, prob.Z), r=st["r"], fitted=st["fitted"],
        objective=st["value"], primal=primal, gap=primal + st["value"], kkt_max=kkt,
        kkt_relative=kkt / max(float(np.abs(e).max()), 1e-300),
        simplex_max=float(np.abs(st["r"].sum(axis=1) - 1.0).max()), decrement=decrement,
        iterations=it, converged=converged, seconds=time.perf_counter() - start,
        trace={k: np.asarray(v) for k, v in trace.items()})
