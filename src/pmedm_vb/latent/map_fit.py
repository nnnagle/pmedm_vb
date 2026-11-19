"""MAP for the zone deviations ``U`` and the choice of ``τ`` (step 3, piece B3 of
``PLAN_latent.md``; ``PLAN_latent_step3.md`` version 2, §4.3).

With ``θ_A`` fixed at its records fit and ``u_b ~ N(0, τ² I)`` iid, the
**exact** objective (negative log posterior over ``U``, ``Z × K``) is

    F(U) = -L(θ_A, U) + ‖U‖² / (2τ²),
    -L = ½ (Y - m(U))' S(U)⁻¹ (Y - m(U)) + ½ log det S(U),

``L`` the exact collapsed log-likelihood of step 2's engine (no constant;
:class:`~pmedm_vb.latent.collapsed.CollapsedLikelihood`, here through
:class:`~pmedm_vb.latent.region_engine.RegionLikelihood`). The **fixed-S**
objective (§4.3a) holds ``S`` at its value at ``U = 0``:

    F_S(U) = ½ (Y - m(U))' S(0)⁻¹ (Y - m(U)) + ½ log det S(0) + ‖U‖² / (2τ²),

which equals ``F`` at ``U = 0`` and removes the non-convexity that comes
from ``S`` moving with ``U`` (source 2 of §4.3). Its gradient (derived here,
the first term of the engine's gradient) is
``∂F_S/∂u_b = -N_b Cov_{p_b}(h, a_b) + u_b/τ²`` with ``a_bi = t_i' D_b' S(0)⁻¹ (Y - m)``.
:class:`ExactObjective` and :class:`FixedSObjective` give value, gradient and
the Gauss–Newton Hessian (``J' S⁻¹ J + τ⁻² I``, ``J = ∂m/∂U``; the mean-part
information of step 2, with ``S(U)`` or ``S(0)``).

**Optimizers** (:func:`run_fit`; SciPy's ``L-BFGS-B`` without bounds):

* ``m10``, ``m20``: plain L-BFGS with memory 10 or 20;
* ``gn``: L-BFGS preconditioned by the Gauss–Newton Hessian, refreshed every
  ``refresh`` (30) iterations: L-BFGS runs in ``V = R(U - U_k)``,
  ``R'R`` the Gauss–Newton Hessian at ``U_k``, for at most ``refresh``
  iterations, then ``U_k`` and ``R`` are updated; it stops when an inner run
  stops for any reason other than its iteration limit;
* every run stops when the largest gradient component is at most ``gtol``,
  the relative change of the objective is below ``ftol``, or the iteration
  cap is reached.

**Starts** (:func:`fit_starts`, §4.3b). A start is ``(name, objective,
optimizer, origin)``, origin ``zero`` or ``previous`` (the previous ``τ``'s
MAP); a ``polish`` start fits the fixed-S objective with ``m10`` from 0 and
then the exact objective with ``m10`` from there. The MAP is the start with
the lowest exact objective; the spread (largest minus smallest) of the starts'
final exact objectives is reported. :data:`START_SETS`:

* ``fixedS``: ``polish`` and ``m10`` on the exact objective from 0;
* ``all4``: the four exact fits of §4.3a: ``m10``, ``m20``, ``gn`` from 0 and
  ``m10`` from the previous ``τ``'s MAP (none for the first ``τ``);
* ``plain2``: ``m10`` from 0 and from the previous ``τ``'s MAP.

:func:`laplace_evidence` (§4.3c): for each ``τ``, Laplace's approximation of
the marginal likelihood at the MAP with the Gauss–Newton Hessian,

    log p(Y | τ) ≈ L(Û) - ‖Û‖²/(2τ²) - ZK log τ - ½ log det(I(Û) + τ⁻² I)

(the ``(2π)^{ZK/2}`` of the Gaussian integral cancels the prior's; the
likelihood's own constant is common to every ``τ``); the ``τ`` with the
largest value is chosen (:func:`choose_tau`). Checks: the largest gradient
component at the optimum, and ``F(Û) ≤ F(0)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize

from pmedm_vb.latent.collapsed import CollapsedLikelihood, Evaluation
from pmedm_vb.latent.region_engine import information, zone_covariances
from pmedm_vb.latent.information import puma_information

OBJECTIVES = ("exact", "fixedS")
OPTIMIZERS = ("m10", "m20", "gn")

#: §4.3b start sets: (name, objective, optimizer, origin); ``polish`` is the
#: fixed-S fit polished on the exact objective.
START_SETS = {
    "fixedS": (("fixedS_polish", "polish", "m10", "zero"), ("exact_m10", "exact", "m10", "zero")),
    "all4": (("exact_m10", "exact", "m10", "zero"), ("exact_m20", "exact", "m20", "zero"),
             ("exact_gn", "exact", "gn", "zero"), ("exact_warm", "exact", "m10", "previous")),
    "plain2": (("exact_m10", "exact", "m10", "zero"), ("exact_warm", "exact", "m10", "previous")),
}


# --------------------------------------------------------------------------
# Objectives
# --------------------------------------------------------------------------


class ExactObjective:
    """``F`` (module notes); counts engine evaluations."""

    kind = "exact"

    def __init__(self, engine: CollapsedLikelihood, theta: np.ndarray, tau: float):
        self.engine, self.theta, self.tau = engine, np.asarray(theta, float), float(tau)
        self.shape = (engine.area.B, engine.K)
        self.evaluations = 0

    def value_grad(self, U: np.ndarray) -> tuple[float, np.ndarray]:
        ev = self.engine.evaluate(self.theta, U)
        self.evaluations += 1
        prec = 1.0 / self.tau ** 2
        return -ev.loglik + 0.5 * prec * float(np.sum(U * U)), -ev.grad_U + prec * U

    def value(self, U: np.ndarray) -> float:
        ev = self.engine.evaluate(self.theta, U, gradient=False)
        self.evaluations += 1
        return -ev.loglik + 0.5 * float(np.sum(U * U)) / self.tau ** 2

    def gn_hessian(self, U: np.ndarray) -> np.ndarray:
        ev = self.engine.evaluate(self.theta, U, gradient=False)
        self.evaluations += 1
        info = information(self.engine, ev)
        info[np.diag_indices_from(info)] += 1.0 / self.tau ** 2
        return info


class FixedSObjective:
    """``F_S`` (module notes): ``S`` held at ``S(0)``."""

    kind = "fixedS"

    def __init__(self, engine: CollapsedLikelihood, theta: np.ndarray, tau: float,
                 ev0: Evaluation | None = None):
        self.engine, self.theta, self.tau = engine, np.asarray(theta, float), float(tau)
        self.shape = (engine.area.B, engine.K)
        self.ev0 = engine.evaluate(self.theta, None, gradient=False) if ev0 is None else ev0
        self.evaluations = 0

    def solve(self, e: np.ndarray) -> list[np.ndarray]:
        """``S(0)⁻¹ e`` per tract (local layout), by the engine's Woodbury pieces."""
        ev0, tracts = self.ev0, self.engine.tracts
        out, Lz = [], 0.0
        for tract, factor in zip(tracts, ev0.factors):
            z = cho_solve(factor, e[tract["rows"]], check_finite=False)
            out.append(z)
            if ev0.K_factor is not None:
                Lz = Lz + tract["Lr"].T @ z
        if ev0.K_factor is not None:
            c = cho_solve(ev0.K_factor, Lz, check_finite=False)
            out = [z - Xt @ c for z, Xt in zip(out, ev0.X)]
        return out

    def _pieces(self, U):
        eng = self.engine
        p = eng.probabilities(self.theta, U)
        mu = eng.pairs.TuT @ eng.pairs.aggregate(p)
        m = eng.stack(mu)
        e = eng.area.Y - m
        r = self.solve(e)
        quad = float(sum(float(e[t["rows"]] @ z) for t, z in zip(eng.tracts, r)))
        prior = 0.5 * float(np.sum(U * U)) / self.tau ** 2
        return p, mu, r, 0.5 * quad + 0.5 * self.ev0.logdet + prior

    def value(self, U: np.ndarray) -> float:
        self.evaluations += 1
        return self._pieces(U)[3]

    def value_grad(self, U: np.ndarray) -> tuple[float, np.ndarray]:
        self.evaluations += 1
        eng = self.engine
        p, mu, r, value = self._pieces(U)
        J_T, J_B = eng.area.J_T, eng.area.J_B
        w0 = np.zeros((eng.pairs.J0, eng.area.B))
        for tract, z in zip(eng.tracts, r):
            for j, b in enumerate(tract["members"]):
                w0[eng.ts, b] += z[:J_T]
                w0[eng.bs, b] += z[J_T + j * J_B: J_T + (j + 1) * J_B]
        phi = eng.pairs.expand(eng.pairs.Tu @ w0)
        grad_L = eng.N[:, None] * eng.covariance_with_h(p, phi)
        return value, -grad_L + U / self.tau ** 2

    def gn_hessian(self, U: np.ndarray) -> np.ndarray:
        ev = self.engine.evaluate(self.theta, U, gradient=False)
        self.evaluations += 1
        at = replace(self.ev0, p=ev.p, mu=ev.mu)   # p_b(U) with S(0)'s factors
        info = puma_information(self.engine, at, zone_covariances(self.engine, at))
        info[np.diag_indices_from(info)] += 1.0 / self.tau ** 2
        return info


# --------------------------------------------------------------------------
# Optimizers
# --------------------------------------------------------------------------


@dataclass
class MapFit:
    tau: float
    U: np.ndarray
    objective: float            # exact F(Û)
    objective_at_zero: float    # F(0)
    loglik: float               # L(Û)
    grad_max: float             # largest |∂F/∂U| (exact) at Û
    iterations: int
    evaluations: int
    converged: bool
    message: str
    seconds: float
    start: str = "exact_m10"
    fitted_objective: str = "exact"
    optimizer: str = "m10"
    origin: str = "zero"
    fixed_s_objective: float = float("nan")    # F_S at Û (fixed-S fits)
    fixed_s_grad_max: float = float("nan")     # largest |∂F_S/∂U| at Û (fixed-S fits)
    eval_seconds: list = field(default_factory=list)
    log_evidence: float = float("nan")
    logdet_hessian: float = float("nan")
    evidence_seconds: float = float("nan")

    def info(self) -> dict:
        return {"start": self.start, "fitted_objective": self.fitted_objective,
                "optimizer": self.optimizer, "origin": self.origin, "tau": self.tau,
                "objective": self.objective, "objective_at_zero": self.objective_at_zero,
                "objective_not_above_zero": bool(self.objective <= self.objective_at_zero),
                "fixed_s_objective": self.fixed_s_objective,
                "fixed_s_grad_max": self.fixed_s_grad_max,
                "loglik": self.loglik, "U_norm": float(np.linalg.norm(self.U)),
                "U_rms": float(np.sqrt(np.mean(self.U ** 2))),
                "grad_max": self.grad_max, "iterations": self.iterations,
                "evaluations": self.evaluations, "converged": self.converged,
                "message": self.message, "seconds": self.seconds,
                "seconds_per_evaluation": float(np.mean(self.eval_seconds))
                if self.eval_seconds else float("nan"),
                "log_evidence": self.log_evidence, "logdet_hessian": self.logdet_hessian,
                "evidence_seconds": self.evidence_seconds}


def _lbfgs(objective, x0: np.ndarray, memory: int, gtol: float, ftol: float, max_iter: int,
           times: list) -> tuple[np.ndarray, int, bool, str]:
    shape = objective.shape

    def fun(x):
        t0 = time.perf_counter()
        value, grad = objective.value_grad(x.reshape(shape))
        times.append(time.perf_counter() - t0)
        return value, grad.ravel()

    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "maxfun": 10 * max_iter, "gtol": gtol,
                            "ftol": ftol, "maxcor": memory})
    return res.x, int(res.nit), bool(res.success), str(res.message)


def _gn_lbfgs(objective, x0: np.ndarray, gtol: float, ftol: float, max_iter: int, times: list,
              refresh: int = 30, memory: int = 20) -> tuple[np.ndarray, int, bool, str]:
    shape, x, total = objective.shape, np.asarray(x0, float).copy(), 0
    message, success = "max_iter", False
    while total < max_iter:
        R = cholesky(objective.gn_hessian(x.reshape(shape)), lower=False)
        base = x.copy()

        def fun(v, base=base, R=R):
            t0 = time.perf_counter()
            u = base + solve_triangular(R, v, lower=False)
            value, grad = objective.value_grad(u.reshape(shape))
            times.append(time.perf_counter() - t0)
            return value, solve_triangular(R, grad.ravel(), trans="T", lower=False)

        res = minimize(fun, np.zeros(x.size), jac=True, method="L-BFGS-B",
                       options={"maxiter": min(refresh, max_iter - total),
                                "maxfun": 10 * refresh, "gtol": gtol, "ftol": ftol,
                                "maxcor": memory})
        x = base + solve_triangular(R, res.x, lower=False)
        total += int(res.nit)
        if res.status != 1:   # 1: an iteration or evaluation limit; anything else ends the fit
            success, message = bool(res.success), str(res.message)
            break
    return x, total, success, message


def run_fit(engine: CollapsedLikelihood, theta: np.ndarray, tau: float, objective: str,
            optimizer: str, *, U0: np.ndarray | None = None, ev0: Evaluation | None = None,
            gtol: float = 1e-4, ftol: float = 1e-14, max_iter: int = 20000,
            refresh: int = 30) -> MapFit:
    """One fit (module notes): ``objective`` ``exact`` or ``fixedS``,
    ``optimizer`` ``m10``, ``m20`` or ``gn``, from ``U0`` (default 0). The
    returned fit carries the exact objective and gradient at its solution."""
    if objective not in OBJECTIVES or optimizer not in OPTIMIZERS:
        raise ValueError(f"unknown objective {objective!r} or optimizer {optimizer!r}")
    start = time.perf_counter()
    exact = ExactObjective(engine, theta, tau)
    obj = exact if objective == "exact" else FixedSObjective(engine, theta, tau, ev0)
    shape = exact.shape
    F0 = exact.value(np.zeros(shape))
    x0 = np.zeros(int(np.prod(shape))) if U0 is None else np.asarray(U0, float).ravel()
    times: list[float] = []
    if optimizer == "gn":
        x, nit, success, message = _gn_lbfgs(obj, x0, gtol, ftol, max_iter, times, refresh)
    else:
        memory = 10 if optimizer == "m10" else 20
        x, nit, success, message = _lbfgs(obj, x0, memory, gtol, ftol, max_iter, times)
    U = x.reshape(shape)
    value, grad = exact.value_grad(U)
    loglik = -(value - 0.5 * float(np.sum(U * U)) / tau ** 2)
    fit = MapFit(tau=float(tau), U=U, objective=float(value), objective_at_zero=float(F0),
                 loglik=float(loglik), grad_max=float(np.abs(grad).max()), iterations=nit,
                 evaluations=0, converged=success, message=message, seconds=0.0,
                 fitted_objective=objective, optimizer=optimizer,
                 origin="zero" if U0 is None else "given", eval_seconds=times)
    if objective == "fixedS":
        fs, gs = obj.value_grad(U)
        fit.fixed_s_objective, fit.fixed_s_grad_max = float(fs), float(np.abs(gs).max())
    fit.evaluations = exact.evaluations + (obj.evaluations if obj is not exact else 0)
    fit.seconds = time.perf_counter() - start
    return fit


def fit_map(engine: CollapsedLikelihood, theta: np.ndarray, tau: float, **kwargs) -> MapFit:
    """Plain L-BFGS (memory 10) on the exact objective, from ``U0`` or 0."""
    return run_fit(engine, theta, tau, "exact", "m10", **kwargs)


def fit_start(engine: CollapsedLikelihood, theta: np.ndarray, tau: float, start: tuple,
              previous: np.ndarray | None, **kwargs) -> MapFit | None:
    """One start of a §4.3b set (module notes); ``None`` for a ``previous``
    start without a previous ``τ``."""
    name, objective, optimizer, origin = start
    if origin == "previous" and previous is None:
        return None
    U0 = previous if origin == "previous" else None
    if objective == "polish":
        t0 = time.perf_counter()
        first = run_fit(engine, theta, tau, "fixedS", optimizer, U0=U0, **kwargs)
        fit = run_fit(engine, theta, tau, "exact", "m10", U0=first.U, **kwargs)
        fit.evaluations += first.evaluations
        fit.iterations += first.iterations
        fit.eval_seconds = first.eval_seconds + fit.eval_seconds
        fit.fixed_s_objective = first.fixed_s_objective
        fit.seconds = time.perf_counter() - t0
        fit.fitted_objective, fit.optimizer = "fixedS+exact", "m10"
    else:
        fit = run_fit(engine, theta, tau, objective, optimizer, U0=U0, **kwargs)
    fit.start, fit.origin = name, origin
    return fit


def fit_starts(engine: CollapsedLikelihood, theta: np.ndarray, tau: float, starts: str,
               previous: np.ndarray | None, **kwargs) -> tuple[MapFit, list[MapFit]]:
    """Every start of set ``starts``; the MAP is the lowest exact objective."""
    fits = [f for s in START_SETS[starts]
            if (f := fit_start(engine, theta, tau, s, previous, **kwargs)) is not None]
    best = min(fits, key=lambda f: f.objective)
    return best, fits


def laplace_evidence(engine: CollapsedLikelihood, theta: np.ndarray, fit: MapFit) -> MapFit:
    """Fill ``fit.log_evidence`` (module notes)."""
    t0 = time.perf_counter()
    ev = engine.evaluate(theta, fit.U, gradient=False)
    info = information(engine, ev)
    prec = 1.0 / fit.tau ** 2
    info[np.diag_indices_from(info)] += prec
    sign, logdet = np.linalg.slogdet(info)
    if sign <= 0:
        raise np.linalg.LinAlgError("the Gauss-Newton Hessian is not positive definite")
    size = fit.U.size
    fit.logdet_hessian = float(logdet)
    fit.log_evidence = float(ev.loglik - 0.5 * prec * np.sum(fit.U ** 2)
                             - size * np.log(fit.tau) - 0.5 * logdet)
    fit.evidence_seconds = time.perf_counter() - t0
    return fit


def choose_tau(fits: list[MapFit]) -> int:
    """Index of the fit with the largest Laplace evidence."""
    return int(np.argmax([f.log_evidence for f in fits]))
