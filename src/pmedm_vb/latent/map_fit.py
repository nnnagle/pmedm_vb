"""MAP for the zone deviations ``U`` and the choice of ``τ`` (step 3, piece B3 of
``PLAN_latent.md``; ``PLAN_latent_step3.md`` version 1, §4.3).

With ``θ_A`` fixed at its records fit and ``u_b ~ N(0, τ² I)`` iid, the
negative log posterior over ``U`` (``Z × K``) is

    F(U) = -L(θ_A, U) + ‖U‖² / (2τ²),

``L`` the exact collapsed log-likelihood of step 2's engine (no constant;
:class:`~pmedm_vb.latent.collapsed.CollapsedLikelihood`, here through
:class:`~pmedm_vb.latent.region_engine.RegionLikelihood`).

* :func:`fit_map`: L-BFGS (SciPy's ``L-BFGS-B`` without bounds) from
  ``U = 0``, with the engine's gradient, in the coordinates ``U`` themselves.
  It stops when the largest gradient component is at most ``gtol``
  (``pgtol``), or the relative change of ``F`` is below ``ftol``, or after
  ``max_iter`` iterations. ``F`` is not convex in ``U`` (the ``log det S`` term
  and the tilt's nonlinearity), so the optimum is a local one: the one plain
  L-BFGS reaches from ``U = 0``. Reported: ``F`` at the optimum and at
  ``U = 0`` (the check ``F(Û) ≤ F(0)``), the largest gradient component at the
  optimum, iterations, evaluations and seconds.
* :func:`laplace_evidence`: ``τ`` is not fitted jointly (parent plan, finding
  2). For each ``τ`` on the grid the marginal likelihood is approximated by
  Laplace's method at the MAP with a Gauss–Newton Hessian, the mean-part
  information ``I(Û)`` of step 2
  (:func:`~pmedm_vb.latent.region_engine.information`) plus ``τ⁻² I``:

      log p(Y | τ) ≈ L(Û) - ‖Û‖²/(2τ²) - ZK log τ - ½ log det(I(Û) + τ⁻² I)

  (the ``(2π)^{ZK/2}`` of the Gaussian integral cancels the prior's; the
  likelihood's own constant is common to every ``τ``). The ``τ`` with the
  largest value is chosen (:func:`choose_tau`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from pmedm_vb.latent.collapsed import CollapsedLikelihood
from pmedm_vb.latent.region_engine import information


@dataclass
class MapFit:
    tau: float
    U: np.ndarray
    objective: float            # F(Û)
    objective_at_zero: float    # F(0)
    loglik: float               # L(Û)
    grad_max: float             # largest |∂F/∂U| at Û
    iterations: int
    evaluations: int
    converged: bool
    message: str
    seconds: float
    eval_seconds: list = field(default_factory=list)
    log_evidence: float = float("nan")
    logdet_hessian: float = float("nan")
    evidence_seconds: float = float("nan")

    def info(self) -> dict:
        return {"tau": self.tau, "objective": self.objective,
                "objective_at_zero": self.objective_at_zero,
                "objective_not_above_zero": bool(self.objective <= self.objective_at_zero),
                "loglik": self.loglik, "U_norm": float(np.linalg.norm(self.U)),
                "U_rms": float(np.sqrt(np.mean(self.U ** 2))),
                "grad_max": self.grad_max, "iterations": self.iterations,
                "evaluations": self.evaluations, "converged": self.converged,
                "message": self.message, "seconds": self.seconds,
                "seconds_per_evaluation": float(np.mean(self.eval_seconds))
                if self.eval_seconds else float("nan"),
                "log_evidence": self.log_evidence, "logdet_hessian": self.logdet_hessian,
                "evidence_seconds": self.evidence_seconds}


def negative_log_posterior(engine: CollapsedLikelihood, theta: np.ndarray, U: np.ndarray,
                           tau: float) -> tuple[float, np.ndarray, float]:
    """``F(U)``, its gradient (``Z × K``) and ``L(θ_A, U)``."""
    ev = engine.evaluate(theta, U)
    prec = 1.0 / tau ** 2
    value = -ev.loglik + 0.5 * prec * float(np.sum(U * U))
    grad = -ev.grad_U + prec * U
    return value, grad, ev.loglik


def fit_map(engine: CollapsedLikelihood, theta: np.ndarray, tau: float, *,
            U0: np.ndarray | None = None, gtol: float = 1e-4, ftol: float = 1e-14,
            max_iter: int = 20000, memory: int = 20) -> MapFit:
    """Û at ``τ`` (module notes); ``U0`` (default 0) is the start."""
    start = time.perf_counter()
    shape = (engine.area.B, engine.K)
    zero = np.zeros(shape)
    F0, _, _ = negative_log_posterior(engine, theta, zero, tau)
    count = {"n": 1}
    times: list[float] = []
    cache: dict = {}

    def fun(x):
        t0 = time.perf_counter()
        value, grad, loglik = negative_log_posterior(engine, theta, x.reshape(shape), tau)
        times.append(time.perf_counter() - t0)
        count["n"] += 1
        cache["u"], cache["value"], cache["grad"], cache["loglik"] = x.copy(), value, grad, loglik
        return value, grad.ravel()

    x0 = zero.ravel() if U0 is None else np.asarray(U0, float).ravel()
    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "maxfun": 10 * max_iter, "gtol": gtol,
                            "ftol": ftol, "maxcor": memory})
    U = res.x.reshape(shape)
    if "u" in cache and np.array_equal(cache["u"], res.x):
        value, grad, loglik = cache["value"], cache["grad"], cache["loglik"]
    else:
        value, grad, loglik = negative_log_posterior(engine, theta, U, tau)
        count["n"] += 1
    grad_max = float(np.abs(grad).max())
    return MapFit(tau=float(tau), U=U, objective=float(value), objective_at_zero=float(F0),
                  loglik=float(loglik), grad_max=grad_max, iterations=int(res.nit),
                  evaluations=int(count["n"]), converged=bool(res.success),
                  message=str(res.message), seconds=time.perf_counter() - start,
                  eval_seconds=times)


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
