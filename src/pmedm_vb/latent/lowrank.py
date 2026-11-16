"""Low-rank zone coefficients shared across areas: θ_b = W u_b.

For unit i and zone b of area (region) R,

    η_ib = h̃_i′β_R + c_b + h̃_i′W u_b,      p_ib ∝ q_ib exp(−η_ib),

the latent-only model (:class:`~pmedm_vb.latent.model.LatentModel` with
``shared="features"``) with each zone's coefficient vector confined to the
r columns of a matrix ``W`` (features x r) **shared by every zone of every
area**. A zone then has r free numbers, a bottleneck on how far noise in its
tables can move it, and ``W`` learns, pooled over all areas, the directions in
feature space along which zones differ. Only the areas' tables enter; the
truth is never used.

**Features.** ``h̃`` is standardised once over all units statewide (design
weights; 0/1 indicators left as they are), not per area as the per-area
models do, so that ``W`` acts on the same scale everywhere.

**Objective.** The sum over areas of each area's objective,
``½ r′(Σ+τ²I)⁻¹ r + α n KL(p‖q)``, plus independent normal priors:
``β_R`` sd ``tau_beta`` (3), ``c_b`` sd ``tau_intercept`` (3),
``u_b`` sd ``tau_u`` (3) and ``W`` sd ``tau_w`` (1).

**Identifiability.** Within each area the zone coefficients satisfy the
weighted sum-to-zero ``Σ_b N_b [c_b, u_b] = 0``, by the same
reparameterisation as the per-area models (``u = M Γ_u``, ``c = M Γ_c``,
``M`` an orthonormal basis of the complement of ``N_b``). ``θ_b = W u_b`` is
unchanged by ``W → W A``, ``u_b → A⁻¹ u_b``: the two ridges fix the scale
(the penalised optimum balances them), leaving rotations of the r-dimensional
space, which change no prediction. Its columns carry no order.

**Fitting.** All parameters jointly (``W`` and every area's ``β``, ``Γ_c``,
``Γ_u``) by L-BFGS on the summed objective with its exact gradient. The
objective is not convex. Two starts:

* ``pca``: ``W`` the top r principal directions of the standardised
  features (statewide, design-weighted), everything else zero;
* ``svd``: from per-area full-rank latent-only fits (``--theta-from``; the
  7-attribute features only): their zone coefficients, moved onto the
  statewide standardisation, stacked over zones and truncated to rank r in
  the features' covariance metric (:func:`svd_start`); ``β`` and the
  intercepts from the same fits.

The lower objective is kept; both are recorded. L-BFGS converges slowly
here (on two smoke regions, tens of thousands of iterations); a fit stopped
at its iteration limit is reported as not converged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.optimize
from scipy.special import logsumexp

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.sigma import Sigma
from pmedm_vb.latent.model import standardise, sum_to_zero_basis
from pmedm_vb.progress import logger
from pmedm_vb.solvers.base import ConstraintOperator


def centring_shift(theta_state: np.ndarray, H_area: np.ndarray, binary: np.ndarray,
                   weight: np.ndarray) -> np.ndarray:
    """What zone intercepts gain when coefficients on the area's
    standardisation move to the statewide one: with the area's centring
    ``m_r`` and spread ``s_r``, ``(h − m_r)/s_r = h̃_g s_g/s_r + (m_g − m_r)/s_r``
    and ``(m_g − m_r)/s_r · θ_r = −mean_r(h̃_g) · θ_g`` (indicators are not
    centred, so they add nothing). ``H_area`` is the area's units on the
    statewide standardisation, ``weight`` their design weights."""
    mean = np.where(binary, 0.0, weight @ H_area / weight.sum())
    return -theta_state @ mean


def statewide_features(F: np.ndarray, binary: np.ndarray, weight: np.ndarray, names: list[str]):
    """Standardise features over all units; drop constant columns."""
    Hs, kept, centre, spread = standardise(F, np.asarray(binary, bool), weight)
    return Hs, [names[j] for j in kept], centre[kept], spread[kept], np.asarray(binary, bool)[kept]


class AreaTerm:
    """One area's data term ``½ r′(Σ+τ²I)⁻¹ r + α n KL(p‖q)`` and its
    gradient in (β, Γ_c, Γ_u, W)."""

    def __init__(self, inputs: PMEDMInputs, sigma: Sigma, H: np.ndarray,
                 zone_weight: np.ndarray, alpha: float) -> None:
        self.inputs, self.sigma = inputs, sigma
        self.op = ConstraintOperator(inputs)
        self.N, self.n = float(inputs.N), float(inputs.n)
        self.Y = inputs.targets()
        self.log_q = np.log(inputs.q)
        self.log_q_total = float(logsumexp(self.log_q))
        self.H = np.asarray(H, dtype=np.float64)
        self.HT = self.H.T.copy()
        self.M = sum_to_zero_basis(zone_weight)
        self.Z, self.U = inputs.q.shape
        self.k = self.H.shape[1]
        self.entropy = float(alpha) * self.n

    def evaluate(self, beta, gc, gu, W, gradient: bool = True):
        HW = self.H @ W                                 # units x r
        Uz = self.M @ gu                                # zones x r
        c = self.M @ gc                                 # zones
        E = (self.H @ beta)[None, :] + c[:, None] + Uz @ HW.T
        a = self.log_q - E
        log_z = float(logsumexp(a)) - self.log_q_total
        p = np.exp(a - a.max())
        p /= p.sum()
        r = self.N * self.op.forward(p) - self.Y
        s = self.sigma.solve(r)
        mean_e = float((p * E).sum())
        kl = -mean_e - log_z
        misfit = 0.5 * float(r @ s)
        value = misfit + self.entropy * kl
        parts = {"misfit": misfit, "kl": kl}
        if not gradient:
            return value, parts, p, None
        G = self.N * self.op.adjoint(s)
        dE = -p * (G - float((p * G).sum())) + self.entropy * p * (E - mean_e)
        grads = (self.HT @ dE.sum(axis=0),
                 self.M.T @ dE.sum(axis=1),
                 self.M.T @ (dE @ HW),
                 self.HT @ (dE.T @ Uz))
        return value, parts, p, grads


@dataclass
class LowRankFit:
    W: np.ndarray
    areas: dict            # area id -> {"beta", "gamma_c", "gamma_u", "theta", "c", "p", "parts"}
    objective: float
    prior: float
    iterations: int
    evaluations: int
    converged: bool
    message: str
    seconds: float
    start: str
    grad_max: float        # largest |gradient| component at the end


class LowRankModel:
    """The summed objective over areas, with ``W`` shared."""

    def __init__(self, terms: dict[str, AreaTerm], rank: int, *, tau_beta: float = 3.0,
                 tau_intercept: float = 3.0, tau_u: float = 3.0, tau_w: float = 1.0) -> None:
        self.terms = terms
        self.ids = list(terms)
        self.k = next(iter(terms.values())).k
        self.r = int(rank)
        self.prec = {"beta": 1 / tau_beta**2, "c": 1 / tau_intercept**2,
                     "u": 1 / tau_u**2, "w": 1 / tau_w**2}
        self.offsets, start = {}, self.k * self.r
        for i in self.ids:
            t = terms[i]
            sizes = (self.k, t.Z - 1, (t.Z - 1) * self.r)
            self.offsets[i] = (start, sizes)
            start += sum(sizes)
        self.n_params = start

    def unpack(self, x: np.ndarray):
        W = x[: self.k * self.r].reshape(self.k, self.r)
        areas = {}
        for i in self.ids:
            o, (nb, nc, nu) = self.offsets[i]
            Z = self.terms[i].Z
            areas[i] = (x[o:o + nb], x[o + nb:o + nb + nc],
                        x[o + nb + nc:o + nb + nc + nu].reshape(Z - 1, self.r))
        return W, areas

    def pack(self, W, areas) -> np.ndarray:
        x = np.zeros(self.n_params)
        x[: self.k * self.r] = W.ravel()
        for i in self.ids:
            o, (nb, nc, nu) = self.offsets[i]
            beta, gc, gu = areas[i]
            x[o:o + nb], x[o + nb:o + nb + nc] = beta, gc
            x[o + nb + nc:o + nb + nc + nu] = np.asarray(gu).ravel()
        return x

    def prior(self, x: np.ndarray) -> float:
        W, areas = self.unpack(x)
        value = 0.5 * self.prec["w"] * float(np.square(W).sum())
        for beta, gc, gu in areas.values():
            value += 0.5 * (self.prec["beta"] * float(beta @ beta) + self.prec["c"] * float(gc @ gc)
                            + self.prec["u"] * float(np.square(gu).sum()))
        return value

    def evaluate(self, x: np.ndarray):
        W, areas = self.unpack(x)
        value = self.prior(x)
        g = np.zeros_like(x)
        gW = self.prec["w"] * W
        for i in self.ids:
            beta, gc, gu = areas[i]
            v, _, _, (g_b, g_c, g_u, g_w) = self.terms[i].evaluate(beta, gc, gu, W)
            value += v
            o, (nb, nc, nu) = self.offsets[i]
            g[o:o + nb] = g_b + self.prec["beta"] * beta
            g[o + nb:o + nb + nc] = g_c + self.prec["c"] * gc
            g[o + nb + nc:o + nb + nc + nu] = (g_u + self.prec["u"] * gu).ravel()
            gW += g_w
        g[: self.k * self.r] = gW.ravel()
        return value, g

    def shrink(self, x: np.ndarray, grid=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0)
               ) -> tuple[np.ndarray, float]:
        """``x`` with every area's zone parameters (Γ_c, Γ_u) scaled by the
        factor in ``grid`` with the lowest objective; a truncated start can
        be far worse than a shrunken one."""
        best = None
        for t in grid:
            W, areas = self.unpack(x)
            y = self.pack(W, {i: (b, t * gc, t * gu) for i, (b, gc, gu) in areas.items()})
            value = self.evaluate(y)[0]
            if best is None or value < best[1]:
                best = (y, value, t)
        return best[0], best[2]

    def fit(self, x0: np.ndarray, *, max_iter: int = 30000, tol: float = 1e-10,
            start: str = "", label: str = "") -> LowRankFit:
        t0 = time.perf_counter()
        counter = {"n": 0}

        def fun(x):
            counter["n"] += 1
            return self.evaluate(x)

        result = scipy.optimize.minimize(
            fun, x0, jac=True, method="L-BFGS-B",
            options={"maxiter": max_iter, "maxfun": 2 * max_iter, "ftol": tol, "gtol": 1e-8,
                     "maxcor": 30})
        W, areas = self.unpack(result.x)
        out, total = {}, self.prior(result.x)
        for i in self.ids:
            beta, gc, gu = areas[i]
            term = self.terms[i]
            v, parts, p, _ = term.evaluate(beta, gc, gu, W, gradient=False)
            total += v
            out[i] = {"beta": beta, "gamma_c": gc, "gamma_u": gu, "theta": term.M @ gu @ W.T,
                      "c": term.M @ gc, "p": p, "parts": parts, "value": v}
        seconds = time.perf_counter() - t0
        grad_max = float(np.abs(self.evaluate(result.x)[1]).max())
        logger.info("%s low-rank fit from %s (r %d, %d parameters): %s after %d iterations, "
                    "f %.6g, max |g| %.3g, %.0fs", label, start, self.r, self.n_params,
                    "converged" if result.success else "NOT converged", result.nit, total,
                    grad_max, seconds)
        return LowRankFit(W=W, areas=out, objective=total, prior=self.prior(result.x),
                          iterations=int(result.nit), evaluations=counter["n"],
                          converged=bool(result.success), message=str(result.message),
                          seconds=seconds, start=start, grad_max=grad_max)


def pca_start(model: LowRankModel, H: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """``W`` the top r principal directions of the standardised features."""
    w = weight / weight.sum()
    mean = w @ H
    cov = (H - mean).T @ ((H - mean) * w[:, None])
    vals, vecs = np.linalg.eigh(cov)
    W = vecs[:, np.argsort(vals)[::-1][: model.r]]
    areas = {i: (np.zeros(model.k), np.zeros(model.terms[i].Z - 1),
                 np.zeros((model.terms[i].Z - 1, model.r))) for i in model.ids}
    return model.pack(W, areas)


def svd_start(model: LowRankModel, full: dict[str, dict], cov: np.ndarray | None = None
              ) -> np.ndarray:
    """A start from per-area full-rank fits, given per area as statewide-scale
    ``theta`` (zones x k, without the intercept), ``c`` (zones) and ``beta``
    (k).

    The zone coefficients of all areas are stacked and truncated to rank r in
    the metric of ``cov`` (the features' covariance; identity if None), which
    best approximates the zones' exponents ``h′θ_b`` rather than the
    coefficients: with ``cov = L L′``, ``W = L′⁻¹ V_r`` and ``u_b = L′θ_b V_r``
    for ``V_r`` the top r right singular vectors of ``Θ L``. Each zone's
    intercept takes up the change in its mean exponent over the area's units,
    and each column of ``W`` and of the ``u`` is rescaled to balance their
    priors (which leaves ``θ`` unchanged). At r = k the start is the full-rank
    fit exactly."""
    k = model.k
    if cov is None:
        L, L_inv_t = np.eye(k), np.eye(k)
    else:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-8 * vals.max())
        L, L_inv_t = vecs * np.sqrt(vals), vecs / np.sqrt(vals)
    stacked = np.vstack([full[i]["theta"] for i in model.ids])
    # Full SVD: beyond the stacked zones' rank its right singular vectors
    # complete an orthonormal basis, so any r up to the feature count works.
    _, _, vt = np.linalg.svd(stacked @ L, full_matrices=True)
    V = vt[: model.r].T
    W = L_inv_t @ V
    U = {i: full[i]["theta"] @ L @ V for i in model.ids}
    u_norm = np.sqrt(sum(np.square(U[i]).sum(axis=0) for i in model.ids))
    w_norm = np.sqrt(np.square(W).sum(axis=0))
    # a minimises a² |W_j|² / τ_w² + |u_j|² / (a² τ_u²): a⁴ = (|u_j| τ_w / (|W_j| τ_u))².
    ratio = np.sqrt(model.prec["u"] / model.prec["w"])
    scale = np.where((u_norm > 0) & (w_norm > 0),
                     np.sqrt(np.maximum(u_norm, 1e-300) / np.maximum(w_norm, 1e-300) * ratio),
                     1.0)
    W = W * scale
    areas = {}
    for i in model.ids:
        term = model.terms[i]
        u = U[i] / scale
        w = term.inputs.units["weight"].to_numpy(dtype=np.float64)
        mean_h = w @ term.H / w.sum()
        c = full[i]["c"] + (full[i]["theta"] - u @ W.T) @ mean_h
        areas[i] = (full[i]["beta"], term.M.T @ c, term.M.T @ u)
    return model.pack(W, areas)


def to_statewide(theta_region: np.ndarray, names_region: list[str], spread_region: np.ndarray,
                 names_state: list[str], spread_state: np.ndarray) -> np.ndarray:
    """Coefficients on an area's standardisation, moved onto the statewide
    one: ``θ_g = θ_r · s_g / s_r`` per feature. The centring difference is a
    per-zone shift of the intercept, :func:`centring_shift`. Features the
    area dropped (constant there) get zero."""
    out = np.zeros(theta_region.shape[:-1] + (len(names_state),))
    where = {n: j for j, n in enumerate(names_region)}
    for j, n in enumerate(names_state):
        if n in where:
            jr = where[n]
            out[..., j] = theta_region[..., jr] * spread_state[j] / spread_region[jr]
    return out
