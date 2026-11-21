"""Stage 2 of the posterior: Laplace for the zone deviations, Gaussian conditioning
for the realized population (step 4, piece C2 of ``PLAN_latent.md``;
``PLAN_latent_step4.md`` version 1, §4.2-§4.3; ``MODEL_latent.qmd`` version 3,
stage 2).

**Laplace** (:func:`laplace_factor`, :func:`draw_U`). At the MAP ``Û`` (for
the region's engine on the deviation basis ``g``,
:func:`~pmedm_vb.latent.region_engine.deviation_engine`),
``U ~ N(Û, H⁻¹)`` with ``H = I(Û) + τ⁻² I``, ``I`` the tables' mean-part
(Gauss–Newton) information (:func:`~pmedm_vb.latent.region_engine.information`).
``θ_A`` and ``τ`` are held at their fitted values (an approximation,
recorded in the plan). Draws: ``U = Û + R⁻¹ ξ``, ``R'R = H``, ``ξ ~ N(0, I)``.

**Gaussian conditioning** (:func:`conditional_moments`). For a draw of ``U``
the engine gives ``p_b``, ``m`` and ``S`` (its per-zone Cholesky factors and
Woodbury pieces). For a functional ``g'C_b`` of zone ``b``'s realized
population (``g`` a column of :attr:`Functionals.G`, over the atoms),
from the model (stage 2):

    E[g'C_b | ·, Y]   = N_b g'p_b + k_b' S⁻¹ (Y - m),
    Var[g'C_b | ·, Y] = N_b g'(diag p_b - p_b p_b') g - k_b' S⁻¹ k_b,
    k_b = N_b D_b T'(diag p_b - p_b p_b') g = N_b D_b Cov_{p_b}(t, g).

In the regional run ``D_b`` reaches only zone ``b``'s own rows, so ``k_b``
lives on them; with ``S = B + LL'`` (``B`` block diagonal by zone, ``X_b =
B_b⁻¹ L_b``, ``K = I + L'B⁻¹L``) the quadratic form is
``k'B_b⁻¹k - (X_b'k)' K⁻¹ (X_b'k)`` and ``S⁻¹(Y - m)`` is one Woodbury solve
per draw. A variance below zero (rounding) is set to zero and counted.

**Samples** (:func:`posterior_samples`): one Gaussian draw of every
functional per draw of ``U``; ``S`` draws give the posterior sample, from
which intervals are empirical quantiles
(:mod:`pmedm_vb.latent.calibration`).

**Functionals** (:func:`build_functionals`, §4.3): the fitted cells (S1), the
cells of the held-out sets B1 and B2, the 15 rare-group totals of the
regional scoring, and the cells of the two-way scoring tables, each a column
of per-atom counts; :func:`zone_truth` gives each zone's true value (its own
records, design-weighted).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_solve, cholesky, solve_triangular

from pmedm_vb.embed.puma_check import RARE, CheckData
from pmedm_vb.latent.collapsed import CollapsedLikelihood, Evaluation
from pmedm_vb.latent.region_engine import information

CLASSES = ("fitted", "B1", "B2", "rare", "joint")


@dataclass
class Functionals:
    G: sp.csc_matrix            # (M, F) per-atom contributions
    cls: np.ndarray             # (F,) class of each functional
    names: np.ndarray           # (F,) "table:category"

    @property
    def F(self) -> int:
        return self.G.shape[1]

    def of(self, c: str) -> np.ndarray:
        return np.flatnonzero(self.cls == c)


def _cells(data: CheckData, tables) -> tuple[sp.spmatrix, list[str]]:
    X = sp.hstack([data.counts[t] for t in tables]).tocsc()
    names = [f"{t}:{c.name}" for t in tables for c in data.tables[t].categories]
    if X.shape[1] != len(names):
        raise ValueError("cell names do not match the counts' columns")
    return X, names


def build_functionals(data: CheckData, sets: dict[str, tuple[str, ...]]) -> Functionals:
    """``sets``: the table sets ``A`` (fitted), ``B1`` and ``B2`` (module notes)."""
    blocks, cls, names = [], [], []
    for c, key in (("fitted", "A"), ("B1", "B1"), ("B2", "B2")):
        X, n = _cells(data, sets[key])
        blocks.append(X)
        cls += [c] * len(n)
        names += n
    for table, category in RARE:
        cats = [c.name for c in data.tables[table].categories]
        blocks.append(data.counts[table][:, cats.index(category)])
        cls.append("rare")
        names.append(f"{table}:{category}")
    joint = [t for t in data.sets.scoring if t in data.two_way]
    if "B17010A-G" in joint:
        X = data.counts["B17010A-G"]
        blocks.append(X)
        cls += ["joint"] * X.shape[1]
        names += [f"B17010A-G:{k}" for k in range(X.shape[1])]
        joint.remove("B17010A-G")
    X, n = _cells(data, joint)
    blocks.append(X)
    cls += ["joint"] * len(n)
    names += n
    G = sp.hstack(blocks).tocsc().astype(float)
    return Functionals(G=G, cls=np.array(cls), names=np.array(names))


def zone_truth(fun: Functionals, zone: np.ndarray, weight: np.ndarray, Z: int) -> np.ndarray:
    """``(Z, F)``: each zone's true value of every functional."""
    out = np.zeros((Z, fun.F))
    Gr = fun.G.tocsr()
    for b in range(Z):
        own = np.flatnonzero(zone == b)
        out[b] = np.asarray(Gr[own].T @ weight[own]).ravel()
    return out


# --------------------------------------------------------------------------
# Laplace
# --------------------------------------------------------------------------


def laplace_factor(engine: CollapsedLikelihood, theta: np.ndarray, U: np.ndarray, tau: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """``H = I(Û) + τ⁻² I`` and its upper Cholesky factor ``R`` (``R'R = H``)."""
    ev = engine.evaluate(theta, U, gradient=False)
    H = information(engine, ev)
    H[np.diag_indices_from(H)] += 1.0 / tau ** 2
    return H, cholesky(H, lower=False)


def draw_U(U_hat: np.ndarray, R: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``(n, Z, K_g)`` draws of ``N(Û, H⁻¹)``."""
    xi = rng.standard_normal((U_hat.size, n))
    return (U_hat.ravel()[:, None] + solve_triangular(R, xi, lower=False)).T.reshape(
        (n, *U_hat.shape))


# --------------------------------------------------------------------------
# Gaussian conditioning
# --------------------------------------------------------------------------


def s_inverse_residual(engine: CollapsedLikelihood, ev: Evaluation) -> list[np.ndarray]:
    """``S⁻¹(Y - m)`` per zone (each zone's rows, in its local order)."""
    Y = engine.area.Y
    out, Lz = [], 0.0
    for tract, factor in zip(engine.tracts, ev.factors):
        z = cho_solve(factor, Y[tract["rows"]] - ev.m[tract["rows"]], check_finite=False)
        out.append(z)
        if ev.K_factor is not None:
            Lz = Lz + tract["Lr"].T @ z
    if ev.K_factor is not None:
        c = cho_solve(ev.K_factor, Lz, check_finite=False)
        out = [z - Xt @ c for z, Xt in zip(out, ev.X)]
    return out


def conditional_moments(engine: CollapsedLikelihood, ev: Evaluation, fun: Functionals,
                        T0: sp.spmatrix, GG: sp.spmatrix | None = None
                        ) -> tuple[np.ndarray, np.ndarray, int]:
    """``(Z, F)`` conditional means and variances, and the number of
    variances below zero (set to 0). ``T0`` is the base cells over the atoms
    (the engine's ``J0`` columns), ``GG`` the elementwise square of ``G``."""
    G = fun.G
    GG = G.multiply(G).tocsc() if GG is None else GG
    T0 = sp.csr_matrix(T0)
    r = s_inverse_residual(engine, ev)
    Z = engine.area.B
    mean, var = np.zeros((Z, fun.F)), np.zeros((Z, fun.F))
    negative = 0
    for b, (tract, factor) in enumerate(zip(engine.tracts, ev.factors)):
        if len(tract["members"]) != 1 or tract["members"][0] != b:
            raise ValueError("conditional_moments expects one zone per tract (the regional run)")
        p, N = ev.p[:, b], engine.N[b]
        Eg = np.asarray(G.T @ p).ravel()
        v0 = np.asarray(GG.T @ p).ravel() - Eg ** 2
        TpG = np.asarray((T0.multiply(p[:, None]).T @ G).todense())        # (C, F)
        k = N * (TpG - np.outer(ev.mu[:, b], Eg))
        src = engine.src
        k = k[src]                                                       # zone's local rows
        Y1 = cho_solve(factor, k, check_finite=False)
        quad = np.einsum("ij,ij->j", k, Y1)
        if ev.K_factor is not None:
            W = tract["Lr"].T @ Y1
            quad -= np.einsum("ij,ij->j", W, cho_solve(ev.K_factor, W, check_finite=False))
        mean[b] = N * Eg + k.T @ r[b]
        vb = N * v0 - quad
        negative += int((vb < 0).sum())
        var[b] = np.maximum(vb, 0.0)
    return mean, var, negative


@dataclass
class PosteriorSample:
    samples: np.ndarray        # (S, Z, F) float32
    negative_variances: int
    seconds: float
    seconds_per_draw: float


def posterior_samples(engine: CollapsedLikelihood, theta: np.ndarray, U_hat: np.ndarray,
                      tau: float, fun: Functionals, T0: sp.spmatrix, n_draws: int,
                      rng: np.random.Generator) -> PosteriorSample:
    """``n_draws`` posterior draws of every functional of every zone (module notes)."""
    start = time.perf_counter()
    _, R = laplace_factor(engine, theta, U_hat, tau)
    Us = draw_U(U_hat, R, n_draws, rng)
    GG = fun.G.multiply(fun.G).tocsc()
    out = np.empty((n_draws, engine.area.B, fun.F), dtype=np.float32)
    negative = 0
    for s, U in enumerate(Us):
        ev = engine.evaluate(theta, U, gradient=False)
        mean, var, neg = conditional_moments(engine, ev, fun, T0, GG)
        negative += neg
        out[s] = mean + np.sqrt(var) * rng.standard_normal(mean.shape)
    seconds = time.perf_counter() - start
    return PosteriorSample(samples=out, negative_variances=negative, seconds=seconds,
                           seconds_per_draw=seconds / max(n_draws, 1))
