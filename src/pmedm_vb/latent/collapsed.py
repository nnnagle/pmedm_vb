"""The collapsed likelihood of the block parameters (step 2, piece B2 of
``PLAN_latent.md``; ``PLAN_latent_step2.md`` version 2, §3, §4.2).

From ``MODEL_latent.qmd`` (notation kept), integrating the realized
populations out:

* ``p_bi = q_i γ^{d(P,i)} exp{θ_b' h(z_i) - A_P(θ_b)}``, ``θ_b = θ_P + u_b``;
* ``Y | θ_P, U ~ N(m, S)``, ``m = Σ_b N_b D_b T' p_b`` and
  ``S = Σ + Σ_b N_b D_b V_b D_b'``, ``V_b = T'(diag p_b - p_b p_b')T``;
* ``L(θ_P, U) = -½ (Y - m)' S⁻¹ (Y - m) - ½ log det S`` (no constant);
* ``Σ = Σ_survey + ω² I`` (:mod:`pmedm_vb.latent.area`).

**Structure** (§3). ``Σ_survey = D + LL'`` with ``L`` the scaled replicate
deviations over every row (untapered, rank ≤ 80), and ``D_b`` reaches only
block ``b``'s own rows and its tract's, so the realization term ``R`` is block
diagonal by tract. Hence ``S = B + LL'``, ``B = blockdiag_t(B_t)``,
``B_t = D_t + ω² I + R_t``, each ``B_t`` assembled densely in the local
layout of :meth:`~pmedm_vb.latent.area.AreaInputs.tract_rows` and factorized
by Cholesky. Exactly (Woodbury identity, matrix determinant lemma), with
``X_t = B_t⁻¹ L_t`` and ``K = I + Σ_t L_t' X_t`` (``80 × 80``):

    S⁻¹ = B⁻¹ - B⁻¹ L K⁻¹ L' B⁻¹,       log det S = Σ_t log det B_t + log det K,
    [S⁻¹]_{tt'} = δ_{tt'} B_t⁻¹ - X_t K⁻¹ X_{t'}'.

No approximation. (The model's ``L``, the log-likelihood, and this ``L``, the
replicate factor, share a letter; the code calls the factor ``Lr``.)

**Base cells.** ``T`` enters through the base cells (``T0``, ``M × J0``;
:mod:`pmedm_vb.latent.area`): ``t_i = E t0_i`` with ``E`` the 0/1 map from
base cells to a block's local positions.

**The pair operator** (:class:`PairOperator`). Identical rows of ``T0`` are
merged (``U`` unique rows; ``Pu = Agg p`` sums each row's atoms). For a row
with nonzeros ``v_a`` at cells ``k_a`` the matrix ``Q`` holds, for every pair
``a ≤ b``, the entry ``c_ab v_a v_b`` (``c = 2`` off the diagonal, 1 on it) in
the column of the cell pair ``(k_a, k_b)``. Then, for every block at once,

* second moments: ``(T0' diag(p_b) T0)_{kl} = (Q' Pu)_{(k,l), b} / c_kl``;
* quadratic forms: ``t0_i' G t0_i = (Q g)_i`` with ``g_(k,l) = G_kl``.

Both cost ``Σ_u nnz(t_u)²/2`` per block (the §3 estimate), as one sparse
product with ``B`` columns, split over threads by rows.

**Gradient** (derived here; §3 gives the ``log det`` part). With
``r = S⁻¹(Y - m)``, ``w_b = E'D_b' r`` (block ``b``'s residual weights on the
base cells), ``a_i = t0_i' w_b``, ``ā_b = μ_b' w_b`` (``μ_b = T0' p_b``) and
``G_b = E' D_b' [S⁻¹]_{tt} D_b E`` (``t`` block ``b``'s tract):

    ∂L/∂u_b = N_b Cov_{p_b}(h, φ_b),
    φ_bi = a_i + ½(a_i² - 2 a_i ā_b)                   [quadratic form: mean and S]
           - ½(t0_i' G_b t0_i - 2 t0_i' G_b μ_b)       [log det S]

and ``∂L/∂θ_P = Σ_b ∂L/∂u_b``. The first line comes from ``dm`` and from
``S``'s dependence in the quadratic form (``½ r' dS r``), the second from
``-½ tr(S⁻¹ dS)``; both use ``dp_bi = p_bi (h_i - h̄_b)' du_b`` and
``tr(G dV_b) = Σ_i dp_bi (t_i'G t_i - 2 t_i'G μ_b)``. Since ``dS`` is block
diagonal by tract, only the tract blocks ``[S⁻¹]_{tt}`` enter; ``B_t⁻¹``
comes from the Cholesky factor (LAPACK ``potri``).

**Timings** (:attr:`Evaluation.timings`, seconds): ``build`` (``p``, ``m``,
``B_t``), ``factor`` (Cholesky of every ``B_t``, ``X_t`` and ``K``), ``loglik``
(solves and ``log det``), ``gradient`` (without the ``log det`` term),
``logdet_gradient`` (the ``log det`` term, including ``[S⁻¹]_{tt}``) and
``total``.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, lapack
from scipy.special import logsumexp

from pmedm_vb.latent.area import AreaInputs


def default_workers() -> int:
    """CPUs this process may use."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover -- not on Linux
        return os.cpu_count() or 1


class RowBlocks:
    """A CSR matrix held as row blocks of about equal nonzeros, so that
    ``A @ X`` runs one block per thread (SciPy's sparse kernels release the
    GIL)."""

    def __init__(self, A: sp.spmatrix, workers: int):
        A = sp.csr_matrix(A)
        self.shape = A.shape
        self.nnz = A.nnz
        self.workers = max(1, int(workers))
        parts = self.workers if A.shape[0] >= 2 * self.workers else 1
        cuts = np.searchsorted(A.indptr, np.linspace(0, A.nnz, parts + 1)[1:-1])
        self.bounds = np.unique(np.r_[0, cuts, A.shape[0]])
        self.blocks = [A[lo:hi] for lo, hi in zip(self.bounds[:-1], self.bounds[1:])]

    def __matmul__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = np.empty((self.shape[0], *X.shape[1:]))
        if len(self.blocks) == 1:
            out[:] = self.blocks[0] @ X
            return out

        def run(k):
            out[self.bounds[k]:self.bounds[k + 1]] = self.blocks[k] @ X

        with ThreadPoolExecutor(len(self.blocks)) as pool:
            list(pool.map(run, range(len(self.blocks))))
        return out

    @property
    def nbytes(self) -> int:
        return sum(b.data.nbytes + b.indices.nbytes + b.indptr.nbytes for b in self.blocks)


def unique_rows(T: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """``(inverse, first)``: each row's unique-row index and each unique
    row's first occurrence."""
    T = sp.csr_matrix(T)
    T.sort_indices()
    seen: dict[bytes, int] = {}
    inverse = np.empty(T.shape[0], dtype=np.int64)
    first = []
    for i in range(T.shape[0]):
        lo, hi = T.indptr[i], T.indptr[i + 1]
        key = T.indices[lo:hi].tobytes() + T.data[lo:hi].tobytes()
        u = seen.get(key)
        if u is None:
            u = seen[key] = len(first)
            first.append(i)
        inverse[i] = u
    return inverse, np.array(first, dtype=np.int64)


class PairOperator:
    """``Q`` over the unique rows of the base cells ``T0`` (module notes).
    Independent of the PUMA: built once per process."""

    def __init__(self, T0: sp.spmatrix, workers: int | None = None):
        start = time.perf_counter()
        workers = default_workers() if workers is None else workers
        T0 = sp.csr_matrix(T0, dtype=float)
        T0.sum_duplicates()
        T0.sort_indices()
        M, J0 = T0.shape
        self.M, self.J0, self.workers = M, J0, workers
        self.inverse, first = unique_rows(T0)
        Tu = T0[first]
        Tu.sort_indices()
        self.U = Tu.shape[0]
        self.agg = sp.csr_matrix((np.ones(M), (self.inverse, np.arange(M))), shape=(self.U, M))
        self.Tu = RowBlocks(Tu, workers)
        self.TuT = RowBlocks(Tu.T.tocsr(), workers)
        nnz = np.diff(Tu.indptr)
        counts = nnz * (nnz + 1) // 2
        indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
        total = int(indptr[-1])
        codes = np.empty(total, dtype=np.int64)
        data = np.empty(total)
        for n in np.unique(nnz):
            if n == 0:
                continue
            rows = np.flatnonzero(nnz == n)
            cells = Tu.indptr[rows][:, None] + np.arange(n)[None, :]
            k, v = Tu.indices[cells].astype(np.int64), Tu.data[cells]
            a, b = np.triu_indices(n)
            slots = (indptr[rows][:, None] + np.arange(a.size)[None, :]).ravel()
            codes[slots] = (k[:, a] * J0 + k[:, b]).ravel()
            data[slots] = (v[:, a] * v[:, b] * np.where(a == b, 1.0, 2.0)[None, :]).ravel()
        used = np.zeros(J0 * J0, dtype=bool)
        used[codes] = True
        pairs = np.flatnonzero(used)
        column = np.cumsum(used) - 1
        index_dtype = np.int32 if pairs.size < 2**31 and total < 2**31 else np.int64
        Q = sp.csr_matrix((data, column[codes].astype(index_dtype), indptr.astype(index_dtype)),
                          shape=(self.U, pairs.size))
        del codes, data
        self.pair_k, self.pair_l = pairs // J0, pairs % J0
        self.pair_c = np.where(self.pair_k == self.pair_l, 1.0, 2.0)
        self.n_pairs = pairs.size
        self.Q = RowBlocks(Q, workers)
        self.Qt = RowBlocks(Q.T.tocsr(), workers)
        self.seconds = time.perf_counter() - start

    def aggregate(self, P: np.ndarray) -> np.ndarray:
        """``Pu``: per-atom columns summed over identical rows."""
        return np.asarray(self.agg @ P)

    def expand(self, F: np.ndarray) -> np.ndarray:
        """Per-unique-row values to per-atom values."""
        return F[self.inverse]

    def second_moments(self, Pu: np.ndarray) -> np.ndarray:
        """``(n_pairs, B)``: ``Q' Pu`` (packed; :meth:`unpack` divides by ``c``)."""
        return self.Qt @ Pu

    def unpack(self, packed: np.ndarray) -> np.ndarray:
        """The symmetric ``J0 × J0`` matrix of one column of :meth:`second_moments`."""
        W = np.zeros((self.J0, self.J0))
        values = packed / self.pair_c
        W[self.pair_k, self.pair_l] = values
        W[self.pair_l, self.pair_k] = values
        return W

    def pack(self, G: np.ndarray) -> np.ndarray:
        """``g`` of a symmetric ``G``, for :meth:`quadratic_forms`."""
        return G[self.pair_k, self.pair_l]

    def quadratic_forms(self, packed: np.ndarray) -> np.ndarray:
        """``(U, B)``: ``t0_u' G_b t0_u`` for every unique row and column."""
        return self.Q @ packed

    def summary(self) -> dict:
        return {"M": self.M, "J0": self.J0, "unique_rows": self.U, "n_pairs": self.n_pairs,
                "Q_nnz": self.Q.nnz, "Q_bytes_both_orientations": self.Q.nbytes + self.Qt.nbytes,
                "workers": self.workers, "seconds": self.seconds}


@dataclass
class Evaluation:
    loglik: float
    quad: float
    logdet: float
    m: np.ndarray
    mu: np.ndarray                  # (J0, B) T0' p_b
    p: np.ndarray                   # (M, B)
    factors: list                   # Cholesky factor of B_t per tract
    X: list                         # B_t⁻¹ L_t per tract (None without a replicate factor)
    K_factor: tuple | None          # Cholesky factor of K = I + L'B⁻¹L
    realization_diag: np.ndarray    # diag of Σ_b N_b D_b V_b D_b' over the stacked rows
    grad_U: np.ndarray | None = None
    grad_U_quad: np.ndarray | None = None
    grad_U_logdet: np.ndarray | None = None
    timings: dict = field(default_factory=dict)

    @property
    def grad_theta(self) -> np.ndarray | None:
        return None if self.grad_U is None else self.grad_U.sum(axis=0)


class CollapsedLikelihood:
    """``L(θ_P, U)`` and its gradient for one PUMA (module notes).

    ``H`` is the basis over all ``M`` atoms, ``logb`` the log base weights
    ``log q_i + d(P,i) log γ`` (:func:`~pmedm_vb.latent.reference.log_base`).
    """

    def __init__(self, area: AreaInputs, H: np.ndarray, logb: np.ndarray, pairs: PairOperator):
        if H.shape[0] != pairs.M or logb.shape != (pairs.M,):
            raise ValueError("H, logb and the pair operator disagree on the atoms")
        if area.columns.base_columns.size != pairs.J0:
            raise ValueError("the area's base cells and the pair operator's differ")
        self.area, self.H, self.logb, self.pairs = area, H, logb, pairs
        self.K = H.shape[1]
        self.N = area.N_b.astype(float)
        self.src = area.sources
        self.ts, self.bs = area.columns.tract_src, area.columns.bg_src
        sigma = area.sigma
        if sigma.is_tapered:
            raise ValueError("the engine expects the untapered Sigma (model version 2)")
        self.d = sigma.d                # D + ω² (Sigma.d carries prior_var)
        self.Lr = sigma.b               # (n_rows, 80) or None
        self.tracts = []
        for t in range(area.n_tracts):
            rows = area.tract_rows(t)
            members = area.tract_members(t)
            self.tracts.append({"rows": rows, "members": members,
                                "local": [area.local_positions(j) for j in range(len(members))],
                                "Lr": None if self.Lr is None else self.Lr[rows]})

    # -- pieces --------------------------------------------------------------

    def thetas(self, theta_P: np.ndarray, U: np.ndarray) -> np.ndarray:
        U = np.zeros((self.area.B, self.K)) if U is None else np.asarray(U, float)
        return np.asarray(theta_P, float)[None, :] + U

    def probabilities(self, theta_P, U) -> np.ndarray:
        """``(M, B)``: ``p_b`` for every block group."""
        eta = self.H @ self.thetas(theta_P, U).T
        eta += self.logb[:, None]
        eta -= logsumexp(eta, axis=0)[None, :]
        return np.exp(eta, out=eta)

    def stack(self, mu: np.ndarray) -> np.ndarray:
        """``m = Σ_b N_b D_b E μ_b`` over the stacked rows, from ``μ`` (``J0 × B``)."""
        area = self.area
        weighted = mu * self.N[None, :]
        onehot = np.zeros((area.B, area.n_tracts))
        onehot[np.arange(area.B), area.bg_tract] = 1.0
        m_T = weighted[self.ts] @ onehot
        m_B = weighted[self.bs]
        return np.concatenate([m_T.ravel(), m_B.ravel()])

    def mean(self, theta_P, U=None) -> np.ndarray:
        """``m(θ_P, U)`` alone."""
        p = self.probabilities(theta_P, U)
        return self.stack(self.pairs.TuT @ self.pairs.aggregate(p))

    def fold(self, G: np.ndarray) -> np.ndarray:
        """``E' G E``: a block's local ``(J_T + J_B)`` square matrix onto the base cells."""
        J_T, ts, bs = len(self.ts), self.ts, self.bs
        G0 = np.zeros((self.pairs.J0, self.pairs.J0))
        G0[np.ix_(ts, ts)] += G[:J_T, :J_T]
        G0[np.ix_(ts, bs)] += G[:J_T, J_T:]
        G0[np.ix_(bs, ts)] += G[J_T:, :J_T]
        G0[np.ix_(bs, bs)] += G[J_T:, J_T:]
        return G0

    def covariance_with_h(self, p: np.ndarray, F: np.ndarray) -> np.ndarray:
        """``(B, K)``: ``Cov_{p_b}(h, F_b)`` per column."""
        pF = p * F
        return (self.H.T @ pF - (self.H.T @ p) * pF.sum(axis=0)[None, :]).T

    @staticmethod
    def tract_inverse(factor, Xt, K_factor) -> np.ndarray:
        """``[S⁻¹]_{tt} = B_t⁻¹ - X_t K⁻¹ X_t'``."""
        c, lower = factor
        inverse, info = lapack.dpotri(c, lower=lower)
        if info != 0:
            raise np.linalg.LinAlgError(f"potri failed ({info})")
        inverse = np.tril(inverse) + np.tril(inverse, -1).T if lower else \
            np.triu(inverse) + np.triu(inverse, 1).T
        if K_factor is not None:
            inverse -= Xt @ cho_solve(K_factor, Xt.T, check_finite=False)
        return inverse

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, theta_P, U=None, *, gradient: bool = True,
                 keep_S: bool = False) -> Evaluation:
        """``L`` at ``(θ_P, U)`` and, with ``gradient``, its gradient (module
        notes). ``keep_S`` stores each ``B_t`` (for checks) in
        ``timings['S']``."""
        area, pairs, N = self.area, self.pairs, self.N
        clock = time.perf_counter
        times = {}
        start = t0 = clock()
        p = self.probabilities(theta_P, U)
        Pu = pairs.aggregate(p)
        mu = pairs.TuT @ Pu
        W = pairs.second_moments(Pu)
        m = self.stack(mu)
        realization = np.zeros(area.n_rows)
        blocks, kept = [], []
        for tract in self.tracts:
            S = np.diag(self.d[tract["rows"]])
            for j, b in enumerate(tract["members"]):
                if N[b] <= 0:
                    continue
                V = pairs.unpack(W[:, b])
                V -= np.outer(mu[:, b], mu[:, b])
                local = tract["local"][j]
                S[np.ix_(local, local)] += N[b] * V[np.ix_(self.src, self.src)]
            realization[tract["rows"]] = np.diag(S) - self.d[tract["rows"]]
            blocks.append(S)
        times["build"] = clock() - t0

        t0 = clock()
        factors = []
        for S in blocks:
            if keep_S:
                kept.append(S.copy())
            factors.append(cho_factor(S, lower=True, overwrite_a=True, check_finite=False))
        del blocks
        X, K_factor = [None] * len(factors), None
        if self.Lr is not None:
            K = np.eye(self.Lr.shape[1])
            for k, (tract, factor) in enumerate(zip(self.tracts, factors)):
                X[k] = cho_solve(factor, tract["Lr"], check_finite=False)
                K += tract["Lr"].T @ X[k]
            K_factor = cho_factor(K, lower=True, check_finite=False)
        times["factor"] = clock() - t0

        t0 = clock()
        quad = logdet = 0.0
        residual, Lz = [], 0.0
        for tract, factor in zip(self.tracts, factors):
            e = area.Y[tract["rows"]] - m[tract["rows"]]
            z = cho_solve(factor, e, check_finite=False)
            quad += float(e @ z)
            logdet += 2.0 * float(np.log(np.diag(factor[0])).sum())
            residual.append(z)
            if K_factor is not None:
                Lz = Lz + tract["Lr"].T @ z
        if K_factor is not None:
            c = cho_solve(K_factor, Lz, check_finite=False)
            quad -= float(Lz @ c)
            logdet += 2.0 * float(np.log(np.diag(K_factor[0])).sum())
            residual = [z - Xt @ c for z, Xt in zip(residual, X)]
        loglik = -0.5 * quad - 0.5 * logdet
        times["loglik"] = clock() - t0

        ev = Evaluation(loglik=loglik, quad=quad, logdet=logdet, m=m, mu=mu, p=p,
                        factors=factors, X=X, K_factor=K_factor,
                        realization_diag=realization, timings=times)
        if keep_S:
            times["S"] = kept
        if not gradient:
            times["total"] = clock() - start
            return ev

        t0 = clock()
        J_T, J_B = area.J_T, area.J_B
        w0 = np.zeros((pairs.J0, area.B))
        for tract, r in zip(self.tracts, residual):
            for j, b in enumerate(tract["members"]):
                w0[self.ts, b] += r[:J_T]
                w0[self.bs, b] += r[J_T + j * J_B: J_T + (j + 1) * J_B]
        a = pairs.Tu @ w0
        abar = (mu * w0).sum(axis=0)
        phi = pairs.expand(a + 0.5 * (a * a - 2.0 * a * abar[None, :]))
        grad_quad = N[:, None] * self.covariance_with_h(p, phi)
        del phi
        times["gradient"] = clock() - t0

        t0 = clock()
        packed = np.zeros((pairs.n_pairs, area.B))
        Gmu = np.zeros((pairs.J0, area.B))
        for tract, factor, Xt in zip(self.tracts, factors, X):
            inverse = self.tract_inverse(factor, Xt, K_factor)
            for j, b in enumerate(tract["members"]):
                if N[b] <= 0:
                    continue
                local = tract["local"][j]
                G0 = self.fold(inverse[np.ix_(local, local)])
                packed[:, b] = pairs.pack(G0)
                Gmu[:, b] = G0 @ mu[:, b]
        phi = pairs.expand(-0.5 * (pairs.quadratic_forms(packed) - 2.0 * (pairs.Tu @ Gmu)))
        grad_logdet = N[:, None] * self.covariance_with_h(p, phi)
        del phi
        times["logdet_gradient"] = clock() - t0
        times["total"] = clock() - start
        ev.grad_U_quad, ev.grad_U_logdet = grad_quad, grad_logdet
        ev.grad_U = grad_quad + grad_logdet
        return ev
