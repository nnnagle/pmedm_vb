"""Sum-to-zero over the whole null space: every direction the data cannot see.

The data see only each unit's tilt in each zone -- the sum over the rows that
apply there (its tract's, its block group's, its PUMA's) of loading times
multiplier. A direction of the multipliers that changes no tilt is invisible:
the fit along it is set by ``Sigma`` alone. Two sources give such directions
(``experiments/null_directions.py`` counts them): a total measured at two
levels (a tract against its block groups, the PUMA against its tracts), and
two tables sharing a universe or a margin in one area (total population in
B01001 and B03002, households in B19001 and B25003). Level ``"nullspace"`` removes
all of them, with the PUMA rows of ``"puma"`` included.

**Only where the tables agree.** Removing a direction ``v`` also removes the
published combination ``v'Y``. That loses nothing when ``v'Y`` is 0 and so is
every replicate's ``v'Y_r``: the tables publish the same value, with the same
sampling error. The null space is computed per tract and for the PUMA
coupling, and only its part along which the targets and all replicates are
zero (to ``CONSISTENCY_RTOL``) is removed; any remainder is kept and reported.

**Which complement.** The published identities hold exactly, so the errors
``e = Y - N X'p`` satisfy ``v'e = 0`` too, and the coherent error model is
``N(0, Sigma)`` conditioned on that: the multipliers are restricted to the
``Sigma``-orthogonal complement of the removed directions,
``{lambda : N' Sigma lambda = 0}``, whose penalty ``lambda' Sigma lambda`` is
then the conditioned covariance's. Every removed direction has zero replicate
contrast, so ``Sigma N = D N`` for the untapered ``Sigma = D + B B'`` -- ``D``
diagonal -- and the complement is ``D``-orthogonal. In coordinates
``mu = D^{1/2} lambda`` that is an ordinary orthogonal projection, which keeps
the per-tract structure; so the solver's coordinates are ``xi`` with

    zeta = D^{-1/2} (I - Q - W W') xi,   lambda_data = [C', E_P] zeta,

``Q`` block diagonal by tract (the null space inside each tract, orthonormal
in ``mu``) and ``W`` a few global columns (the directions coupling tracts
through the PUMA rows), plus the ridge ``(kappa/2) ||(Q + W W') xi||^2`` of
:mod:`pmedm_vb.assemble.hierarchy`. ``D`` depends on ``alpha`` and the
variance floor, so the object is built for one ``Sigma``. With the tract taper
the replicate term is cut at tract boundaries and ``Sigma N = D N`` holds only
inside tracts; the complement is then ``D``-orthogonal, not exactly
``Sigma``-orthogonal, for the few directions coupling tracts.

Works on the plain layout or on the per-area collapse of
:mod:`pmedm_vb.assemble.collapse` (level ``"nullspace-c<threshold>"``).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pmedm_vb.assemble.collapse import CollapsedHierarchy
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.sigma import Sigma
from pmedm_vb.progress import logger

#: Singular values below this times the largest count as zero, for the data map.
NULL_RTOL = 1e-9
#: A direction is consistent when its target and replicate contrasts are below
#: this times the scale of the targets and replicates it touches.
CONSISTENCY_RTOL = 1e-8


def _orth(a: np.ndarray, rtol: float = 1e-10) -> np.ndarray:
    if not a.size:
        return a.reshape(a.shape[0], 0)
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    return u[:, s > rtol * max(s.max(), 1.0)] if s.size else u[:, :0]


def _null(a: np.ndarray, rtol: float) -> np.ndarray:
    """Orthonormal basis of the null space of ``a`` (columns)."""
    if a.shape[0] == 0:
        return np.eye(a.shape[1])
    _, s, vt = np.linalg.svd(a, full_matrices=a.shape[0] < a.shape[1])
    rank = int((s > rtol * s.max()).sum()) if s.size and s.max() > 0 else 0
    return vt[rank:].T


def _consistent(V: np.ndarray, y: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, int]:
    """The part of span(V) along which ``y`` and every column of ``L`` vanish."""
    if not V.shape[1]:
        return V, 0
    R = V.T @ np.column_stack([y, L])                         # (r, 81)
    scale = max(np.abs(y).max(initial=0.0), np.abs(L).max(initial=0.0), 1.0)
    u, s, _ = np.linalg.svd(R, full_matrices=True)
    rank = int((s > CONSISTENCY_RTOL * scale * np.sqrt(V.shape[0])).sum())
    return V @ u[:, rank:], rank


class NullSpaceHierarchy(CollapsedHierarchy):
    """Level ``"nullspace"``: PUMA rows, and every consistent invisible direction removed.

    Built with :meth:`build_null` for one ``Sigma`` (``alpha``, taper,
    variance floor). Offers the interface of
    :class:`~pmedm_vb.assemble.hierarchy.Hierarchy`.
    """

    @classmethod
    def build_null(cls, inputs: PMEDMInputs, collapse: float | None, *, alpha: float,
                   taper: str | None, variance_floor) -> "NullSpaceHierarchy":
        if alpha is None:
            raise ValueError("the null-space hierarchy is built for one Sigma: pass alpha, "
                             "taper and variance_floor")
        base = (CollapsedHierarchy.build(inputs, "puma", -1.0, merge_zeros=False) if collapse is None
                else CollapsedHierarchy.build(inputs, "puma", collapse))
        self = cls.__new__(cls)
        self.__dict__.update(base.__dict__)
        self.level = "nullspace" if collapse is None else f"nullspace-c{collapse:g}"
        self.base = "nullspace"
        self.group = np.full(self.m, -1)
        self.bg_group = self.group.copy()
        self.settings = (alpha, taper, variance_floor)
        sigma = CollapsedHierarchy.sigma(self, inputs, alpha, taper, variance_floor)
        self.scale = np.sqrt(sigma.d)
        self._find_null(inputs)
        return self

    # -- the null space ------------------------------------------------------

    def _find_null(self, inputs: PMEDMInputs) -> None:
        x_t, x_b = sp.csr_matrix(inputs.X_T), sp.csr_matrix(inputs.X_B)
        n_tracts, c_t = inputs.Y_T.shape
        n_zones, c_b = inputs.Y_B.shape
        split = inputs.Y_T.size
        zone_tract = inputs.zone_tracts()
        k = self.n_puma
        # E: tract category -> PUMA cell
        E = np.zeros((c_t, k))
        for cat in range(c_t):
            E[cat, self.puma_of_row[cat * n_tracts]] = 1.0
        y_cells, l_cells = self.Y_c, self.l_c
        y_p, l_p = self.Y_P, self.l_P
        local, lifts, dropped = [], [], 0
        L_set = None                                          # intersection of the PUMA parts
        per_tract = []
        for t in range(n_tracts):
            cells = self.tract_cells[t]
            pos = np.full(self.m, -1)
            pos[cells] = np.arange(cells.size)
            zones = np.flatnonzero(zone_tract == t)
            rows_t = np.arange(c_t) * n_tracts + t
            blocks_theta, blocks_p = [], []
            for z in zones:
                support = np.flatnonzero(inputs.q[z] > 0)
                d = np.zeros((support.size, cells.size))
                xt = x_t[support].toarray()
                np.add.at(d.T, pos[self.row_cell[rows_t]], xt.T)
                rows_b = split + np.arange(c_b) * n_zones + z
                np.add.at(d.T, pos[self.row_cell[rows_b]], x_b[support].toarray().T)
                blocks_theta.append(d)
                blocks_p.append(xt @ E)
            D_t, D_p = np.vstack(blocks_theta), np.vstack(blocks_p)
            V = _null(D_t, NULL_RTOL)
            V, n_bad = _consistent(V, y_cells[cells], l_cells[cells])
            dropped += n_bad
            local.append(V)
            # (theta_t, lambda_P) pairs with no tilt: their lambda_P parts
            Z = _null(np.hstack([D_t, D_p]), NULL_RTOL)
            Lt = _orth(Z[cells.size:])
            L_set = Lt if L_set is None else _intersect(L_set, Lt)
            per_tract.append((cells, D_t, D_p))
        # Global directions: for each lambda_P in the intersection, the tract parts.
        globals_ = []
        for j in range(L_set.shape[1] if L_set is not None else 0):
            ell = L_set[:, j]
            g = np.zeros(self.size)
            g[self.m:] = ell
            ok = True
            for cells, D_t, D_p in per_tract:
                theta, *_ = np.linalg.lstsq(D_t, -D_p @ ell, rcond=None)
                if np.abs(D_t @ theta + D_p @ ell).max() > 1e-6 * max(1.0, np.abs(D_p @ ell).max()):
                    ok = False
                    break
                g[cells] = theta
            if ok:
                globals_.append(g)
        G = np.column_stack(globals_) if globals_ else np.zeros((self.size, 0))
        y_ext = np.concatenate([y_cells, y_p])
        l_ext = np.vstack([l_cells, l_p])
        G, n_bad = _consistent(_orth(G), y_ext, l_ext) if G.shape[1] else (G, 0)
        dropped += n_bad
        # To mu = D^{1/2} lambda coordinates: orthonormal local blocks, then the
        # global columns orthogonal to them.
        s = self.scale
        self.V = [_orth(s[cells][:, None] * V) for cells, V in zip(self.tract_cells, local)]
        W = s[:, None] * G
        for cells, V in zip(self.tract_cells, self.V):
            if V.shape[1]:
                W[cells] -= V @ (V.T @ W[cells])
        self.W = _orth(W)
        self.n_removed = sum(V.shape[1] for V in self.V) + self.W.shape[1]
        self.n_inconsistent = dropped
        self._Q = [V @ V.T for V in self.V]
        logger.info("null space: %d directions removed (%d within tracts, %d coupling tracts), "
                    "%d inconsistent kept", self.n_removed, self.n_removed - self.W.shape[1],
                    self.W.shape[1], dropped)

    # -- maps (xi is (size,) or (size, k)) -----------------------------------

    def _mu_project(self, x: np.ndarray) -> np.ndarray:
        out = np.array(x, dtype=float)
        for cells, V in zip(self.tract_cells, self.V):
            if V.shape[1]:
                out[cells] -= V @ (V.T @ x[cells])
        if self.W.shape[1]:
            out -= self.W @ (self.W.T @ x)
        return out

    def _unscale(self, x: np.ndarray) -> np.ndarray:
        return x / (self.scale if x.ndim == 1 else self.scale[:, None])

    def project(self, xi: np.ndarray, which: str = "all") -> np.ndarray:
        return self._mu_project(xi)

    def null(self, xi: np.ndarray) -> np.ndarray:
        return xi - self._mu_project(xi)

    def zeta(self, xi: np.ndarray) -> np.ndarray:
        return self._unscale(self._mu_project(xi))

    def zeta_T(self, g: np.ndarray) -> np.ndarray:
        return self._mu_project(self._unscale(g))

    def lambda_data(self, xi: np.ndarray) -> np.ndarray:
        z = self.zeta(xi)
        lam = z[: self.m][self.row_cell]
        rows = self.puma_of_row >= 0
        lam[rows] += z[self.m:][self.puma_of_row[rows]]
        return lam

    def lambda_data_T(self, g: np.ndarray) -> np.ndarray:
        return self.zeta_T(np.concatenate([self.collapse(g), self.puma_sum(g)], axis=0))

    def ridge_gradient(self, xi: np.ndarray) -> np.ndarray:
        return self.kappa * self.null(xi)

    def tract_group_basis(self) -> np.ndarray:
        return self.W

    global_basis = tract_group_basis

    def local_projector(self, rows: np.ndarray) -> np.ndarray:
        t = int(self.cell_area[rows[0]])
        if not np.array_equal(rows, self.tract_cells[t]):
            raise ValueError("local_projector expects one tract's cells, in order")
        return self._Q[t]

    def sigma(self, inputs: PMEDMInputs, alpha: float, taper: str | None,
              variance_floor) -> Sigma:
        if (alpha, taper, variance_floor) != self.settings:
            raise ValueError(f"built for alpha, taper, floor = {self.settings}, "
                             f"asked for {(alpha, taper, variance_floor)}")
        return CollapsedHierarchy.sigma(self, inputs, alpha, taper, variance_floor)

    def torch_maps(self, device: str):
        return _TorchNull(self, device)


def _intersect(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Orthonormal basis of span(A) & span(B), both orthonormal."""
    if not A.shape[1] or not B.shape[1]:
        return A[:, :0]
    Z = _null(np.hstack([A, -B]), 1e-9)
    return _orth(A @ Z[: A.shape[1]])


class _TorchNull:
    def __init__(self, h: NullSpaceHierarchy, device: str) -> None:
        import torch

        self.torch = torch
        self.m, self.n_puma, self.kappa = h.m, h.n_puma, h.kappa
        rows, cols, vals, start = [], [], [], 0
        for cells, V in zip(h.tract_cells, h.V):
            r = V.shape[1]
            if r:
                ii, jj = np.nonzero(np.abs(V) > 0)
                rows.append(cells[ii])
                cols.append(start + jj)
                vals.append(V[ii, jj])
                start += r
        if start:
            index = torch.as_tensor(np.vstack([np.concatenate(rows), np.concatenate(cols)]),
                                    device=device)
            self.V = torch.sparse_coo_tensor(index, torch.as_tensor(np.concatenate(vals),
                                                                    dtype=torch.float64, device=device),
                                             (h.size, start)).coalesce()
            self.Vt = self.V.t().coalesce()
        else:
            self.V = None
        self.W = torch.as_tensor(h.W, dtype=torch.float64, device=device)
        self.inv_scale = torch.as_tensor(1.0 / h.scale, dtype=torch.float64, device=device)
        self.row_cell = torch.as_tensor(h.row_cell, device=device)
        tract = np.flatnonzero(h.puma_of_row >= 0)
        self.tract_rows = torch.as_tensor(tract, device=device)
        self.tract_puma = torch.as_tensor(h.puma_of_row[tract], device=device)

    def _removed(self, xi):
        """``(Q + W W') xi`` for a ``(batch, size)`` tensor."""
        out = torch_zero = xi.new_zeros(xi.shape)
        if self.V is not None:
            out = out + self.torch.sparse.mm(self.V, self.torch.sparse.mm(self.Vt, xi.T)).T
        if self.W.shape[1]:
            out = out + (xi @ self.W) @ self.W.T
        del torch_zero
        return out

    def zeta(self, xi):
        return (xi - self._removed(xi)) * self.inv_scale

    def lambda_data(self, xi):
        z = self.zeta(xi)
        lam = z[:, : self.m][:, self.row_cell]
        if self.n_puma:
            lam = lam.index_add(1, self.tract_rows, z[:, self.m:][:, self.tract_puma])
        return lam

    def ridge(self, xi):
        r = self._removed(xi)
        return 0.5 * self.kappa * (r * r).sum(1)
