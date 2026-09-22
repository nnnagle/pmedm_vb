"""``Sigma`` as a diagonal plus a low-rank factor, never materialised.

The design covariance from 80 successive-differences replicates is a sum of 80
outer products, so it has rank at most 80 however many constraint cells there
are. A block-group run has thousands, which makes a dense ``Sigma`` both large
and singular: ``Sigma^-1`` does not exist and forming the matrix is wasted work
either way.

What is carried instead is convex shrinkage toward the diagonal, rearranged so
the diagonal is preserved exactly:

.. math::

    s_i = \\frac{4}{80}\\sum_r (y_{ir} - \\hat y_i)^2
    \\qquad
    D_i = v_i - (1 - \\alpha)\\, s_i
    \\qquad
    \\Sigma(\\alpha) = D + (1 - \\alpha)\\frac{4}{80} L L'

for ``v`` the published (or modelled) per-cell variance and ``L`` the
``(n, 80)`` deviations.

**Why ``D`` is a residual rather than a ridge.** Any ``D`` added *on top of* the
replicate covariance inflates every variance above what ACS publishes. That
diagonal is the one quantity in this pipeline with an independent published
answer -- recomputed MOEs match Census's own to rounding -- so buying
invertibility by corrupting it is a bad trade. Defining ``D`` as the residual
leaves the variances alone at every ``alpha`` and shrinks only the
correlations, which is the part 80 replicates estimate badly.

It also disarms a trap in the obvious formulation. Under a naive
``alpha * diag(v) + (1 - alpha) * Sigma_hat``, a zero count -- whose replicates
are identically its estimate -- comes out at ``alpha * w * k``: the modelled
zero-cell variance silently scaled down, making the cells we know least about
look the most certain. Here ``s_i`` is exactly zero for those cells, so
``D_i = v_i = w * k`` at any ``alpha``.

**The same thing written as variance and correlation.** With ``D = diag(v)``
and ``R`` the replicate correlation matrix, this is exactly

.. math::

    \Sigma(\alpha) = D^{1/2}\,[\,(1-\alpha) R + \alpha I\,]\,D^{1/2}

-- algebraically identical, verified to 3e-17. ``R`` here is literally
``L~ L~'`` for ``L~`` the row-normalised deviations: unit diagonal, cosines off
it. So ``alpha`` shrinks the *correlation* matrix toward the identity while the
variances stay exactly as published, which is the interpretation the residual
definition was chosen for. The factored form is stored instead only because it
is ``O(n * 80)`` where the correlation form is ``O(n^2)``.

**Why the middle factor cannot be ``R`` alone.** ``D^{1/2} R D^{1/2}`` has the
published variances on its diagonal and is still singular: congruence by an
invertible matrix preserves rank, so scaling rows and columns by ``sqrt(v)``
rescales eigenvalues without creating nonzero ones, and the rank stays that of
``L~``. The ``alpha I`` term is the only part of the expression that lifts the
remaining ``n - 80`` eigenvalues off zero, and it does so without disturbing
the diagonal, since ``(1 - alpha) * 1 + alpha = 1``.

A zero cell also has no correlation row to speak of: ``||L_i|| = 0`` makes
``L~_i`` a ``0/0``, so the correlation form needs a convention for it, whereas
here ``L_i = 0`` simply contributes nothing and ``D_i = v_i = w * k`` carries
the cell. At block group those are common rather than exceptional.

This is *not* an ``LDL'`` decomposition and cannot be made into one with ``D``
holding the variances. ``LDL'`` pivots are conditional variances,
``Var(cell j | cells 1..j-1)``, which fall strictly below the marginal variance
wherever cells are correlated, and which depend on the order the cells happen
to be in. ``diag(v)`` here is the marginal variance and is order-independent.

**Polarity.** ``alpha = 1`` is classic diagonal PMEDM -- the low-rank term is
identically zero, not merely small -- and ``alpha`` toward 0 approaches the full
design covariance:

===========  ==========  ==========  =========================
``alpha``    rank        diagonal    off-diagonal vs. the full
===========  ==========  ==========  =========================
1.00         full        exact       0 percent
0.50         full        exact       50 percent
0.10         full        exact       90 percent
0.01         full        exact       99 percent
0.00         <= 80       exact       100 percent
===========  ==========  ==========  =========================

The diagonal is exact throughout; that is the point of the residual
construction. What ``alpha`` trades is *correlation structure against
conditioning*, not against rank -- rank only collapses at exactly 0, where
``D_i = v_i - s_i = 0`` on every non-degenerate cell and the Woodbury identity
divides by it.

It is worth being clear about which object here is rank-deficient. The raw
design covariance ``(4/80) L L'`` has rank at most 80 however many cells there
are, so a ``Sigma`` carrying the full replicate structure and nothing else
cannot be inverted. ``D`` is what makes ``Sigma(alpha)`` full rank, and it is
strictly positive only for ``alpha > 0``. So 99 percent of the correlation
structure *and* full rank *and* the published diagonal are simultaneously
available; 100 percent of it is not.

**Why this shape, for the solvers.** The point of a diagonal-plus-low-rank
``Sigma`` is not that it is cheap on its own, but that it composes with the
rest of the dual Hessian. From ``pmedm_derivation.md`` that Hessian is
``X' diag(p) X - (X'p)(p'X) + c * Sigma`` for ``c = n / N^2``. Writing
``S = X' diag(p) X`` and ``u = X'p`` and substituting ``Sigma = D + BB'``:

.. math::

    H = (S + cD) \;+\; [\sqrt{c}B \;\; u]
        \begin{bmatrix} I_{80} & \\ & -1 \end{bmatrix}
        [\sqrt{c}B \;\; u]'

So the whole Hessian is *sparse plus diagonal*, corrected by rank 81: eighty
directions up from the replicates and one down from the outer product the
derivation calls the annoying part. A Newton step is then one factorisation of
``S + cD``, 81 solves against it, and an 81x81 dense solve -- verified against a
direct solve to 1e-16.

Two things follow. The ``cD`` term is what makes ``S + cD`` factorisable at
all: ``S`` alone can be rank-deficient, and a strictly positive diagonal is
exactly the regularisation it needs. And the cost of a *full* covariance is the
same kind of computation as a diagonal one -- the derivation already has to
handle a rank-1 dense downdate, and this makes it rank 81 rather than
introducing anything new.

This is also why an ``LDL'`` or Cholesky factor of ``Sigma`` would not help.
Such a factor is dense and ``(n, n)``, and ``Sigma`` is not the matrix being
inverted: the solver inverts ``H``, and only a low-rank representation survives
being added to ``S``.

For the VB path, the pieces the ELBO needs are here already --
:meth:`Sigma.logdet`, :meth:`Sigma.solve` for the quadratic form, and
:meth:`Sigma.draw`. Whether the variational family should itself be
diagonal-plus-low-rank is open; matching this structure would keep the KL term
in the same identities.

**Tapering to within-tract blocks.** ``groups`` optionally labels each row,
and the low-rank term is then kept only inside a group:
``Sigma = D + sum_g B_g B_g'``, with ``B_g`` the rows of ``B`` in group ``g``.
:meth:`~pmedm_vb.assemble.inputs.PMEDMInputs.sigma` labels rows by tract, and
this is the default there. ``groups=None`` is the untapered form above --
equivalently, one group holding every row -- and stays available so the two
can be compared.

The reason is ``experiments/covariance_structure.py``, run on B01001 at block
group for Knox County (2020-2024). Across block groups the RMS replicate
correlation was 0.1154 against 0.1118 expected from 80 replicates of pure noise
(ratio 1.03), and the signed mean was -0.0012 against a null standard error of
0.0064. The global ``L L'`` therefore spends its between-zone entries on noise,
which ``alpha`` shrinks but does not remove. The same run found within-zone
correlations differing between zones by 1.68 times the split-half null, which
is what ruled out a single pooled correlation matrix instead.

Three properties survive the mask. It is still a covariance: the block mask is
positive semi-definite, so its elementwise product with ``L L'`` is too (Schur
product theorem), and ``D > 0`` keeps the sum definite. The diagonal is
untouched, so ``v`` stays exact and ``alpha = 1`` is still classic PMEDM. And
since every operation here is already per-group, the untapered form is literally
the one-group case of the same code.

The mask is by tract rather than by block group because that is what was *not*
tested: one table at one level says nothing about correlation between tables,
or between a tract cell and the block-group cells inside it. Both come from the
same sample and are the likeliest to be real, and a tract block keeps them. It
also matches the Hessian: ``S`` is block diagonal by tract, so a tract-tapered
``S + cSigma`` is too, and the rank-81 correction above collapses to the rank-1
downdate alone.

``alpha`` is deliberately *not* stored on the assembled inputs. It is a
parameter of the run, and holding ``v`` and ``L`` raw lets one assembled problem
serve an entire sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from pmedm_vb.data.variance import N_REPLICATES, SDR_FACTOR


@dataclass(frozen=True)
class Sigma:
    """Error covariance over the stacked constraint vector.

    Attributes
    ----------
    v:
        ``(n,)`` published or modelled per-cell variances, strictly positive.
        This is :func:`pmedm_vb.data.variance.variances` with a zero-cell
        policy already applied.
    l:
        ``(n, 80)`` replicate deviations, from
        :func:`pmedm_vb.data.variance.deviations`. ``None`` means an exactly
        diagonal ``Sigma``, which is what ``alpha = 1`` also produces.
    alpha:
        Shrinkage toward the diagonal, in ``(0, 1]``.
    groups:
        ``(n,)`` integer labels. The low-rank term is kept only between rows
        sharing a label -- see *Tapering* in the module docstring. ``None``
        keeps it everywhere, the untapered form.

    Notes
    -----
    Row order must match the stacked constraint vector ``[vec(Y_T); vec(Y_B)]``
    column-major, the same order ``v`` is in. A permutation of one against the
    other is undetectable here and produces a converged solver with wrong
    uncertainties.
    """

    v: np.ndarray
    l: np.ndarray | None
    alpha: float
    groups: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(
                f"alpha must be in (0, 1], got {self.alpha}. At exactly 0 the "
                f"residual diagonal vanishes on every non-degenerate cell and "
                f"Woodbury divides by it"
            )
        if self.v.ndim != 1:
            raise ValueError(f"v must be one-dimensional, got shape {self.v.shape}")
        if self.l is not None and self.l.shape != (self.v.size, N_REPLICATES):
            raise ValueError(
                f"l must be {(self.v.size, N_REPLICATES)}, got {self.l.shape}"
            )
        if self.groups is not None:
            if self.groups.shape != self.v.shape:
                raise ValueError(
                    f"groups must be {self.v.shape}, got {self.groups.shape}"
                )
            if not np.issubdtype(self.groups.dtype, np.integer):
                raise ValueError(f"groups must be integer, got {self.groups.dtype}")
        if not np.all(self.v > 0):
            bad = int((self.v <= 0).sum())
            raise ValueError(
                f"v must be strictly positive; {bad} cell(s) are not. A zero "
                f"here means the zero-cell policy was not applied -- see "
                f"variances(..., policy='model')"
            )
        if not np.all(self.d > 0):
            bad = int((self.d <= 0).sum())
            raise ValueError(
                f"residual diagonal is non-positive on {bad} cell(s). v and l "
                f"describe different cells, or are ordered differently: for a "
                f"cell with replicate variance s, D = v - (1-alpha)*s is "
                f"positive only when v >= s"
            )

    # -- factors ---------------------------------------------------------

    @property
    def scale(self) -> float:
        """``(1 - alpha) * 4/80``, the multiplier on the low-rank term."""
        return (1.0 - self.alpha) * SDR_FACTOR

    @cached_property
    def d(self) -> np.ndarray:
        """Residual diagonal ``v - (1 - alpha) * s``, strictly positive.

        Tapering removes off-diagonal entries only, so this is the same with
        or without ``groups``.
        """
        if self.l is None or self.scale == 0.0:
            return self.v
        return self.v - self.scale * np.square(self.l).sum(axis=1)

    @cached_property
    def b(self) -> np.ndarray | None:
        """``sqrt(scale) * L``, so that ``Sigma = D + B B'`` before tapering."""
        if self.l is None or self.scale == 0.0:
            return None
        return np.sqrt(self.scale) * self.l

    @property
    def is_diagonal(self) -> bool:
        return self.b is None

    @property
    def is_tapered(self) -> bool:
        return self.groups is not None

    @property
    def global_factor(self) -> np.ndarray | None:
        """``B`` when it couples every row, else ``None``.

        What a solver has to carry as a dense low-rank correction. Tapered, the
        low-rank term lives inside the groups instead -- see
        :meth:`block_dense`.
        """
        return None if self.is_tapered else self.b

    @cached_property
    def members(self) -> list[np.ndarray]:
        """Row indices of each group, in label order; one group if untapered."""
        if self.groups is None:
            return [np.arange(self.v.size)]
        order = np.argsort(self.groups, kind="stable")
        _, starts = np.unique(self.groups[order], return_index=True)
        return np.split(order, starts[1:])

    # -- operations ------------------------------------------------------

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """``Sigma @ x``, in ``O(n * 80)``.

        The dual needs this rather than a solve: the objective carries
        ``0.5 * (n/N^2) * lambda' Sigma lambda``, the gradient
        ``(n/N^2) Sigma lambda``, and ``Sigma`` enters the Hessian additively.
        """
        out = _scale_rows(x, self.d)
        b = self.b
        if b is None:
            return out
        for rows in self.members:
            out[rows] += b[rows] @ (b[rows].T @ x[rows])
        return out

    def solve(self, x: np.ndarray) -> np.ndarray:
        """``Sigma^-1 @ x`` by Woodbury, in ``O(n * 80^2 + groups * 80^3)``.

        ``(D + BB')^-1 = D^-1 - D^-1 B (I + B' D^-1 B)^-1 B' D^-1``, one group
        at a time. Each capacitance matrix is 80x80 and symmetric positive
        definite, so it goes through a Cholesky factorisation rather than a
        general solve.
        """
        y = _scale_rows(x, 1.0 / self.d)
        b = self.b
        if b is None:
            return y
        out = y.copy()
        for rows, factor in zip(self.members, self._capacitance_factors):
            correction = b[rows] @ cho_solve(factor, b[rows].T @ y[rows])
            out[rows] -= _scale_rows(correction, 1.0 / self.d[rows])
        return out

    def logdet(self) -> float:
        """``log det Sigma`` by the matrix determinant lemma.

        ``det(D + BB') = det(I + B' D^-1 B) * det(D)`` per group, so this costs
        a Cholesky of each 80x80 capacitance matrix and never forms ``Sigma``.
        """
        total = float(np.log(self.d).sum())
        if self.b is None:
            return total
        for c, _ in self._capacitance_factors:
            total += 2.0 * float(np.log(np.diag(c)).sum())
        return total

    def diagonal(self) -> np.ndarray:
        """``diag(Sigma)``, which equals ``v`` exactly at every ``alpha``.

        The point of the residual construction, and cheap to assert against the
        published variances.
        """
        b = self.b
        if b is None:
            return self.d
        return self.d + np.square(b).sum(axis=1)

    def draw(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """Draw ``size`` vectors from ``N(0, Sigma)``, without factorising.

        ``D + BB'`` samples exactly as ``sqrt(D) * z1 + B @ z2`` for independent
        standard normals ``z1`` of length ``n`` and ``z2`` of length 80, since
        the covariance of that sum is ``D + BB'`` by construction. No square
        root of ``Sigma`` is needed, which is the point: a Cholesky factor of
        an ``(n, n)`` matrix is exactly what this representation exists to
        avoid, and the VB objective needs draws rather than a factor. Tapered,
        each group gets its own independent ``z2``, which is what zeroes the
        covariance between groups.

        Returns
        -------
        numpy.ndarray
            ``(n, size)``, one draw per column.
        """
        n = self.v.size
        out = np.sqrt(self.d)[:, None] * rng.standard_normal((n, size))
        b = self.b
        if b is None:
            return out
        for rows in self.members:
            out[rows] += b[rows] @ rng.standard_normal((b.shape[1], size))
        return out

    def block_dense(self, rows: np.ndarray) -> np.ndarray:
        """The dense ``(len(rows), len(rows))`` part of ``Sigma`` a solver adds
        to its own sparse blocks.

        Tapered, that is ``Sigma`` itself restricted to ``rows``. Untapered, it
        is ``D`` alone, and the low-rank term is left to :attr:`global_factor`:
        it couples every row, so it has no block to live in.
        """
        block = np.diag(self.d[rows])
        b = self.b
        if b is None or not self.is_tapered:
            return block
        same = self.groups[rows][:, None] == self.groups[rows][None, :]
        return block + same * (b[rows] @ b[rows].T)

    def to_dense(self) -> np.ndarray:
        """Materialise ``Sigma``. For checking against a direct solve only.

        ``(n, n)`` in memory, which for a block-group run is the thing this
        whole representation exists to avoid.
        """
        dense = np.diag(self.d)
        b = self.b
        if b is None:
            return dense
        for rows in self.members:
            dense[np.ix_(rows, rows)] += b[rows] @ b[rows].T
        return dense

    @cached_property
    def _capacitance_factors(self) -> list[tuple[np.ndarray, bool]]:
        """Cholesky of ``I + B_g' D_g^-1 B_g`` per group, the 80x80 matrices
        Woodbury inverts."""
        b, d = self.b, self.d
        factors = []
        for rows in self.members:
            capacitance = np.eye(b.shape[1]) + b[rows].T @ (b[rows] / d[rows, None])
            try:
                factors.append(cho_factor(capacitance))
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "capacitance matrix is not positive definite; Sigma is not "
                    "a valid covariance for these inputs"
                ) from error
        return factors


def _scale_rows(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """``diag(s) @ x`` for a vector or a matrix of column vectors."""
    return s[:, None] * x if x.ndim == 2 else s * x
