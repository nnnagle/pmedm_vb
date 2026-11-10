"""Cell-by-cell Poisson rows for the sparse cells: the ``+poisson<c>`` level suffix.

Step 3 of the sparse-cells plan. The cells resting on about one sampled case
leave the Gaussian hierarchy of :mod:`pmedm_vb.assemble.hierarchy` and become
rows of their own with a count-shaped error; what remains keeps the hierarchy.

**Which cells.** At the finest published level -- block group, or tract for a
category constrained at tract only -- every cell with ``k_hat = Y^2 / v < c``
(``v`` the published variance, ``sigma_v``), published zeros included, is
pulled out. A positive cell no support unit can carry (``q > 0`` and ``X > 0``
nowhere in its area) is *not* pulled out: its Poisson likelihood would be
``-inf`` at every allocation (the dual unbounded below), so it stays Gaussian
(counted as ``kept_unsupported`` by :meth:`PoissonHierarchy.describe`).
``c = 0`` pulls out nothing (a zero's ``k_hat`` is 0, not below 0).

**Gaussian part [derived from the hierarchy's construction].** Per tract and
category the remaining block groups ``R`` keep the hierarchy:

- if nothing was pulled out, the tract row and its block group contrasts,
  exactly as the ``puma`` (or ``tract``) level has them;
- if some were, a *remaining total* row whose loading is the sum of ``R``'s
  block group rows, target ``sum_R Y_b`` and replicate deviations ``sum_R l_b``
  (block group replicates sum exactly to the tract's, so this is the tract's
  replicates less the pulled-out block groups'), with variance
  ``SDR ||sum_R l_b||^2 + sum_R max(v_b - s_b, 0)`` (the replicate variance
  plus the members' non-replicate parts, as :mod:`~pmedm_vb.assemble.collapse`
  does for merged cells -- an assumption, not a published variance), and
  contrasts among ``R`` when ``|R| >= 2`` (one remaining block group *is* the
  remaining total);
- no tract row if nothing remains.

The remaining totals of a category sum to zero across the PUMA, and its PUMA
row's loading, target and replicates are their sums (variance from the summed
replicates, or the sum of the totals' variances when those have no spread, as
in the plain hierarchy). For a tract-only category the remaining tracts are the
"remaining totals". ``Sigma`` is built from these rows' original replicate
deviations, with ``alpha`` shrinkage and the run's variance floor as before,
and the prior's ``tau^2`` on the diagonal (:mod:`pmedm_vb.assemble.prior`).

**Poisson rows.** Each pulled-out positive cell is its own row, with its own
free multiplier (in no sum-to-zero group) on its full tilt: the multipliers the
data see are ``lambda_data = C' zeta`` for the 0/1 loading ``C`` of every row
(below), so a pulled-out block group's records see that row's multiplier and
nothing else for that constraint. Zeros are merged, coarsest first:

- a category zero across the PUMA is one row on the PUMA total;
- a zero tract (all its block groups zero) is one row on the tract;
- otherwise a tract's zero block groups of a category are one row on their sum;
- for a tract-only category, its zero tracts are one row on their sum (the
  same rule one level up -- a decision, as ``experiments/cell_cases.py``
  counts them);
- a zero row no support unit carries is dropped: its allocation is 0 whatever
  the multipliers, and it would only add a coordinate the data do not see.

**The error term [derived].** With ``t = n zeta / N`` (the convention of
:mod:`~pmedm_vb.assemble.tweedie`), a row enters the dual as
``(1/n) K(t)`` in place of the Gaussian ``Y t + v t^2 / 2``, with

.. math::

    K(t) = -\\hat k \\log(1 - w t), \\qquad t < 1 / w, \\qquad w = v / Y,

the cumulant generating function of ``w`` times a ``Poisson``-sized count
seen ``k_hat`` times: ``K'(0) = k_hat w = Y`` and ``K''(0) = k_hat w^2 = v``,
so it matches the cell's published estimate and variance. Its conjugate is
``T / w - k_hat log T`` up to a constant: the Poisson log-likelihood of
``k_hat`` cases of size ``w`` as a function of the true count ``T``. A zero row
has ``k_hat = 0``: ``K = 0`` below the wall ``t = 1/w`` and ``+inf`` from it,
whose conjugate is the linear penalty ``T / w`` (the Poisson probability of no
case, ``exp(-T / w)``). With a shared ``w`` linear penalties add, so merging
zeros leaves the MAP unchanged; the separate rows would add free multipliers
the data do not inform.

**``w`` for a zero row [assumed; decision D3 of the plan].** The pooled
``sum(v) / sum(Y)`` of the same tract and category's pulled-out positive block
groups (``"pooled"``; for a tract-only category, of its pulled-out positive
tracts), if any; otherwise ``N / n`` times the design-weighted mean count the
category's carrying units hold, ``sum_i q_i X_ik / sum_i q_i [X_ik > 0]`` over
support units with ``q_i = sum_zones q``. For a household-universe table that
mean is exactly 1, so ``w = N / n`` (``"N/n"``); for a person-universe table it
is the persons in the category per carrying unit (``"N/n*persons"``). Positive
rows use ``"v/Y"``. Each row's rule is in :attr:`PoissonHierarchy.w_rule`.

**The prior.** The Gaussian prior of ``+prior<sd>`` must also hold the Poisson
rows' multipliers. Their Gaussian ``Sigma`` term is replaced by ``K``, so the
prior part ``zeta^2 / (2 n s^2)`` -- ``(c/2) tau^2 zeta^2``, ``c = n / N^2`` --
is kept explicitly: these rows sit in ``Sigma`` with no replicate deviations
and the term subtracts only ``y zeta + (c/2) v0 zeta^2``.

**The wall and the transform.** Each Poisson row's multiplier is bounded above
by ``b = N / (n w)``. HMC and VB sample ``u`` with ``zeta = b - exp(u)``, the
upper-bounded transform of the Stan Reference Manual, and its log-Jacobian
``u`` is added to the log density, so no trajectory reaches a wall. Under
``exp(-n f)`` that is ``-(1/n) sum log(b - zeta)`` added to ``f``: **the
objective of this level, as every solver sees it, includes that barrier**, and
is the transformed target written in ``zeta`` [derived]. Then:

- the MAP is the mode of the transformed target. ``f`` is convex and the
  barrier is convex, so it is strictly convex in ``zeta`` and Newton (in
  ``zeta``, with the backtracking line search that refuses the wall) finds its
  unique minimiser; that point is the unique stationary point in ``u`` too,
  the map being a diffeomorphism. A zero row's mode is finite there, at
  ``b - zeta = 1 / (n s_r)`` without a prior, ``s_r`` the row's share of the
  allocation (its gradient condition). Without the barrier a zero row's MAP is
  *on* the wall (its dual term is decreasing up to it), which is why it is
  included;
- the Laplace approximation in ``u`` is ``D H D`` at that mode, ``H`` the
  ``zeta``-Hessian of the objective above and ``D = dzeta/du = -exp(u)`` on the
  Poisson rows (1 elsewhere): exact at a stationary point, where the
  gradient term of the change of variables vanishes;
- :class:`~pmedm_vb.solvers.vb._DualTarget` maps its argument through the
  transform, so HMC (:mod:`~pmedm_vb.solvers.mcmc`) and VB both work in
  ``u`` -- VB's Gaussian is then a Gaussian in ``u``, as ADVI's would be.

Saved MAP results keep ``xi`` in the linear coordinates and ``lam`` in today's
stacked layout; HMC traces keep ``lam`` (the multipliers the data see, today's
layout) and ``xi`` in the sampled coordinates (``u`` on the Poisson rows).
Because the loading is ``C' zeta``, a tract row's ``lam`` is 0 where its
category's remaining total is carried by the remaining block group rows, and a
pulled-out block group's ``lam`` is its own row's multiplier.

**Not modelled.** Household clustering beyond what ``w = v/Y`` absorbs; the
cross-table correlation of a sparse cell (one household filling several
tables' cells: those rows are now independent); unequal weights inside a cell.

It travels with the hierarchy level: ``"puma+prior3+poisson2"``. The ``tract``
and ``puma`` levels only; not with a collapse, roll-up, ``+decor``,
``+tweedie`` or ``+k0w``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp

from pmedm_vb.assemble.sigma import Sigma
from pmedm_vb.data.variance import SDR_FACTOR

SEPARATOR = "+poisson"


def split_poisson(level: str) -> tuple[str, float | None]:
    """``"puma+prior3+poisson2"`` -> ``("puma+prior3", 2.0)``; none -> ``(level, None)``.
    Another suffix after it is kept."""
    if SEPARATOR not in level:
        return level, None
    head, _, rest = level.partition(SEPARATOR)
    spec, plus, tail = rest.partition("+")
    try:
        c = float(spec)
    except ValueError:
        raise ValueError(f"Poisson rows are +poisson<c>, e.g. +poisson2; got {level!r}") from None
    if not c >= 0:
        raise ValueError(f"the Poisson threshold must be non-negative, got {level!r}")
    return head + (plus + tail if plus else ""), c


def poisson_level(level: str, c: float | None) -> str:
    """The level string with ``+poisson<c>`` appended, for a CLI's ``--poisson-k``."""
    if c is None:
        return level
    if not c > 0:
        raise ValueError(f"--poisson-k must be positive, got {c}")
    return f"{level}{SEPARATOR}{c:g}"


def sampled_to_xi(hierarchy, x):
    """The solver's linear coordinates ``xi`` from the sampled ones (``u`` on the
    Poisson rows); ``x`` itself for any other level. Rows on axis 0."""
    term = getattr(hierarchy, "tweedie", None)
    return term.from_sampled(x) if isinstance(term, PoissonRows) else x


def _table(name: str) -> str:
    return name.split(".", 1)[0]


@dataclass
class PoissonHierarchy:
    """The rows of a ``+poisson<c>`` level and the maps between coordinates.

    Rows ``0..m-1`` are *local* (each loads on one tract's stacked rows: the
    tract block of the MAP's Hessian), rows ``m..size-1`` *global* (the PUMA
    rows, zero PUMA rows and merged zero tracts of a tract-only category).
    ``C`` (``size x m_full``, 0/1) is every row's loading on today's stacked
    rows; ``lambda_data = C' zeta``. ``group`` labels the sum-to-zero groups of
    the local rows (block group contrasts, local; remaining totals across the
    PUMA, global), ``bg_group`` the local ones alone. ``kind`` and ``w_rule``
    say what each row is and, for a Poisson row, where its ``w`` came from.
    """

    level: str
    base: str
    c: float
    m: int
    m_full: int
    C: sp.csr_matrix
    kind: np.ndarray
    area: np.ndarray            # taper group: the tract of a local row, n_tracts for global rows
    group: np.ndarray
    bg_group: np.ndarray
    Y: np.ndarray
    single: np.ndarray          # Gaussian rows that are one stacked row: that row's own v
    puma_members: list          # per global Gaussian row, its remaining-total rows
    poisson: np.ndarray         # row indices of the Poisson rows
    k_hat: np.ndarray           # per Poisson row
    w: np.ndarray               # per Poisson row
    w_rule: np.ndarray          # per Poisson row
    v0: np.ndarray              # per row: the placeholder v the Poisson rows sit in Sigma with
    kappa: float
    n_tracts: int
    N: float
    n: int
    tract_cells: list
    notes: dict = field(default_factory=dict)
    v_published: np.ndarray | None = None

    is_trivial = False
    is_collapsed = True         # the MAP's CollapsedHessian handles a general 0/1 loading
    prior_sd = None
    prior_var = 0.0
    tweedie = None
    decor = None
    decor_rows = None

    # -- construction ------------------------------------------------------

    @classmethod
    def build(cls, inputs, base: str, c: float) -> "PoissonHierarchy":
        from pmedm_vb.solvers.base import ConstraintOperator

        if base not in ("tract", "puma"):
            raise ValueError(f"+poisson<c> needs the tract or puma level, got {base!r}")
        if inputs.sigma_l is None:
            raise ValueError("Poisson rows need the replicate deviations, and sigma_l is None")
        inputs.constraint_tracts()  # raises unless A_B is the identity
        n_tracts, c_t = inputs.Y_T.shape
        n_zones, c_b = inputs.Y_B.shape
        split, m_full = inputs.Y_T.size, inputs.n_constraints
        zone_tract = inputs.zone_tracts()
        t_names, b_names = list(inputs.tract_constraints), list(inputs.bg_constraints)
        b_index = {name: k for k, name in enumerate(b_names)}
        y = inputs.targets().astype(float)
        v = np.asarray(inputs.sigma_v, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            k_hat = np.where(v > 0, y * y / v, np.inf)
        supported = ConstraintOperator(inputs).forward((inputs.q > 0).astype(float)) > 0
        nN = inputs.N / inputs.n
        unit_weight = np.asarray(inputs.q, dtype=float).sum(axis=0)

        def persons(X, k: int) -> float:
            """Design-weighted mean of column ``k`` over the support units carrying it."""
            col = np.asarray(sp.csc_matrix(X)[:, k].todense()).ravel()
            carry = (col > 0) & (unit_weight > 0)
            if not carry.any():
                return np.nan
            return float((unit_weight[carry] * col[carry]).sum() / unit_weight[carry].sum())

        def fallback_w(X, k: int) -> tuple[float, str]:
            mean = persons(X, k)
            if not np.isfinite(mean):
                return nN, "N/n"
            return nN * mean, "N/n" if abs(mean - 1.0) < 1e-12 else "N/n*persons"

        rows: list[dict] = []           # local rows
        global_rows: list[dict] = []
        notes = dict(kept_unsupported=0, dropped_zero_rows=0, dropped_zero_cells=0)

        def pulled_positive(s: int) -> bool:
            if not (y[s] > 0 and k_hat[s] < c):
                return False
            if not supported[s]:
                notes["kept_unsupported"] += 1
                return False
            return True

        def zero_row(target: list, cols, kind: str, tract: int, w_rule) -> None:
            cols = np.asarray(cols, dtype=int)
            if not supported[cols].any():
                notes["dropped_zero_rows"] += 1
                notes["dropped_zero_cells"] += int(cols.size)
                return
            w, rule = w_rule
            target.append(dict(kind=kind, tract=tract, cols=cols, Y=0.0, k=0.0, w=w, rule=rule,
                               v0=float(v[cols].max())))

        def poisson_row(s: int, kind: str, tract: int) -> None:
            rows.append(dict(kind=kind, tract=tract, cols=np.array([s]), Y=y[s], k=k_hat[s],
                             w=v[s] / y[s], rule="v/Y", v0=v[s]))

        for kt, name in enumerate(t_names):
            t_rows = kt * n_tracts + np.arange(n_tracts)
            kb = b_index.get(name)
            b_rows = None if kb is None else split + kb * n_zones + np.arange(n_zones)
            X_t = inputs.X_T
            if c > 0 and (y[t_rows] == 0).all() and (b_rows is None or (y[b_rows] == 0).all()):
                zero_row(global_rows, t_rows, "zero_puma", n_tracts, fallback_w(X_t, kt))
                continue
            totals = []                 # this category's remaining-total rows
            if kb is None:              # tract-only: the tracts are the finest cells
                pos = [t for t in range(n_tracts) if pulled_positive(t_rows[t])]
                zeros = [t for t in range(n_tracts) if c > 0 and y[t_rows[t]] == 0]
                for t in pos:
                    poisson_row(t_rows[t], "poisson_tract", t)
                if zeros:
                    pv, py = v[t_rows[pos]].sum(), y[t_rows[pos]].sum()
                    rule = (pv / py, "pooled") if pos else fallback_w(X_t, kt)
                    cols = t_rows[zeros]
                    if len(zeros) == 1:
                        zero_row(rows, cols, "zero_tract", zeros[0], rule)
                    else:
                        zero_row(global_rows, cols, "zero_tracts", n_tracts, rule)
                for t in sorted(set(range(n_tracts)) - set(pos) - set(zeros)):
                    totals.append(len(rows))
                    rows.append(dict(kind="tract", tract=t, cols=np.array([t_rows[t]]), Y=y[t_rows[t]],
                                     single=True, tgroup=kt))
            else:
                for t in range(n_tracts):
                    zones = np.flatnonzero(zone_tract == t)
                    bz = b_rows[zones]
                    if c > 0 and y[t_rows[t]] == 0 and (y[bz] == 0).all():
                        zero_row(rows, [t_rows[t]], "zero_tract", t, fallback_w(X_t, kt))
                        continue
                    pos = [s for s in bz if pulled_positive(s)]
                    zeros = [s for s in bz if c > 0 and y[s] == 0]
                    for s in pos:
                        poisson_row(s, "poisson_bg", t)
                    if zeros:
                        rule = ((v[pos].sum() / y[pos].sum(), "pooled") if pos
                                else fallback_w(inputs.X_B, kb))
                        zero_row(rows, zeros, "zero_bg", t, rule)
                    keep = np.array(sorted(set(bz) - set(pos) - set(zeros)), dtype=int)
                    if keep.size == bz.size:      # nothing pulled out: the plain hierarchy
                        totals.append(len(rows))
                        rows.append(dict(kind="tract", tract=t, cols=np.array([t_rows[t]]),
                                         Y=y[t_rows[t]], single=True, tgroup=kt))
                        for s in bz:
                            rows.append(dict(kind="bg", tract=t, cols=np.array([s]), Y=y[s],
                                             single=True, bgroup=(kt, t)))
                    elif keep.size:
                        totals.append(len(rows))
                        rows.append(dict(kind="remaining", tract=t, cols=keep, Y=y[keep].sum(),
                                         single=keep.size == 1, tgroup=kt))
                        if keep.size >= 2:
                            for s in keep:
                                rows.append(dict(kind="bg", tract=t, cols=np.array([s]), Y=y[s],
                                                 single=True, bgroup=(kt, t)))
            if base == "puma" and totals:
                cols = np.concatenate([rows[r]["cols"] for r in totals])
                global_rows.insert(sum(1 for g in global_rows if g["kind"] == "puma"),
                                   dict(kind="puma", tract=n_tracts, cols=cols,
                                        Y=float(y[cols].sum()), members=totals))
        # Block-group-only categories: no tract row to keep a hierarchy for.
        for kb, name in enumerate(b_names):
            if name in t_names:
                continue
            b_rows = split + kb * n_zones + np.arange(n_zones)
            for t in range(n_tracts):
                bz = b_rows[zone_tract == t]
                pos = [s for s in bz if pulled_positive(s)]
                zeros = [s for s in bz if c > 0 and y[s] == 0]
                for s in pos:
                    poisson_row(s, "poisson_bg", t)
                if zeros:
                    rule = ((v[pos].sum() / y[pos].sum(), "pooled") if pos
                            else fallback_w(inputs.X_B, kb))
                    zero_row(rows, zeros, "zero_bg", t, rule)
                for s in sorted(set(bz) - set(pos) - set(zeros)):
                    rows.append(dict(kind="bg", tract=t, cols=np.array([s]), Y=y[s], single=True))

        m = len(rows)
        every = rows + global_rows
        size = len(every)
        r_idx = np.concatenate([np.full(r["cols"].size, i) for i, r in enumerate(every)])
        c_idx = np.concatenate([r["cols"] for r in every])
        C = sp.csr_matrix((np.ones(r_idx.size), (r_idx, c_idx)), shape=(size, m_full))
        if C.data.max(initial=0) > 1:
            raise AssertionError("a row loads twice on one stacked row")
        # Sum-to-zero groups over the local rows.
        group = np.full(m, -1)
        labels: dict = {}
        for i, r in enumerate(rows):
            key = ("b", r["bgroup"]) if "bgroup" in r else ("t", r["tgroup"]) if (
                "tgroup" in r and base == "puma") else None
            if key is not None:
                group[i] = labels.setdefault(key, len(labels))
        bg_group = np.where([("bgroup" in r) for r in rows], group, -1) if m else group.copy()
        is_poisson = np.array(["k" in r for r in every])
        poisson = np.flatnonzero(is_poisson)
        area = np.array([r["tract"] for r in every])
        self = cls(
            level=f"{base}{SEPARATOR}{c:g}", base=base, c=float(c), m=m, m_full=m_full, C=C,
            kind=np.array([r["kind"] for r in every]), area=area, group=group, bg_group=bg_group,
            Y=np.array([float(r["Y"]) for r in every]),
            single=np.array([bool(r.get("single", False)) for r in every]),
            puma_members=[r.get("members") for r in global_rows],
            poisson=poisson,
            k_hat=np.array([float(every[i]["k"]) for i in poisson]),
            w=np.array([float(every[i]["w"]) for i in poisson]),
            w_rule=np.array([every[i]["rule"] for i in poisson]),
            v0=np.array([float(r.get("v0", np.nan)) for r in every]),
            kappa=1.0 / inputs.n, n_tracts=n_tracts, N=float(inputs.N), n=int(inputs.n),
            tract_cells=[np.flatnonzero(area[:m] == t) for t in range(n_tracts)], notes=notes)
        self.v_published = self.row_variances(inputs, v)[0]
        return self

    def poisson_term(self) -> "PoissonRows":
        rows = self.poisson
        return PoissonRows(rows=rows, k_hat=self.k_hat, w=self.w, y=self.Y[rows] / self.N,
                           v0=self.v0[rows], n=self.n, N=self.N)

    def describe(self) -> str:
        """Row counts by kind, the Poisson rows by ``w`` rule, and what was left out."""
        kinds, counts = np.unique(self.kind, return_counts=True)
        rules, rule_counts = np.unique(self.w_rule, return_counts=True)
        text = (f"{self.level}: {self.size:,} rows ({self.m:,} local, {self.n_puma:,} global) from "
                f"{self.m_full:,} stacked rows; " + ", ".join(f"{k} {n}" for k, n in zip(kinds, counts))
                + "; Poisson w by rule: " + (", ".join(f"{r} {n}" for r, n in zip(rules, rule_counts))
                                             or "none")
                + f"; positive cells kept Gaussian for want of support {self.notes['kept_unsupported']}"
                + f"; zero rows dropped for want of support {self.notes['dropped_zero_rows']} "
                f"({self.notes['dropped_zero_cells']} cells)")
        return text

    # -- sizes -------------------------------------------------------------

    @property
    def n_puma(self) -> int:
        return self.size - self.m

    @property
    def size(self) -> int:
        return self.C.shape[0]

    @property
    def y_ext(self) -> np.ndarray:
        return self.Y / self.N

    @property
    def targets_ext(self) -> np.ndarray:
        return self.Y

    @property
    def v_ext(self) -> np.ndarray:
        """Per row, its variance from the published (unfloored) ones, as
        :meth:`row_variances` builds it; for a zero row the largest of its
        members' modelled zero variances."""
        return self.v_published

    def _counts(self, group: np.ndarray) -> np.ndarray:
        return np.bincount(group[group >= 0], minlength=int(group.max()) + 1 if group.max() >= 0 else 0)

    # -- numpy maps (vectors are (rows,) or (rows, k)) ---------------------

    def project(self, theta: np.ndarray, which: str = "all") -> np.ndarray:
        from pmedm_vb.assemble.hierarchy import _group_means

        group = self.group if which == "all" else self.bg_group
        if group.max(initial=-1) < 0:
            return np.array(theta, dtype=float)
        return theta - _group_means(theta, group, self._counts(group))

    def null(self, theta: np.ndarray) -> np.ndarray:
        return theta - self.project(theta)

    def zeta(self, xi: np.ndarray) -> np.ndarray:
        return np.concatenate([self.project(xi[: self.m]), xi[self.m:]], axis=0)

    zeta_T = zeta

    def loading(self, z: np.ndarray) -> np.ndarray:
        """``C' z``: row values onto today's stacked rows, without the projection."""
        return np.asarray(self.C.T @ z)

    def lambda_data(self, xi: np.ndarray) -> np.ndarray:
        return self.loading(self.zeta(xi))

    def lambda_data_T(self, g: np.ndarray) -> np.ndarray:
        return self.zeta_T(np.asarray(self.C @ g))

    def extend(self, stacked: np.ndarray) -> np.ndarray:
        """Today's stacked values summed into the rows: ``C g``."""
        return np.asarray(self.C @ stacked)

    def cell_loading(self, cells: np.ndarray, full_rows: np.ndarray) -> np.ndarray:
        """``C`` restricted to ``cells`` and ``full_rows`` (one tract's block), dense."""
        return self.C[cells][:, full_rows].toarray()

    def ridge_gradient(self, xi: np.ndarray) -> np.ndarray:
        out = np.zeros_like(xi, dtype=float)
        out[: self.m] = self.kappa * self.null(xi[: self.m])
        return out

    def tract_group_basis(self) -> np.ndarray:
        """``(size, g)``: unit-norm indicators of the groups spanning tracts
        (each category's remaining totals)."""
        local = set(np.unique(self.bg_group[self.bg_group >= 0]).tolist())
        labels = [g for g in np.unique(self.group[self.group >= 0]) if g not in local]
        U = np.zeros((self.size, len(labels)))
        for j, label in enumerate(labels):
            members = np.flatnonzero(self.group == label)
            U[members, j] = 1.0 / np.sqrt(members.size)
        return U

    def local_projector(self, rows: np.ndarray) -> np.ndarray:
        g = self.bg_group[rows]
        Q = np.zeros((rows.size, rows.size))
        for label in np.unique(g[g >= 0]):
            members = np.flatnonzero(g == label)
            Q[np.ix_(members, members)] = 1.0 / members.size
        return Q

    # -- Sigma -------------------------------------------------------------

    def sigma(self, inputs, alpha: float, taper: str | None,
              variance_floor: str | float | None) -> Sigma:
        """``Sigma`` over every row: a Gaussian row's replicate deviations are
        ``C l`` (its members' summed); its variance is its stacked row's (floored)
        one if it is one row, else the replicate variance plus its members'
        non-replicate parts (a PUMA row: its summed replicates' variance, or its
        remaining totals' summed variances without spread). A Poisson row has no
        deviations and the placeholder ``v0`` its term subtracts again. Taper
        groups: each local row's tract, the global rows one more. With a prior,
        its ``prior_var`` on the diagonal."""
        from pmedm_vb.assemble.prior import with_prior

        if taper not in ("tract", None):
            raise ValueError(f"taper must be 'tract' or None, got {taper!r}")
        v, l = self.row_variances(inputs, inputs.floored_variances(variance_floor))
        groups = self.area.astype(int) if taper == "tract" else None
        return with_prior(Sigma(v=v, l=l, alpha=alpha, groups=groups), self.prior_var)

    def row_variances(self, inputs, stacked_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(v, l)`` per row from the stacked rows' variances ``stacked_v`` (see :meth:`sigma`)."""
        L = np.asarray(inputs.sigma_l, dtype=float)
        l = np.asarray(self.C @ L)
        s_row = SDR_FACTOR * np.square(L).sum(axis=1)
        excess = np.asarray(self.C @ np.maximum(stacked_v - s_row, 0.0))
        s = SDR_FACTOR * np.square(l).sum(axis=1)
        v = s + excess
        single = np.flatnonzero(self.single)
        v[single] = np.asarray(self.C[single] @ stacked_v)
        for j, members in enumerate(self.puma_members):
            if members is None:
                continue
            r = self.m + j
            v[r] = s[r] if s[r] > 0 else float(v[members].sum())
        p = self.poisson
        v[p] = self.v0[p]
        l[p] = 0.0
        return v, l

    def torch_maps(self, device: str):
        return _TorchPoisson(self, device)


@dataclass
class PoissonRows:
    """The Poisson rows' term in the dual, as the change from the Gaussian one.

    ``rows`` index ``zeta`` (and ``xi``: the rows are free); per row ``k_hat``,
    ``w``, ``y = Y / N`` and ``v0``, the placeholder variance they sit in
    ``Sigma`` with. Per row, with ``t = n zeta / N`` and ``b = N / (n w)``:

    ``(1/n) K(t) - y zeta - (c/2) v0 zeta^2 - (barrier/n) log(b - zeta)``,

    ``inf`` from the wall. The last term is the log-Jacobian of
    ``zeta = b - exp(u)`` (module docstring); ``barrier=False`` gives the dual
    alone, for checks. Held in the hierarchy's ``tweedie`` slot, which every
    solver already evaluates.
    """

    rows: np.ndarray
    k_hat: np.ndarray
    w: np.ndarray
    y: np.ndarray
    v0: np.ndarray
    n: int
    N: float
    barrier: bool = True

    @property
    def c(self) -> float:
        return self.n / self.N**2

    @property
    def b(self) -> np.ndarray:
        """Each row's wall in ``zeta``: ``t = n zeta / N < 1 / w``."""
        return self.N / (self.n * self.w)

    # -- the transform -------------------------------------------------------

    def to_sampled(self, xi: np.ndarray) -> np.ndarray:
        """``u = log(b - zeta)`` on the rows (axis 0), ``xi`` elsewhere."""
        out = np.array(xi, dtype=float)
        b = self.b.reshape((-1,) + (1,) * (out.ndim - 1))
        out[self.rows] = np.log(b - out[self.rows])
        return out

    def from_sampled(self, x: np.ndarray) -> np.ndarray:
        """``zeta = b - exp(u)`` on the rows (axis 0)."""
        out = np.array(x, dtype=float)
        b = self.b.reshape((-1,) + (1,) * (out.ndim - 1))
        out[self.rows] = b - np.exp(out[self.rows])
        return out

    def jacobian_diag(self, xi: np.ndarray) -> np.ndarray:
        """``dxi/dx`` at ``xi``: ``-(b - zeta)`` on the rows, 1 elsewhere."""
        d = np.ones(xi.size)
        d[self.rows] = -(self.b - xi[self.rows])
        return d

    # -- numpy, for the MAP ----------------------------------------------

    def _gap(self, zeta: np.ndarray) -> np.ndarray:
        return self.b - zeta[self.rows]

    def feasible(self, zeta: np.ndarray) -> bool:
        return bool(np.all(self._gap(zeta) > 0))

    def value(self, zeta: np.ndarray) -> float:
        if not self.rows.size:
            return 0.0
        gap = self._gap(zeta)
        if np.any(gap <= 0):
            return np.inf
        z = zeta[self.rows]
        one_minus = self.n * self.w * gap / self.N         # 1 - w t
        k = -np.sum(self.k_hat * np.log(one_minus)) / self.n
        gaussian = self.y @ z + 0.5 * self.c * np.sum(self.v0 * z * z)
        barrier = np.sum(np.log(gap)) / self.n if self.barrier else 0.0
        return float(k - gaussian - barrier)

    def gradient(self, zeta: np.ndarray) -> np.ndarray:
        """In ``zeta`` (full length): ``K'(t)/N - y - c v0 zeta + barrier/(n (b - zeta))``."""
        out = np.zeros(zeta.size)
        if self.rows.size:
            gap = self._gap(zeta)
            z = zeta[self.rows]
            out[self.rows] = (self.k_hat / (self.n * gap) - self.y - self.c * self.v0 * z
                              + (1.0 / (self.n * gap) if self.barrier else 0.0))
        return out

    def curvature(self, zeta: np.ndarray) -> np.ndarray:
        """The rows' second derivative over ``c``, the variance in place of ``v0``:
        ``K''(t) + barrier N^2 / (n^2 (b - zeta)^2)``, with ``K'' = k w^2 / (1 - w t)^2``."""
        gap = self._gap(zeta)
        return (self.k_hat + (1.0 if self.barrier else 0.0)) * (self.N / (self.n * gap)) ** 2

    def sigma_at(self, sigma, zeta: np.ndarray):
        """``sigma`` with these rows' ``v`` replaced by :meth:`curvature`: the
        Hessian at ``zeta`` is then the Gaussian one built from it (the rows have
        no replicate deviations, so ``D`` there is ``v`` plus the prior)."""
        if not self.rows.size:
            return sigma
        v = np.array(sigma.v, dtype=float)
        v[self.rows] = self.curvature(zeta)
        return replace(sigma, v=v)

    # -- torch, for VB and HMC --------------------------------------------

    def torch_term(self, device: str, dtype) -> "TorchPoissonRows":
        import torch

        def tensor(x):
            return torch.as_tensor(np.asarray(x, dtype=float), dtype=dtype, device=device)

        return TorchPoissonRows(rows=torch.as_tensor(self.rows, device=device),
                                k_hat=tensor(self.k_hat), b=tensor(self.b), y=tensor(self.y),
                                v0=tensor(self.v0), w=tensor(self.w), n=self.n, N=self.N,
                                barrier=self.barrier)


@dataclass
class TorchPoissonRows:
    """:class:`PoissonRows` on ``(batch, rows)`` tensors, and the transform."""

    rows: object
    k_hat: object
    b: object
    y: object
    v0: object
    w: object
    n: int
    N: float
    barrier: bool = True

    def to_xi(self, x):
        """``zeta = b - exp(u)`` on the rows of ``(batch, size)`` sampled points."""
        if not self.rows.numel():
            return x
        out = x.clone()
        out[:, self.rows] = self.b - out[:, self.rows].exp()
        return out

    def feasible(self, zeta):
        import torch

        if not self.rows.numel():
            return torch.ones(zeta.shape[0], dtype=torch.bool, device=zeta.device)
        return (zeta[:, self.rows] < self.b).all(1)

    def __call__(self, zeta, extend: bool = False):
        """Per draw, the change from the Gaussian term; ``inf`` from the wall.
        ``extend`` is ignored: in the sampled coordinates the wall is never met."""
        import torch

        if not self.rows.numel():
            return torch.zeros(zeta.shape[0], dtype=zeta.dtype, device=zeta.device)
        z = zeta[:, self.rows]
        gap = self.b - z
        ok = (gap > 0).all(1)
        safe = torch.where(gap > 0, gap, torch.ones_like(gap))
        log_gap = torch.log(safe)
        one_minus = self.n * self.w * safe / self.N
        c = self.n / self.N**2
        k = -(self.k_hat * torch.log(one_minus)).sum(1) / self.n
        gaussian = z @ self.y + 0.5 * c * (self.v0 * z * z).sum(1)
        value = k - gaussian - (log_gap.sum(1) / self.n if self.barrier else 0.0)
        return torch.where(ok, value, torch.full_like(value, float("inf")))


class _TorchPoisson:
    def __init__(self, h: PoissonHierarchy, device: str) -> None:
        import torch

        from pmedm_vb.assemble.hierarchy import _TorchHierarchy

        self.torch = torch
        self.m, self.n_puma, self.kappa, self.m_full = h.m, h.n_puma, h.kappa, h.m_full
        self._proj = _TorchHierarchy.__new__(_TorchHierarchy)
        grouped = np.flatnonzero(h.group >= 0)
        self._proj.torch = torch
        self._proj.grouped = torch.as_tensor(grouped, device=device)
        self._proj.group = torch.as_tensor(h.group[grouped], device=device)
        counts = h._counts(h.group) if grouped.size else np.zeros(0)
        self._proj.counts = torch.as_tensor(counts, dtype=torch.float64, device=device)
        coo = h.C.tocoo()
        self.load_rows = torch.as_tensor(coo.row, device=device)
        self.load_cols = torch.as_tensor(coo.col, device=device)

    def project(self, theta):
        from pmedm_vb.assemble.hierarchy import _TorchHierarchy

        return _TorchHierarchy.project(self._proj, theta)

    def zeta(self, xi):
        return self.torch.cat([self.project(xi[:, : self.m]), xi[:, self.m:]], dim=1)

    def lambda_data(self, xi):
        zeta = self.zeta(xi)
        out = zeta.new_zeros(zeta.shape[0], self.m_full)
        return out.index_add(1, self.load_cols, zeta[:, self.load_rows])

    def ridge(self, xi):
        theta = xi[:, : self.m]
        null = theta - self.project(theta)
        return 0.5 * self.kappa * (null * null).sum(1)
