# Plan, step 2: the collapsed-likelihood engine and the information diagnostic

> **Status: fixed.** Lower-level plan for step 2 of `PLAN_latent.md`
> (version 1), for the model in `MODEL_latent.qmd` (version 1), building on
> step 1 (`STEP1_latent.qmd`; run of record `latent/step1_sigma`). Change it
> only when the plan itself changes, and record each change below.
>
> | version | date | change |
> |---|---|---|
> | 1 | 2026-11-18 | approved as drafted; §9 items accepted as proposed |
>
> "Derived here" marks derivations made for this project, not taken from
> the literature.

## 1. Scope and traceability

Step 2 of the parent plan: **B2 prototype and the information diagnostic**.
It decides whether $S$ is computable at PUMA scale, and whether the tables
identify $u_b$. It fits no parameters: everything is evaluated at the
step 1 reference, $\theta_b = \hat\theta_P$ ($u_b = 0$), or at test points
for verification.

| parent item | where in this plan |
|---|---|
| B2 collapsed-likelihood engine ($m$, $S$, $S^{-1}$, $\log\det S$, gradient) | §4.2 |
| information diagnostic (finding 1) | §4.3 |
| "decides": $S$ computable; tables identify $u_b$ | §6 decision rules |
| A3 $Y$, $\Sigma$, $D_b$ (first used here) | §4.1 |
| A4 $N_b$ (first used here) | §4.1 |
| constraints | §7 |

Out of scope: fitting $u_b$ or $\tau$ (B3, step 3), the conditional PMEDM
(C1), any sampler. No departure from the parent plan or the model.

**PUMAs:** 4701502 first, then the four test PUMAs 4703000, 4702300,
4702900 and 4701900.

## 2. Model quantities (from `MODEL_latent.qmd`, version 1)

* Block superpopulation $p_{bi} = q_i\gamma^{d(P,i)}
  \exp\{(\theta_P + u_b)'h(z_i) - A_P(\theta_P + u_b)\}$, with $h$ =
  `kern100`, $\gamma = 0.5$ and $\hat\theta_P$ from step 1.
* Collapsed likelihood $Y \mid \theta_P, U \dot\sim \mathcal N(m, S)$:
  $$
  m = \sum_b N_b D_b T' p_b, \qquad
  S = \Sigma + \sum_b N_b D_b T'(\mathrm{diag}\,p_b - p_b p_b')T D_b'.
  $$
* Prior $u_b \mid \tau \sim \mathcal N(0, \tau^2 I_K)$.

$T$ is restricted to the constraint cells (block-group and tract), the
cells that enter $Y$.

## 3. Analysis

**$S$ is block diagonal by tract** (derived here, from the existing code's
structure).

* $\Sigma$ is used as in the paper baseline: `PMEDMInputs.sigma(alpha,
  taper="tract")`. The replicate covariance is kept only within tracts, so
  $\Sigma$ is block diagonal by tract (`assemble/sigma.py`, "Tapering").
* $D_b$ maps block $b$'s cells to its own block-group rows and its tract's
  rows only. So the realization term is block diagonal by tract too.
* Hence $S = \mathrm{blockdiag}_t(S_t)$, with $S_t$ covering one tract's rows
  and its block groups' rows: about $J_T + n_t J_B$ rows (for 4701502,
  $J_T = 489$, $J_B = 184$, so roughly $10^3$ per tract).
* $S^{-1}$ and $\log\det S$ are then exact per-tract dense Cholesky
  factorizations, about $10^9$ flops per tract, and the collapsed likelihood
  is a sum over tracts. B2 needs no low-rank approximation.

**The gradient of $\log\det S$ reduces to per-atom quadratic forms**
(derived here). With $G_b = D_b' S_t^{-1} D_b$ and $\mu_b = T'p_b$,

$$
\frac{\partial}{\partial u_b}\,\tfrac12\log\det S
= \tfrac12 N_b\, \mathrm{Cov}_{p_b}\!\big(h,\; t' G_b t - 2\, t' G_b \mu_b\big),
$$

a covariance between $h$ and one scalar per atom. Its cost is
$\sum_i \mathrm{nnz}(t_i)^2$ per block, since $t_i$ is sparse. This is the
expensive part of B2, and §4.2 measures it.

**The information diagnostic** (derived here). At $u = 0$, every block in
the PUMA has $p_b = q^P$. With $C = \mathrm{Cov}_{q^P}(t, h)$ ($J \times K$),
the Jacobian of $m$ with respect to $u_b$ is $N_b D_b C$. The mean part of
the Fisher information is

$$
I_{bb'} = N_b N_{b'}\, C' D_b' S_t^{-1} D_{b'}\, C
\quad (b, b' \text{ in the same tract; zero otherwise}).
$$

* It omits the information from $S$'s dependence on $u$, so it is a lower
  bound on the full Fisher information (both parts are positive
  semi-definite).
* Against the prior precision $\tau^{-2}$, the **effective number of
  identified directions** for block $b$ is
  $$
  \mathrm{df}_b(\tau) = \mathrm{tr}\big[I_{bb}(I_{bb} + \tau^{-2} I)^{-1}\big]
  \in [0, K].
  $$
* For a functional of the block's type mix, $f = \mathbb E_{p_b}[g]$ (for
  example a rare-group share), $\partial f/\partial u_b = a =
  \mathrm{Cov}_{q^P}(g, h)$. The **share of its prior variance removed by
  the tables** is
  $$
  1 - \frac{a'(I_{bb} + \tau^{-2}I)^{-1}a}{\tau^2\, a'a}.
  $$

**A scale for $\tau$.** No value of $\tau$ is fixed yet. Step 1's
$\hat\theta_P$ across the 58 PUMAs gives a data-based scale for how much type
mixes differ between areas:
$\tau_{\mathrm{ref}} = \big(\tfrac1K\sum_k \mathrm{Var}_P(\hat\theta_{P,k})\big)^{1/2}$.
The diagnostic is reported at $\tau \in \{0.5, 1, 2\}\times\tau_{\mathrm{ref}}$.

## 4. Pieces

### 4.1 Area inputs (A3, A4)

* **$Y$, $\Sigma$, nesting:** from `build_puma` for the PUMA's constraint
  tables (by import). $\Sigma$ = `sigma(alpha, taper="tract")` with the paper
  baseline's $\alpha$.
* **$T$ for the area:** step 1's statewide $T$, restricted to the
  constraint cells, with its columns mapped one to one onto `build_puma`'s
  $X_T$ and $X_B$ columns. Step 1 showed the rows agree exactly for
  4701502's records.
* **$D_b$:** from `build_puma`'s aggregation ($A_B$ identity for block
  groups, $A_T$ for tracts).
* **$N_b$** (known, per the model; working rule for this step):
  - households: the block group's published occupied-unit total;
  - group-quarters persons: the tract's published group-quarters total,
    split across its block groups in proportion to their occupied units,
    unless `build_puma` already carries block-group unit totals, in which
    case those are used.
* **Checks:** column mapping one to one; $\sum_b N_b$ against the PUMA's
  $N$ from `build_puma`.

### 4.2 Collapsed-likelihood engine (B2)

* **Evaluate** $p_b$ for every block, $m$, the per-tract blocks $S_t$, their
  Cholesky factors, the log-likelihood
  $-\tfrac12(Y - m)'S^{-1}(Y - m) - \tfrac12\log\det S$, and its gradient
  with respect to $(\theta_P, U)$: the mean part through $C_b$ and the
  $\log\det$ part through §3's quadratic forms.
* **Atoms:** statewide; per block, one pass of $M \times K$ for $p_b$ and of
  the sparse $T$ for $\mu_b$, $V_b$ and the quadratic forms.
* **Timings to report,** per PUMA: building $p$, $m$ and $S$; factorization;
  the log-likelihood; the gradient without the $\log\det$ term; the
  $\log\det$ term; peak memory.
* **Checks (before ISAAC):**
  - per-tract assembly of $S$ against a dense brute-force build on a small
    set of tracts;
  - solve and $\log\det$ against dense NumPy;
  - the full gradient, including the $\log\det$ term, against finite
    differences at a random $(\theta_P, U)$ near the reference;
  - $I$ from §3 against the outer product of finite-differenced $m$.

### 4.3 Information diagnostic

At $u = 0$ for each PUMA:

* $I_{bb}$ per block group and $I_t$ jointly per tract.
* Per block group, at each $\tau$: $\mathrm{df}_b(\tau)$, the leading
  eigenvalues of $I_{bb}$, and the number above $\tau^{-2}$. Per tract: the
  joint df divided by the number of block groups, to show what tract rows
  add.
* Share of prior variance removed for these functionals:
  - rare groups: AIAN, NH/PI and NH Asian shares; group quarters by type;
  - common: owner share, age 65+ share, households below poverty.
* The diagonal of $S$ split into survey $\Sigma$ and realization variance, by
  table and level. This checks the parent plan's expectation that survey
  error dominates by roughly the mean weight.

## 5. Compute

All per PUMA, CPU on ISAAC. The diagnostic needs one pass over the atoms
for $C$, plus per-tract factorizations: minutes. The engine timings need a
few evaluations of the log-likelihood and gradient: from seconds to tens of
minutes depending on the $\log\det$ term. One job of 5 PUMAs within a few
hours; to be checked on the smoke run.

## 6. Decision rules (fixed before results)

Thresholds are proposals to confirm at review.

* **Computability** (which collapsed likelihood step 3 uses):
  - if one full log-likelihood and gradient evaluation, including the
    $\log\det$ term, takes at most 60 s for a PUMA on one ISAAC CPU node:
    exact collapsed likelihood in step 3;
  - otherwise step 3 holds $S$ fixed at its value at the current parameters
    within each outer iteration and refreshes it between iterations, and
    this approximation is recorded.
* **Identification of $u_b$,** by the median over block groups of
  $\mathrm{df}_b(\tau_{\mathrm{ref}})$ across the 5 PUMAs:
  - at least 5: well identified; step 3 as planned.
  - from 1 to 5: weakly identified; step 3 proceeds and adds the
    $u_b = 0$ variant (blocks share $\theta_P$) as an explicit comparison.
  - below 1: not identified. Blocks collapse to $\theta_P$ and differ only
    through their realized populations. Stop and review the model before
    step 3.
* **Reported, not decided:** variance reduction for the functionals in
  §4.3, and the survey vs realization split of $S$.

## 7. Code and constraints

* **New modules:** `src/pmedm_vb/latent/area.py` (A3, A4 and the $T$
  mapping), `collapsed.py` (B2), `information.py` (diagnostic).
* **Experiment:** `experiments/latent/step2_engine.py`, `.sbatch`,
  `collect_step2.py`.
* **Unchanged:** `assemble/`, `solvers/`, existing `data/` modules, `embed/`,
  the step 1 code. Imported, not changed.
* **Batch:** runs offline (`PMEDM_VB_OFFLINE=1`).
* **Verification:** scripts in the scratchpad, not committed. Locally,
  `build_puma`'s summary-file inputs are not cached, so the engine's math is
  verified with the real $T$ and stubbed $Y$ and $\Sigma$ (as in step 1);
  the real inputs are first used on ISAAC.

## 8. Outputs handed to step 3

* The decisions under §6: exact or fixed-$S$ collapsed likelihood, and the
  identification category for $u_b$.
* The engine (`collapsed.py`) and the area inputs (`area.py`).
* $\tau_{\mathrm{ref}}$ and the diagnostic tables, as a starting scale for
  the $\tau$ grid in step 3.
* `summary.txt`: timings, the diagnostic per PUMA and block group, the
  variance reductions, the split of $S$, and the decisions.

## 9. Resolved at approval

1. The thresholds in §6, as proposed: 60 s per evaluation; df bands at 1
   and 5.
2. The working rule for $N_b$ in §4.1.
3. The $\tau$ scale from step 1's spread of $\hat\theta_P$, and the grid
   $\{0.5, 1, 2\}\times\tau_{\mathrm{ref}}$.
4. The functionals in §4.3.
