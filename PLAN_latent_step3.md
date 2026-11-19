# Plan, step 3: MAP on the regional run, against the standard PMEDM

> **Status: fixed.** Lower-level plan for step 3 of `PLAN_latent.md`
> (version 1), for the model in `MODEL_latent.qmd` (version 2), building on
> step 1 (`STEP1_latent.qmd`) and step 2 (`STEP2_latent.qmd`). Change it
> only when the plan itself changes, and record each change below.
>
> | version | date | change |
> |---|---|---|
> | 1 | 2026-11-18 | approved as drafted; §9 items accepted as proposed |
> | 2 | 2026-11-19 | §4.3: the collapsed posterior over $U$ has local optima (different optimizers stop at different objectives) and plain L-BFGS needs thousands of evaluations. Added a diagnostic run (§4.3a) that separates the two sources of non-convexity, a rule choosing the MAP method from it (§4.3b), and the MAP defined as the lowest exact objective over a set of starts. §5 compute updated. |
> | 3 | 2026-11-19 | §4.3a: Levenberg–Marquardt added to the diagnostic on both objectives (the problem is a penalized nonlinear regression, for which LM is the standard solver). §4.3b: the case "exact fits agree, fixed-$S$ fits do not" resolved (the exact fits decide); LM replaces plain L-BFGS in the chosen set when it reaches the best objective. |
>
> "Derived here" marks derivations made for this project, not taken from
> the literature.

## 1. Scope and traceability

Step 3 of the parent plan: **B3, C1, D2**. It decides how accurate the
model's MAP is against the standard PMEDM, with truth known.

| parent item | where in this plan |
|---|---|
| B3 MAP for $u_b$ and $\tau$ | §4.3 |
| C1 conditional PMEDM MAP (realized populations) | §4.4 |
| D2 regional run, adapted (region = PUMA's role, its PUMAs = block groups' role) | §3, §4.1, §4.5 |
| finding 2: $\tau$ not by joint MAP | §4.3 ($\tau$ on a grid, chosen by Laplace evidence) |
| finding 3: the existing solver can serve as C1 | §4.4 |
| finding 4 and the adopted default: leave-zone-out support | §3 |
| "decides": MAP accuracy against today's PMEDM | §6 |
| step 2's decision: exact collapsed likelihood | §4.3 |
| constraints | §7 |

Out of scope: posterior and coverage (step 4), samplers (step 5), block
groups (step 6). No departure from the parent plan or the model.

## 2. Model quantities (from `MODEL_latent.qmd`, version 2)

In the regional run, the region plays the PUMA's part and its PUMAs (zones)
play the block groups'.

* **Region reference** $q^A_i \propto q_i\,\gamma^{d(A,i)}\exp\{\theta_A'h(z_i)\}$
  on the statewide atoms, with step 1's settings ($h$ = `kern100`,
  $\gamma = 0.5$, $\sigma_\theta = 1$). $\theta_A$ is fitted on the region's
  own records by the step 1 pseudo-likelihood. $d(A,i)$ is the number of
  adjacency steps from atom $i$'s PUMA to the nearest PUMA of the region (0
  inside the region).
* **Zone superpopulation** $p_b \propto q^A \odot \exp\{u_b'h\}$, with
  $u_b \sim \mathcal N(0, \tau^2 I)$ iid within the region.
* **Realized population** $C_b \sim \mathrm{Multinomial}(N_b, p_b)$, with
  $N_b$ the zone's weighted units (known: it is truth).
* **Tables** $Y \mid C \sim \mathcal N(\mathcal A(C), \Sigma)$ with
  $\Sigma = \Sigma_{\mathrm{survey}} + \omega^2 I$, $\omega = N_A/(n_A s)$,
  $s = 3$, $N_A$ and $n_A$ the region's units and records.

## 3. Setup

* **Regions and zones:** the 13 regions of `experiments/latent/regions_tn.json`
  (3–6 contiguous PUMAs each), as in the earlier regional runs.
* **Fitted tables:** each zone's S1 cells from its own records
  (`latent.regions.build_region`), noise-free relative to truth.
* **$\Sigma_{\mathrm{survey}}$:** the replicate covariance across zones as
  `build_region` builds it (its one-case floor covers zero cells), with
  $\alpha = 0.01$, untapered. The standard PMEDM's zero-count floor is
  specific to published tables, so it does not apply here.
* **Support modes:**
  - **leave-zone-out (primary):** when fitting zone $b$, the zone's own
    records get zero base weight, in every variant. This mimics block
    groups, whose residents are mostly not in the PUMS. Allocation accuracy
    is undefined in this mode.
  - **with own records (secondary):** comparable with the earlier regional
    runs; allocation accuracy reported.
* $\theta_A$ uses all of the region's records in both modes, as a PUMA's
  reference uses all of its records.

## 4. Pieces

### 4.1 Region references

* Fit $\theta_A$ for each region (step 1's `reference.py`, with $d(A,i)$
  above).
* **Check:** the reproduction check of step 1 on the region's own records
  (pooled rare-group ratios), so a region's reference is known to be sound
  before zones are fitted.

### 4.2 Collapsed-likelihood inputs for regions

* Adapt step 2's engine (`collapsed.py`) to a region: each zone is its own
  tract with no shared tract rows ($D_b$ the zone's own rows); $L$ the
  replicate deviations across all zones (rank at most 80); $T$ restricted to
  the S1 cells over statewide atoms.
* **Check:** the engine's per-region $S$, solve, $\log\det$ and gradient
  against dense brute force for one region (as in step 2).

### 4.3 MAP for $U$ (B3)

The collapsed log posterior over $U$ is not convex, for two reasons
(derived here):

1. **The tilt and the tables use different features.** $p_b \propto
   q^A\exp\{u_b'h\}$ tilts on $h(z)$, while the tables measure $x$ through
   $T$, so the mean $m(U) = \sum_b N_b D_b T'p_b(U)$ is nonlinear in $U$.
2. **The covariance moves with $U$.** $S(U)$ includes the realization
   variance under $p_b(U)$, so $-\tfrac12\log\det S(U)$ and $S(U)^{-1}$ depend
   on $U$.

The only convex term, the prior on $U$ (precision 3–50), is small against
the data's curvature (eigenvalues up to about $4\times 10^5$). Version 1's
build found different optimizers stopping at different objectives (for
example 9,122 and 9,146 on one region at $0.5\,\tau_{\mathrm{ref}}$).

**In all cases:** the MAP for each $\tau \in \{0.5, 1,
2\}\times\tau_{\mathrm{ref}}$ ($\tau_{\mathrm{ref}} = 0.284$) is the
**lowest exact collapsed objective over a set of starts** (§4.3b), with
$\theta_A$ fixed at its records fit. The spread of the starts' final
objectives is reported per region and $\tau$.

#### 4.3a Diagnostic (run first)

* **Regions:** `knoxville_core` and `upper_cumberland`, leave-zone-out
  mode, the real $\varphi$.
* **For each $\tau$, ten fits:**
  - on the exact objective: plain L-BFGS with memory 10, plain L-BFGS with
    memory 20, L-BFGS preconditioned by a Gauss–Newton Hessian refreshed
    every 30 iterations, and plain L-BFGS (memory 10) started from the
    previous $\tau$'s solution (none for the first $\tau$);
  - the same four on the **fixed-$S$ objective**, with $S$ held at its value
    at $U = 0$ (this removes source 2);
  - **Levenberg–Marquardt from $U = 0$ on each objective.** Each step solves
    $(G + \tau^{-2}I + \mu I)\,\Delta = -\nabla F$, with $G = J'S^{-1}J$
    the mean-part Gauss–Newton matrix of step 2 ($S = S(0)$ for the fixed-$S$
    objective, $S(U)$ for the exact one) and $\nabla F$ the objective's own
    gradient. A step is accepted if it lowers that objective; $\mu$ is
    divided by 10 on acceptance and multiplied by 10 on rejection. Same stop
    rule as L-BFGS (largest gradient component at most $10^{-4}$), at most
    500 iterations. On the fixed-$S$ objective this is standard LM for
    penalized nonlinear least squares; on the exact objective it is a damped
    quasi-Newton method (it omits the covariance part of the Hessian).
* **Reported for each fit:** final objective (exact, and fixed-$S$ where
  relevant), largest gradient component, evaluations, time, and the
  distance between solutions ($\lVert\Delta U\rVert$ and the largest change
  in fitted zone shares on the scoring tables).
* The ladder and scoring are not run in the diagnostic.

#### 4.3b Choosing the MAP method (fixed before results)

Fits **agree** when their final objectives are within 1 log-likelihood unit
of each other, for every region and $\tau$.

* **If the fixed-$S$ fits agree and the exact fits do not,** the local optima
  come from source 2. The set of starts for the full run is: the fixed-$S$
  solution, then polished on the exact objective with plain L-BFGS (memory
  10); and plain L-BFGS (memory 10) on the exact objective from $U = 0$.
* **If the fixed-$S$ fits also disagree,** they come from source 1, which is
  structural. The set of starts is all four exact fits of §4.3a.
* **If the exact fits already agree** (whether or not the fixed-$S$ fits
  do), the set is plain L-BFGS (memory 10) from $U = 0$ and from the
  previous $\tau$'s solution. The exact objective defines the MAP, so its
  agreement decides.
* **LM:** if exact-objective LM reaches within 1 unit of the lowest exact
  objective of all fits, for every region and $\tau$, it replaces plain
  L-BFGS (memory 10) everywhere in the chosen set. On the fixed-$S$
  objective the same test decides whether LM replaces L-BFGS for the
  fixed-$S$ fit.

The chosen set is recorded with the diagnostic's numbers before the full
run. The code carries all three sets behind one option, so the full run
needs no code change.

#### 4.3c Choosing $\tau$ without truth

* The Laplace approximation of the marginal likelihood $p(Y \mid \tau)$ at
  each MAP, with a Gauss–Newton Hessian (the mean-part information of step 2
  plus $\tau^{-2}I$). The evidence-chosen $\tau$ is the primary; the best
  $\tau$ by truth is reported beside it, as a check on the rule.
* **Checks:** gradient at the optimum (largest component); MAP objective
  not above its value at $U = 0$.

### 4.4 Conditional PMEDM (C1)

Given $p_b$ at the MAP, the realized composition's MAP over statewide atoms,
jointly over the region's zones:

$$
\min_{\{r_b\}} \sum_b N_b\, \mathrm{KL}(r_b \Vert p_b)
  + \tfrac12 e'\Sigma^{-1}e,
\qquad e = \textstyle\sum_b N_b D_b T'r_b - Y.
$$

* Via the existing dual solver where it applies (parent finding 3), with
  zone shares fixed at $N_b/N_A$ since $N_b$ is known; or a dedicated dual
  Newton solver if that is simpler. Either way, verified as below.
* **Checks:**
  - duality gap and KKT conditions at the solution;
  - on a small case, against a direct solve of the primal;
  - with base $p_b = $ the region's own design distribution and weight $n_A$,
    it reproduces the standard PMEDM's solution (a plumbing check).

### 4.5 Variants and scoring (D2)

A ladder, so each difference has one cause:

| variant | support and base | entropy weight | zone deviations |
|---|---|---|---|
| `pmedm_own` | standard PMEDM: region's own records, design weights | $n$ (standard) | per-cell multipliers |
| `pmedm_ref` | standard PMEDM on statewide atoms with base $q^A$ | $n$ | per-cell multipliers |
| `latent_u0` | conditional PMEDM, base $p_b = q^A$ | $N_b$ | none ($U = 0$) |
| `latent` | conditional PMEDM, base $p_b$ at the MAP | $N_b$ | $u_b$, each $\tau$ |
| `latent_n` | as `latent` at the chosen $\tau$ | $n_A$ | $u_b$ |

* `pmedm_own` → `pmedm_ref` isolates the support; `pmedm_ref` →
  `latent_u0` the entropy weight; `latent_u0` → `latent` the zone
  deviations. `latent_n` is the parent plan's sensitivity check on the
  weight (model fork 7).
* The two standard-PMEDM variants use the standard settings: $\alpha = 0.01$,
  untapered, multiplier prior sd 3 (`hierarchy="none+prior3"`; a region has
  no PUMA or tract rows).
* **Scored on** the realized MAP $\hat r_b$ for every variant, and on
  $p_b$ itself for `latent_u0` and `latent`, with the regional run's
  scoring (`latent.scoring.score_region`):
  - per-zone joint structure on the scoring tables (TVD, log-odds-ratio
    error);
  - rare groups: pooled ratio and spread across zones (RMS log ratio);
  - misfit on the fitted cells in SE units;
  - transfer: fitted on S1, scored on transfer set B in SE units
    (`latent.regions.transfer_sets`);
  - allocation accuracy (with-own-records mode only).

## 5. Compute

CPU on ISAAC.

* **Diagnostic (§4.3a):** 2 regions × 3 values of $\tau$ × 8 fits, run in
  parallel; about 1–1.5 hours on one 16-core node. Plain L-BFGS needs about
  1,500–8,700 evaluations per $\tau$ at about 0.3 s each for a 4-zone region.
* **Full run:** per region, $\theta_A$ in seconds, the $U$ MAP over the
  chosen set of starts, and the conditional PMEDM over (zones × 155,047)
  pairs in seconds. 13 regions, 2 support modes and the ladder: about
  3–6 hours on one 16-core node, within one 24-hour job.

**GPU** is not needed in step 3. It matters at block-group scale (step 6).
There, step 2's 3–5 s per evaluation suggests the MAP is about 15 minutes per
PUMA on CPU (some 200 evaluations). The decision is left to step 6, using
step 3's measured evaluation counts (larger than step 2 assumed).

## 6. Decision rules (fixed before results)

Thresholds are proposals to confirm at review.

* **Primary comparison:** `latent` at the evidence-chosen $\tau$ against
  `pmedm_own`, leave-zone-out mode, over all zones of all regions.
* **Go to step 4** if `latent` is:
  - no more than 5% worse than `pmedm_own` on mean log-odds-ratio error and
    on mean TVD; and
  - no more than 10% worse on rare-group spread (RMS log ratio, pooled over
    the rare groups).
* **Improvement** is reported for any of those metrics, or transfer
  misfit, where `latent` is at least 5% better.
* **Otherwise stop and review,** using the ladder to locate the loss
  (support, weight, or zone deviations).
* **Reported, not decided:** the with-own-records mode; the truth-best
  $\tau$ against the evidence-chosen one; `latent_n`; scores of $p_b$.

## 7. Code and constraints

* **New modules:**
  - `src/pmedm_vb/latent/region_reference.py` (§4.1 and the leave-zone-out
    bases);
  - `region_engine.py` (§4.2, the adapter to `collapsed.py`);
  - `map_fit.py` (§4.3);
  - `conditional.py` (§4.4).
* **Experiment:** `experiments/latent/step3_regional.py`, `.sbatch`,
  `collect_step3.py`.
* **Unchanged:** `assemble/`, `solvers/`, existing `data/` modules, `embed/`,
  the step 1 and step 2 code, and the earlier regional experiment code.
  Imported, not changed.
* **Batch:** runs offline (`PMEDM_VB_OFFLINE=1`).
* **Verification:** scripts in the scratchpad, not committed (the checks in
  §4.1–4.4, plus an end-to-end smoke run on two regions with the smoke
  $\varphi$).

## 8. Outputs handed to step 4

* The decision under §6, and the ladder's scores.
* The fitted $\theta_A$, $\hat U$ per $\tau$, the evidence and the chosen
  $\tau$ per region; the realized MAPs $\hat r_b$.
* The MAP machinery (`map_fit.py`, `conditional.py`), which step 4's Laplace
  approximation and Gaussian conditioning start from.
* `summary.txt`: the ladder in both modes, by metric and region; $\tau$
  selection; timings and evaluation counts.

## 9. Resolved at approval (as proposed)

1. The thresholds in §6: 5% on joint structure, 10% on rare groups.
2. Leave-zone-out as the primary mode.
3. The ladder in §4.5, including `pmedm_ref`.
4. Choosing $\tau$ by Laplace evidence with a Gauss–Newton Hessian, and
   $\theta_A$ held fixed at its records fit.
