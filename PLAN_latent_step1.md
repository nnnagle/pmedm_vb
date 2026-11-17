# Plan, step 1: locating each PUMA in type space

> **Status: fixed.** Lower-level plan for step 1 of `PLAN_latent.md`
> (version 1), for the model in `MODEL_latent.qmd` (version 1). Change it
> only when the plan itself changes, and record each change below.
>
> | version | date | change |
> |---|---|---|
> | 1 | 2026-11-17 | approved as drafted; §9 items accepted as proposed |
> | 2 | 2026-11-17 | §4.6 and §6: the rare-group criterion is the RMS log ratio over (PUMA, fold) pairs, not over PUMAs with folds summed. Summed over folds, the own-records baseline ($\gamma = 0$) reproduces each PUMA's total almost exactly by construction (every record is in four training sets), so that version hides its fold noise. The summed version is still reported. |

## 1. Scope and traceability

Step 1 of the parent plan: pieces **A1, A2, A5, A6, B1, D1**. It decides
whether $h$ can locate a PUMA in type space (rare groups especially), the
cost and gain of statewide variety, and $\gamma$. It uses no tables to fit
anything; tables appear only as checks.

| parent item | where in this plan |
|---|---|
| A1 atoms | §4.1 |
| A2 contributions $T$ | §4.2 (built in full here; used for scoring now, for fitting from step 2) |
| A5 basis $h$, candidates fixed by "Adopted defaults" | §4.4 |
| A6 $d(P,i)$, $\gamma$ | §4.3 |
| B1 $\theta_P$ from the PUMA's records | §4.5 |
| D1 reproduction check | §4.6 |
| "decides": $h$, statewide variety, $\gamma$ | §6 decision rules |
| smaller note: group-quarters type in the check | §4.6 |
| constraints (no changes to `assemble/`, `solvers/`, `data/`; offline batch) | §7 |

Out of scope: B2, B3, C1–C4, $u_b$, $\tau$, block-group or tract tables.
No departure from the parent plan or the model.

## 2. Model quantities (from `MODEL_latent.qmd`, version 1)

* Atoms $i = 1..M$: $(z_i, x_i)$, statewide weights $q_i$.
* $q^P_i(\theta_P) = q_i\,\gamma^{d(P,i)} \exp\{\theta_P' h(z_i) -
  A_P(\theta_P)\}$.
* $\ell_P(\theta_P) = n_P\big[\sum_{i\in\mathcal P}\tilde w_i\,\theta_P'
  h(z_i) - A_P(\theta_P)\big]$.
* Prior $\theta_P \sim \mathcal N(0, \Omega)$, here
  $\Omega = \sigma_\theta^2 I$ on standardized features. $\sigma_\theta$ is
  a setting of the model's "weak prior", chosen in §6.

The fit maximizes $\ell_P(\theta_P) + \log \mathcal N(\theta_P; 0, \Omega)$.

## 3. Design: cross-fitted reproduction

Fitting $\theta_P$ on a PUMA's records and checking it against the same
records is circular. So each PUMA's records are split into **5 folds**
(random by unit, stratified by household vs group quarters):

* fit $\theta_P$ on 4 folds ($n_P$ = training records, weights renormalized
  within the training folds);
* score $q^P$ against the held-out fold.

**Baselines**, scored the same way:

| baseline | settings | role |
|---|---|---|
| own records (PUMS support) | $\gamma = 0$ | noise floor: held-out fold predicted from the training folds' own records |
| state average | $\theta = 0$, $\gamma = 1$ | how far the PUMA is from the state |
| geography only | $\theta = 0$, $\gamma \in \{0.25, 0.5\}$ | neighbours without types |

$\gamma = 0$ restricts the support to the PUMA's own records ($d = 0$
only). A tilted statewide reference that beats it on rare groups shows the
gain of variety; one worse beyond fold noise shows the cost.

## 4. Pieces

### 4.1 Atoms (A1)

* **Inputs:** statewide units, design and replicate weights, PUMA labels
  (`embed.puma_check.load_check_data`); $z$ from the `stage1_v2` $d = 16$
  run (`embed.puma_check.load_phi`).
* **Output:** an atoms object holding $z$ ($M \times 16$), $w$, replicate
  weights ($M \times 80$), PUMA, unit type.
* **Checks:** $M = 155{,}047$ units; $z$ design-weighted mean 0 and
  covariance $I$; weights sum to the PUMS population totals.

### 4.2 Contributions $T$ (A2)

* **Inputs:** the published-cell definitions used by `build_puma`
  (`assemble.allcells.constraint_tables` and the held-out sets, block-group
  and tract), applied to the statewide units.
* **Output:** sparse $T$ ($M \times J$), with cell names.
* **Check:** for PUMA 4701502, the rows of its own records equal
  `build_puma`'s $X$ exactly.
* **Use in step 1:** scoring only (§4.6). Fitting uses it from step 2.

### 4.3 Distances (A6)

* **Inputs:** 2020 PUMA boundaries (cached), the boundary-sharing adjacency
  rule in `experiments/latent/make_regions.py`. The function moves to
  `src/pmedm_vb/latent/geography.py`, and `make_regions.py` imports it
  (behaviour unchanged).
* **Output:** $d(P,Q)$, the number of adjacency steps between PUMAs
  (breadth-first search), and $d(P,i) = d(P, \mathrm{PUMA}_i)$.
* **Checks:** symmetric, zero diagonal, graph connected.

### 4.4 Bases (A5)

All candidates append the two group-quarters indicators. Every other
column is standardized over the statewide support (design-weighted).

| name | features | source |
|---|---|---|
| `dec7` | decoder category probabilities, 7 attributes, one category dropped per attribute | `embed.puma_check.decoder_features` |
| `dec37` | the same, all 37 attributes | same |
| `quad` | $z$, squares and cross-products | `embed.puma_check.quadratic_features` |
| `kern100`, `kern200` | Gaussian features $\exp\{-\lVert z - c_k\rVert^2 / 2\ell^2\}$ at 100 or 200 centres | new |

* **Kernel centres:** design-weighted k-means of $z$, fixed seed.
  $\ell$ = the median distance from atoms to their nearest centre.
* **Minimality:** drop columns that are constant or collinear on the
  support, so that $\mathrm{Cov}_q(h)$ is positive definite. Report $K$ and
  the condition number.

### 4.5 Fitting $\theta_P$ (B1)

* **Solver:** Newton with the exact Hessian
  $n_P\,\mathrm{Cov}_{q^P}(h) + \Omega^{-1}$ and a backtracking line
  search.
  - Stop when the largest gradient component is below $10^{-8}$ relative
    to the objective's scale, or after 100 iterations.
  - $\gamma = 0$ is handled as a support mask.
* **Output per fit:** $\hat\theta_P$, $q^P$, iterations, final gradient, Kish
  effective sample size of $q^P$, share of $q^P$ outside the PUMA.
* **Checks (before ISAAC):**
  - gradient against finite differences;
  - with $\sigma_\theta = \infty$, $\mathbb E_{q^P}[h]$ equals the training
    folds' weighted mean of $h$;
  - with $\gamma = 0$ and $\sigma_\theta = \infty$, $\hat\theta_P = 0$ and
    $q^P$ equals the PUMA's own normalized weights.
* **Cost:** about 10–30 iterations of $M \times K$ products per fit
  ($K \le 400$).

### 4.6 Reproduction check (D1)

All quantities are shares within the PUMA: predicted from $q^P$, observed
from the held-out fold.

* **Scored tables:**
  - the check tables: S1, the transfer set B;
  - the published-cell definitions through $T$;
  - rare groups: B03002 AIAN, NH/PI and other race; B17010C and E; ages
    80 and over; unit type (institutional and noninstitutional group
    quarters).
* **Metrics:**
  - $z = (\text{predicted} - \text{held-out}) / \mathrm{SE}$, with the SE
    from the held-out fold's replicate weights and the check's floor (one
    sampled case). Summarized as RMS $z$ and the share with $|z| \le 1$.
  - TVD and log-odds-ratio error for two-way tables
    (`embed.puma_check.score_tables`).
  - Rare groups: the ratio of predicted to held-out, pooled over PUMAs;
    the RMS log ratio over (PUMA, fold) pairs, each held-out fold scored
    on its own (the criterion in §6); and, reported beside it, the RMS log
    ratio across PUMAs with folds summed (`embed.puma_check.rare_groups`).
  - Quadrature: Kish effective sample size overall and within each rare
    group.
* **Secondary, optional:** $q^P$ fitted on all of a PUMA's records,
  against the published PUMA-level ACS tables (cached from the API), in
  MOE units.

## 5. Grid and compute

* $h \in$ {`dec7`, `dec37`, `quad`, `kern100`, `kern200`}
  × $\gamma \in \{0, 0.25, 0.5, 1\}$ × $\sigma_\theta \in \{1, 3, 10\}$.
* That's 60 settings × 58 PUMAs × 5 folds ≈ 17,400 fits, plus the
  baselines.
* Estimated at about 0.1–1 s per fit: at most about 5 CPU-hours, so one
  ISAAC CPU job with multiprocessing, within 24 h. The estimate is
  checked on the smoke run.
* Then a final fit on all of each PUMA's records at the chosen settings
  (§8).

## 6. Decision rules (fixed before results)

The thresholds are proposals to confirm before the run.

* **Choosing $h$:** at each $h$'s best $(\gamma, \sigma_\theta)$, rank by
  held-out RMS $z$ over all scored cells, and by the rare groups' RMS log
  ratio over (PUMA, fold) pairs. Choose the smallest $K$ within 10% of the
  best on both.
* **Choosing $\gamma$:** for the chosen $h$, the largest $\gamma$ whose RMS
  $z$ is within 10% of the best $\gamma$'s. This prefers variety at equal
  accuracy.
* **Choosing $\sigma_\theta$:** jointly with $\gamma$. If the best is at
  the edge of the grid, extend the grid.
* **Variety, reported:** gain and cost against $\gamma = 0$, as effective
  sample size and rare-group accuracy.
* **Stop rule:** if, for every $h$, held-out RMS $z$ exceeds the noise
  floor's by more than 50%, or the pooled ratio for AIAN or NH/PI is
  outside $[0.8, 1.25]$, step 1 fails. Revisit the stage 1 representation
  before step 2.

## 7. Code and constraints

* **New modules:** `src/pmedm_vb/latent/atoms.py` (A1, A2), `basis.py`
  (A5), `geography.py` (A6), `reference.py` (B1), `reference_check.py`
  (D1 scoring).
* **Experiment:** `experiments/latent/step1_reference.py`, `.sbatch`,
  `collect_step1.py`, and a grid config (JSON).
* **Unchanged:** `assemble/`, `solvers/`, existing `data/` modules, the
  PMEDM experiments. `embed/` is imported, not changed.
* **Batch:** runs offline (`PMEDM_VB_OFFLINE=1`).
* **Verification:** scripts in the scratchpad, not committed (the checks in
  §4.1–4.5, plus an end-to-end smoke run on two PUMAs with the smoke
  $\varphi$).

## 8. Outputs handed to step 2

* The chosen $h$ (builder, standardization parameters, kernel centres)
  and the chosen $\gamma$ and $\sigma_\theta$, frozen.
* $\hat\theta_P$ and $q^P$ for every PUMA, fitted on all its records.
* $T$ (A2) and the PUMA distances (A6).
* `summary.txt`: per setting, the metrics in §4.6, the baselines, and the
  decisions under §6.

## 9. Resolved at approval

1. The decision thresholds in §6, as proposed.
2. 5 folds, stratified by unit type.
3. The secondary check against published PUMA tables is implemented as an
   option and run when the cached tables are available.
