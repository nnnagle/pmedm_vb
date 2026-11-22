# Plan: solving the latent-population PMEDM

> **Status: fixed.** This is the agreed work plan for the model in
> `MODEL_latent.qmd`. Change it only when the plan itself changes, and
> record each change below. Lower-level plans for individual steps live in
> their own files and refer to the step numbers here.
>
> | version | date | change |
> |---|---|---|
> | 1 | 2026-11-17 | initial plan |
> | 2 | 2026-11-20 | step 3 failed its decision rule; step 3b added as its review: the basis of the zone deviations (`PLAN_latent_step3b.md`), deciding whether they stay before step 4 |
> | 3 | 2026-11-22 | step 4b added (`PLAN_latent_step4b.md`): the Dirichlet-multinomial realized level of model version 4, with the PUMA total known and block totals random; run if step 4 finds the posterior over-confident |
>
> "Derived here" marks derivations made for this project, not taken from
> the literature.

## Pieces

### A. Inputs

| piece | what | status |
|---|---|---|
| A1 atoms | statewide units (households and group-quarters persons) with $z_i$ (stage 1, $d = 16$), $x_i$, $w_i$ | exists (`stage1_v2` run) |
| A2 contributions $T$ | $t(x_i)$ for every block-group and tract cell, over statewide atoms | table definitions exist in `assemble/`; a statewide build is needed, by import only |
| A3 $Y$, $\Sigma$, $D_b$ | published block-group and tract tables, replicate covariance, nesting | exists (`build_puma`, `Sigma`) |
| A4 $N_b$ | units per block group (households plus group-quarters persons) | derivable from the tables; treated as known |
| A5 basis $h$ | decoder category probabilities, quadratic in $z$, kernel features on $z$ | decoder and quadratic exist; kernel is new |
| A6 $d(P,i)$, $\gamma$ | distances between PUMAs from adjacency | adjacency exists (`make_regions.py`) |

### B. Parameters

* **B1. $\theta_P$ from the PUMA's records.** A concave, $K$-dimensional
  Newton fit: tilted statewide atoms match the PUMA's weighted mean of $h$.
  No tables. Carries the reproduction check, so it is also the test bench
  for $h$ and $\gamma$.
* **B2. Collapsed-likelihood engine.** Builds $m(\theta)$ and $S(\theta)$
  and applies $S^{-1}$ and $\log\det S$. $S$ has one row per published
  block-group and tract cell (order $10^4$ per PUMA): diagonal plus survey
  low rank (80 replicates) plus one $J\times J$ term per area, with tract
  rows coupling block groups. The per-area term
  $N_b T'\mathrm{diag}(p_b)T$ costs about $\sum_i \mathrm{nnz}(t_i)^2$ per
  area. The hardest computational piece; the gradient of $\log\det S$ is
  the hardest part of it.
* **B3. MAP for $u_b$ and $\tau$.** About $B\times K$ parameters, L-BFGS
  using B2. The gradient of $m$ needs $\mathrm{Cov}_{p_b}(h,t)$ per area.

### C. Realized populations

* **C1. Conditional PMEDM MAP** per tract over statewide atoms, base $p_b$,
  weight $N_b$. About 15 million pairs per PUMA: GPU path.
* **C2. Stage 2 Gaussian conditioning.** Closed form with B2's solver;
  batched solves for many functionals.
* **C3. Samplers for $C$.** Swap and gradient-informed swap: incremental
  residual updates, parallel by tract.
* **C4. Samplers for the parameters.** Collapsed HMC, or Gibbs with C3.

### D. Validation

* **D1. Reproduction check**, inside B1.
* **D2. Regional run**, adapted: the region plays the PUMA, its zones
  (PUMAs) play the block groups.
* **D3. Block groups**: PUMA 4701502 against today's PMEDM on held-out
  tables.

## Order

Each step is chosen by what it decides. Steps 1 and 2 are cheap and can
stop the work before the expensive pieces are built.

| step | pieces | decides | uses tables |
|---|---|---|---|
| 1 | A1, A2, A5, A6, B1, D1 | whether $h$ can locate a PUMA in type space (rare groups especially); the cost and gain of statewide variety; $\gamma$ | no (tables only as checks) |
| 2 | B2 prototype, information diagnostic | whether $S$ is computable at PUMA scale; whether the tables identify $u_b$ | yes |
| 3 | B3, C1, D2 | MAP accuracy against today's PMEDM, with truth | yes |
| 3b | B3, C1, D2 with a separate deviation basis | whether the zone deviations stay, and on which basis; $\tau$ chosen by held-out tables | yes |
| 4 | C2, D2 coverage | whether the posterior is calibrated | yes, with $Y$ drawn with survey error |
| 4b | C2, D2 coverage under model version 4 | whether the overdispersed realized level is calibrated, and $\kappa$ | yes, as step 4 |
| 5 | C3, C4 | only if step 4 fails for rare groups | yes |
| 6 | D3 | the real application | yes |

## Adopted defaults

* **The information diagnostic comes before any MAP fitting** (step 2).
* **$h$ candidates for step 1:** decoder category probabilities (7 and 37
  attributes), quadratic in $z$, and a kernel basis with 100 to 200
  centres.
* **Leave-zone-out support is standard for the regional run's coverage
  test** (and available for MAP scoring): when fitting a zone, its own
  records are dropped from the support.

## Analysis findings

1. **The tables may barely identify $u_b$.** The tables' information about
   a block's deviation is approximately (derived here)
   $$
   I_b \approx N_b^2\, \mathrm{Cov}_{p_b}(h, t)\, D_b' S^{-1} D_b\,
   \mathrm{Cov}_{p_b}(t, h).
   $$
   With large block-group survey variance it may be small next to the
   prior $1/\tau^2$. Then blocks collapse to $\theta_P$ and differ only
   through the realized level, which the weight $N_b$ holds tightly. Step 2
   computes it at $\hat\theta_P$ before B3 is built.
2. **The MAP of $\tau$ tends to collapse to zero** (joint MAP over a
   variance and its random effects is degenerate; recalled from the
   Bayesian literature, to verify). Fix $\tau$ on a grid for the first
   MAP, or estimate it from a Laplace-approximate marginal likelihood.
3. **The existing PMEDM solver serves as C1** (derived here). Over (unit,
   block) pairs, $\mathrm{KL}(r\Vert q) = \sum_b \pi_b \mathrm{KL}(r_b\Vert
   p_b) + \mathrm{KL}(\pi\Vert\pi^0)$, with $\pi$ the blocks' shares. With
   $q_{ib} = (N_b/N)\,p_{bi}$ and weight $N$, the existing solver solves the
   conditional PMEDM plus a term letting block totals move. Fix the shares
   when $N_b$ is known; the extra term models uncertain $N_b$. `solvers/`
   needs no change.
4. **The regional run is optimistic for support-based methods.** Each
   zone's true records are among the statewide atoms; real block groups'
   residents mostly are not in the PUMS. Comparisons with today's PMEDM
   stay fair (same advantage), but absolute accuracy and coverage are
   flattering. Hence the leave-zone-out default above.

## Smaller notes

* Group-quarters persons are atoms like households; $N_b$ counts both, and
  the reproduction check includes group-quarters type.
* $\gamma$ enters both the PUMA reference and the block superpopulation; it
  is chosen in step 1.

## Constraints

* No changes to `src/pmedm_vb/assemble/`, `src/pmedm_vb/solvers/`, existing
  `src/pmedm_vb/data/` modules or existing experiments. New code under
  `src/pmedm_vb/latent/` and `experiments/latent/`, importing the rest.
* Census API calls only on the ISAAC login node or dtn1/dtn2; batch jobs
  run offline.
