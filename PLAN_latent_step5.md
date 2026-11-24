# Plan, step 5: drawing the realized populations (draft)

> **Status: draft, for review.** Step 5 of `PLAN_latent.md` (version 7),
> for the draft model version 5 (`MODEL_latent_v5_draft.md`): free shares,
> or their cut form if step 4e adopts it. Once approved, change it only
> when the plan itself changes, and record each change below.
>
> | version | date | change |
> |---|---|---|
> | 1 | 2026-11-24 | draft |
>
> "Derived here" marks derivations made for this project, not taken from
> the literature.

## 1. Why

The target is the posterior over the realized populations, $C \mid Y$:
draws of non-negative integer counts on real records, from which any table,
rare-group total or nonlinear summary (a median, a rate) can be computed.
Stage 2 (Laplace for the parameters, Gaussian conditioning for $C$) gives
calibrated intervals for linear functionals of $C$ (steps 4–4d), but its
draws are real-valued and can be negative: on simulation S a large share of
90% intervals reach below zero (B1 0.39, rare groups 0.30, joint 0.25), and
it gives no populations to compute nonlinear summaries from. This step
builds a sampler for $C$ itself. (`PLAN_latent.md` versions 1–6 made it
conditional on rare groups failing; version 7 makes it unconditional.)

## 2. What is sampled

**A hybrid, as stage 2:** the parameters $(\theta_A, U)$ and the zone
totals are drawn as now (Laplace for $U$; the shares' Laplace approximation
under free shares, or the totals module under the cut), and only $C$ given
them is sampled, exactly, by MCMC. The parameters' approximation is the one
steps 4–4d validated; full MCMC over the parameters (piece C4) is deferred.

**Target given a parameter draw** (model version 5): given the totals,
zones are independent a priori,

$$
\pi(C \mid \cdot, Y) \propto \prod_b \mathrm{DirMult}\big(C_b;\, N_b,\,
\kappa s_b\, p_b\big)\; \exp\{-\tfrac12 e'\Sigma^{-1}e\},
\qquad e = Y - \textstyle\sum_b D_b T'C_b,
$$

with $\Sigma$ the table covariance ($\hat\Sigma + \omega^2 I$ in
simulation S). The zone totals are fixed within a draw, so moves stay
within a zone.

## 3. The sampler

**State:** each zone's $N_b$ individuals as atom labels (an integer array
per zone), the counts $C_b$, the residual $e$ and what is needed of
$\Sigma^{-1}e$.

**Move:** in zone $b$ pick an individual uniformly (its atom $i$), propose
atom $j \sim Q$, accept with probability (derived here)

$$
\alpha = \min\Big\{1,\;
\frac{C_{bj} + \kappa s_b p_{bj}}{C_{bi} - 1 + \kappa s_b p_{bi}}\;
\frac{Q_i(C')}{Q_j(C)}\;
\exp\{d'\Sigma^{-1}e - \tfrac12 d'\Sigma^{-1}d\}\Big\},
\qquad d = D_b\{t(x_j) - t(x_i)\},
$$

$C'$ the state after the move. The first factor is the Dirichlet-
multinomial's ratio combined with the uniform pick (as $\kappa \to \infty$
it becomes `MODEL_latent.qmd`'s $p_{bj}/p_{bi}$). **Checked** on a toy
(4 atoms, 6 individuals, 84 states, exact posterior by enumeration): total
variation 0.004 after 2 million moves; with the proposal correction left
out, 0.36.

**Proposals,** a mixture whose state-dependent parts are evaluated exactly
at $C$ and $C'$:

* **(a) occupancy:** another individual's atom in the same zone
  ($Q_j = C_{bj}/N_b$): moves individuals onto atoms already in use, which
  the Dirichlet-multinomial favours when $\kappa s_b p_{bj}$ is small;
* **(b) the realized MAP** $\hat r_b$ (fixed within a parameter draw): a
  PMEDM-shaped proposal that already reflects the tables, reaching rare
  types;
* **(c) the superpopulation** $p_b$: a heavy-tailed fallback;
* **(d) tables-informed, locally balanced** (`MODEL_latent.qmd` stage 3,
  option ii): $Q_j \propto p_{bj}\exp\{\tfrac12 t(x_j)'D_b'\Sigma^{-1}e\}$,
  recomputed from the current residual. Exact, the best mixing per move,
  but each move needs scores over all atoms (a product of size
  $M \times J$) and the reverse normalizer: practical on a GPU, not
  otherwise. Built if the pilot shows (a)–(c) mixing too slowly.

**Linear algebra:** $d$ touches only zone $b$'s rows and the nonzero cells
of $t(x_j) - t(x_i)$. With $\Sigma$ block-diagonal by zone (simulation S),
each zone's $\Sigma^{-1}$ block is held dense and zones are exactly
independent, so all zones (and all chains) move in parallel. With a
replicate covariance across zones (simulation G, the real application),
$\Sigma = D + LL'$ with rank about 80: keep $e$, $z = L'D^{-1}e$ and
$K^{-1}z$ ($K = I + L'D^{-1}L$), so a move costs about
$\mathrm{rank}\times\mathrm{nnz}(d) + \mathrm{rank}^2$ (derived here), and
zones then move in sequence within a chain.

**Chains:** one chain per parameter draw, all advanced together as one
batch (torch, CPU or GPU; `solvers/mcmc.py` batches its HMC chains the same
way). Start: a multinomial draw from $\hat r_b$ with the draw's totals.
Burn-in, then thinned draws; lengths set by the pilot.

## 4. Phases

1. **Exactness** (gate): enumeration checks as above for the full code path
   (several zones, the mixture proposals, both covariance forms), total
   variation shrinking with run length; and agreement with the stage 2
   Gaussian at the MAP parameters on a mid-sized synthetic problem where
   the Gaussian is accurate.
2. **Pilot** (2 regions, 1 replicate of simulation S): effective samples
   per second for the functionals of every class, rare cells especially,
   on CPU and on GPU, with proposals (a)–(c) and, if needed, (d);
   $\hat R$ from 4 chains at the MAP parameters, from dispersed starts. It
   sets burn-in, thinning, the proposal mixture and the hardware.
3. **Regional run** on S (model version 5 as step 4e leaves it: the cut if
   adopted, otherwise free shares), all regions and replicates if the pilot
   allows, otherwise 3 replicates per region (decided at the pilot,
   recorded). 50 parameter draws per unit, 4 thinned draws of $C$ each:
   200 draws, scored exactly as steps 4c–4e, so its calibration sits
   beside stage 2's.

## 5. Decision rules (fixed before results)

* **Gate (phase 1):** total variation against enumeration within Monte
  Carlo error and shrinking with run length, for every proposal and
  covariance form; otherwise fix before any regional run.
* **Mixing (phase 2):** median effective sample size per draw of at least
  0.1 per sweep for every functional class, rare included, and
  $\hat R < 1.01$ at the MAP parameters (thresholds chosen here). Otherwise
  build proposal (d); if that fails too, review.
* **Agreement with stage 2 (phase 3):** for cells with posterior mean at
  least 50, mean differences within 0.1 posterior sd and variance ratios
  within $[0.8, 1.25]$ in at least 90% of cells (chosen here). A larger
  gap is reviewed, since one of the two is wrong there.
* **Calibration (phase 3):** step 4c's bands for every class. With
  agreement and calibration, the sampler is the stage 3 for step 6 (the
  real application), and step 6 reports draws of $C$; the share of
  intervals below zero is 0 by construction.

## 6. Compute

Unknown until the pilot. The sequential part is the moves within a chain:
one sweep is about $N_A$ moves (a few hundred thousand per region), so per
sweep a few seconds in compiled code, tens of seconds batched in torch.
The GPU helps by running many chains per batch and by making proposal (d)
affordable. The pilot reports seconds per sweep, sweeps to convergence and
effective samples per second, and the regional run is sized from them.

## 7. To resolve at approval

1. The hybrid (parameters as stage 2, MCMC for $C$ only); full parameter
   MCMC deferred.
2. Model version 5 as step 4e leaves it (cut or free shares); version 4,
   whose zone totals move within the realized level and would need moves
   across zones, not sampled.
3. Proposals (a)–(c) first, (d) if needed; CPU or GPU decided by the pilot.
4. The thresholds in §5.
5. Whether a compiled kernel (numba, which the project environment lacks)
   may live in a separate environment if torch batching is too slow; the
   main environment is not changed.
