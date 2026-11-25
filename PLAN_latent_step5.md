# Plan, step 5: drawing the realized populations

> **Status: fixed.** Step 5 of `PLAN_latent.md` (version 7),
> for model version 5 in its cut form (`MODEL_latent.qmd` version 5, after
> step 4e). Once approved, change it only
> when the plan itself changes, and record each change below.
>
> | version | date | change |
> |---|---|---|
> | 1 | 2026-11-24 | approved; every item of §7 as drafted (items 1 and 2 with the reasons now given) |
> | 2 | 2026-11-25 | after the pilot (§8): proposal (d) is dropped; two moves are added, the user's kernel proposal (e) and exact two-atom pile moves (§3a); a phase 1b gate for them, and the pilot is rerun (§4). Thresholds unchanged |
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

## 3a. Kernel and pile moves (version 2)

The pilot (§8) found the single-unit moves exact but slow where the fitted
tables leave the posterior free: which records hold a zone's units among
records that look alike to the fitted tables (B1, B2, joint cells: one
effective draw per 400–1,400 sweeps; chains from different starts still
apart after 5,000). Two causes (derived here, consistent with the pilot,
not proven): changing those directions needs coordinated moves, whose
half-way states misfit precise tables; and with small per-atom
concentrations the posterior piles units on few records, so a single unit
rarely opens a new record. Proposal (d) would steer single moves by the
fitted tables' residual, which speeds the directions that already mix; it is
dropped. Added:

**(e) Kernel proposal (the user's).** Raw kernel weights of the deviation
basis $g$, $K_{ik} = \exp\{-\lVert z_i - c_k\rVert^2/2\ell^2\}$, recomputed
from the basis's stored centres $c_k$ and bandwidth $\ell$ (the stored
columns are standardized, not these weights). For a unit at atom $i$ of
zone $b$: pick a centre $k$ with probability $K_{ik}/\sum_{k'}K_{ik'}$, then
an atom $j$ with probability $w_{bj}K_{jk}/S_{bk}$, $S_{bk} =
\sum_{j'} w_{bj'}K_{j'k}$, $w_b = p_b$ by default. Its probability, needed
in both directions for the Hastings ratio, is

$$
Q(j \mid i) = \sum_k \frac{K_{ik}}{\sum_{k'} K_{ik'}}\cdot\frac{w_{bj} K_{jk}}{S_{bk}},
$$

a sum over the centres. It joins (a)–(c) in the single-unit mixture.
Proposed records are near $i$ in type space, so their fitted profiles are
usually close and the move is accepted more often. Basis: `kern30`
(default) or `kern100`, chosen at the pilot.

**Pile moves (exact two-atom Gibbs).** Pick atom $i$ of zone $b$ from
$\hat r_b$, and atom $j$ from $i$ by the kernel route of (e); both choices
depend only on fixed weights, not on the counts. With
$n = C_{bi} + C_{bj}$ fixed, redraw the split exactly from its conditional
(derived here): for $m = 0, \dots, n$ the new $C_{bi}$,

$$
\log \Pr(m) = \text{const} + \log\frac{\Gamma(m + \alpha_{bi})}{m!}
  + \log\frac{\Gamma(n - m + \alpha_{bj})}{(n - m)!}
  + \Delta\, d'\Sigma^{-1}e - \tfrac12 \Delta^2 d'\Sigma^{-1}d,
$$

with $\alpha_{bj} = \kappa s_b p_{bj}$, $\Delta = C_{bi} - m$ the units
moved from $i$ to $j$, $d = D_b\{t(x_j) - t(x_i)\}$ and $e$ the current
residual. A random-scan Gibbs step on the pair $(C_{bi}, C_{bj})$ with a
state-independent choice of pair leaves the target invariant. It moves a
whole pile at once, the move single units rarely make; for atoms with
identical fitted profiles ($d = 0$) it is exactly the redraw of their split.
Cost $O(n)$ plus one quadratic form.

**A sweep** is $N_b$ single-unit moves per zone (mixture (a), (b), (c), (e))
plus a number of pile moves per zone set at the pilot. Not included now:
exact within-class Gibbs over whole classes (covered for pairs by the pile
moves), tempering, the continuous relaxation; they are the options if the
rerun pilot fails.

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

**Version 2:** 1b. **Exactness of the new moves** (gate): enumeration
checks as phase 1 for proposal (e) alone and in the mixture, and for the
pile moves alone and with single-unit moves, both covariance backends, with
broken variants (the kernel proposal's reverse probability left out; the
pile conditional without its likelihood term). 2b. **The pilot rerun:**
the same two regions and lengths as §8, reporting the same diagnostics
beside the first pilot's, for the move settings chosen (kernel basis, pile
moves per sweep), plus households by size (units with 1, …, 7+ persons and
persons in households of 7+) among the traced functionals.

## 5. Decision rules (fixed before results)

* **Gate (phase 1):** total variation against enumeration within Monte
  Carlo error and shrinking with run length, for every proposal and
  covariance form; otherwise fix before any regional run.
* **Mixing (phase 2):** median effective sample size per draw of at least
  0.1 per sweep for every functional class, rare included, and
  $\hat R < 1.01$ at the MAP parameters (thresholds chosen here). Otherwise
  build proposal (d); if that fails too, review. (Version 2: the first pilot
  failed this rule; (d) is replaced by §3a's moves, judged by the same rule
  in phase 2b; if they fail, review, with §3a's remaining options.)
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

## 7. Resolved at approval

1. **Approved:** the hybrid (parameters as stage 2, MCMC for $C$ only);
   full parameter MCMC deferred.
2. **Approved:** model version 5 as step 4e leaves it (cut or free shares).
   Version 4 is not sampled: its zone totals are part of $C$, so a chain
   would need moves across zones, and a total's posterior (sd about 1,000
   units) would then mix by a random walk of single-unit moves; under
   version 5 each parameter draw brings an independent draw of the totals.
3. **Approved:** proposals (a)–(c) first, (d) if needed; CPU or GPU (ISAAC's
   GPUs are available) decided by the pilot.
4. **Approved:** the thresholds in §5.
5. **Approved:** a compiled kernel (numba, which the project environment
   lacks) may live in a separate environment if torch batching is too slow;
   the main environment is not changed.

## 8. The first pilot (version 1's moves)

Simulation S, replicate 0, knoxville_core (4 zones, 213,772 units) and
nashville_core (6 zones, 345,520), the cut form at step 4c's pairs, 4
chains at the MAP parameters from dispersed starts and 4 parameter draws,
6,000 sweeps (1,000 burn-in), proposals (a)–(c) at 0.4/0.3/0.3, numba on 16
threads.

* **Exact and fast:** zone totals held exactly, drift about $10^{-13}$;
  0.18–0.22 s per sweep for 4 chains (5–6 million moves per second), about
  500–600 times torch's.
* **Mixing** (median bulk ESS per sweep; max $\hat R$): fitted and rare
  0.019–0.026 (1.09–1.51); B1, B2, joint 0.0007–0.0028 (1.38–1.99);
  nashville_core (smaller $\kappa$, $c = 0.125$) the slower. The mixing rule
  fails.
* **Agreement with stage 2,** where the chains mix: fitted and rare
  variance ratios about 1.0, mean differences about 0.08 sd; the unfitted
  classes not assessable at their Monte Carlo error.
* **Fitted profiles:** 155,047 atoms have 94,463 distinct fitted profiles;
  46% of atoms share one, mostly in pairs (median class size 1, 99th
  percentile 13).
