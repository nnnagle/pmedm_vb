# Open questions

Questions under investigation, so that several can be worked on at once. Each
entry: its status, the question, why it came up, what is running or needed, and
what would answer it. Move an entry to **Answered** with its answer when done.

| # | Question | Status (2026-10-22) |
|---|---|---|
| 1 | HMC draws consistent with Sigma on fitted cells? | alpha 0.01: bulk matches Sigma's diagonal, tails do not, correlations do not (d/m 18-27) |
| 2 | n_eff against D/H for IPF | not started: the share-distribution check is not written |
| 3 | Why IPF does not converge | exp03 4701504: every miss is in a block group the LP cannot meet exactly; other PUMAs and experiments not yet read |
| 4 | Tweedie error for small cells and zeros, keeping Sigma | zero cells implemented (`+tweedie<w>`); VB fixed (continuation from 90%); starts near the MAP pass on 4701502 (job 6459108); exp04 to use K0 (#5) |
| 5 | A k = 0 squared-Gaussian dual term: count-like cells with Sigma exact | k = 0's MAP stalls; k = 0.01 converges on 4701502 (14 iterations) with Sigma barely changed (job 6459604); next: HMC starts |

## Open

### 1. Are the HMC draws consistent with Sigma on the fitted cells?

- **Status:** marginal check done (jobs 6456752, 6457035; `$R/draw_z.csv`,
  `$R/draw_z_cells.csv`): not consistent. Proposed next: the multivariate check
  (below). Held-out cells are out of scope here.
- **The question:** in the model the error e = N X'p - Y on the fitted cells
  has the Gaussian penalty e' Sigma^-1 e / 2 (`pmedm_derivation.md`); each
  cell's variance is its published v at every alpha (alpha only shrinks the
  correlations). Do the HMC reference draws' errors look like that?
- **Why it came up:** PUMA 4701503 (Baseline, alpha 0.01), the least diverse
  draw (1457) puts 72 women aged 70-75 in block group 470930038014, published 0
  (SE 8.5): z = +8.5 on a constraint cell. The largest of ~39 million |z| would
  be about 5.9 under normal tails.
- **Marginal result, constraint cells only (z = error / sqrt(v)):**
  - Far heavier tails than Sigma allows, at both alphas. Baseline, alpha 0.01,
    per draw and PUMA: about 33 of ~5,900 nonzero block-group cells over |z|
    5.9, 17 zero cells, 42 nonzero tract cells; under N(0, Sigma) even the
    largest would be about 4. The median draw's largest |z| is 11-47 by group.
  - Excursions, not fixed cells: at alpha 0.01 almost no cell is off in most
    draws (persistent, median |z| > 5.9: 0 at block group in Baseline and Null
    space, 5-6 at tract), yet ~16,000 of 23,700 nonzero block-group cells
    exceed 5.9 in some draw. The posterior spread of the allocated counts is
    much wider than sqrt(v).
  - Alpha 1 adds persistent misfits (Baseline block group 65, tract 53;
    Roll-up 104, 167): tiny counts with tiny SEs (median published 1-6, SE
    0.6-2.9). Worst: B26001.total (group quarters) in tracts of 4701501,
    published 1-18, SE 0.6-1.5, median allocation 39-72. Why alpha matters
    here is not known. Roll-up's persistent ones are by construction (it fits
    merged cells; z is per leaf).
  - The 4701503 z = +8.5 is typical of these draws, not odd.
- **Limit:** the marginal check ignores Sigma's correlations, strong at alpha
  0.01.
- **Correction:** `draw_z.py` divides by the published SE, but with the
  multiplier prior (`+prior3`) the model's Sigma is the published one plus
  tau^2 I, tau = N / (n s), about 6.3 people (`prior.py`). Where v is small
  that overstates z badly (B26001.total: SE 0.63 against a model sd of about
  6.3, z about 59 -> 6; the 4701503 cell: +8.5 -> about +6.8). The results
  above are against the published variances, not the model's.
- **Done: `experiments/draw_sigma.py`**, alpha 0.01, job 6457912 (completed, 42 min)
  (`$R/draw_sigma.csv`, `$R/draw_sigma_hist.csv`; summaries from "Correlations
  included" on in `/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/slurm-6457912.log`),
  on the model's own rows
  (PUMA rows, roll-up cells) and its own Sigma at each alpha, as `solve_map`'s
  `mahalanobis`: (1) per row, z = e / sqrt(diag Sigma), pooled by level and
  zero/nonzero: the bulk (median |z|, share within 1 and 1.96) and the tails
  (share over 3, 4, 5, 5.9; 99.9% and 99.999% points) beside a normal's, with
  the published-variance version beside it; (2) per draw d / m = e' Sigma^-1 e
  / m, correlations included. Not a chi-square test: for the MAP the residuals
  are shrunk and d / m is below 1 when the model holds (`map_dual`).
- **Expected (user):** at alpha 0.01, the bulk matches the diagonal of Sigma;
  the tails do not match a normal.
- **Per-cell result, alpha 0.01, constraint leaf cells, against the model's
  diagonal sqrt(v + tau^2)** (tau 6.0-6.6 by PUMA; from `draw_z_cells.csv`,
  Baseline, Null space alike): **the expectation holds.**
  - Bulk, roughly consistent: nonzero cells' median draw 0.33 model sds from
    the published value (90% of cells within 0.75); the typical cell's 90%
    band of draws reaches 1.96-2.01 sds, i.e. a spread of about 1 sd (0.34 +
    1.645 sigma = 1.96; a rough back-calculation). Zero cells tighter: centre
    0, band 0.3-0.4 sds. 10% of nonzero cells' bands reach beyond 3.2.
  - Tails, not normal: the typical cell's worst of 4,000 draws is 7.1-7.3 sds
    (nonzero), 5.3-6.1 (zero), against about 3.5-4 for 4,000 normal draws; 1%
    of cells beyond 25; worst 85-94. Zero cells: band 0.3-0.4 sds, worst draw
    5-6 -- narrow bulk, rare far excursions.
  - tau barely moves the bulk (band 2.07 -> 1.96) but shrinks the extremes
    (worst cell 524 -> 86): the astronomical published-SE z came mostly from
    tiny-SE cells. Roll-up's leaf-level numbers are approximate (its model rows
    are merged cells).
- **Pooled result (job 6457912), alpha 0.01, the model's rows, z against
  sqrt(v + tau^2), all draws and PUMAs:**
  - Bulk matches the diagonal: nonzero rows median |z| 0.67-0.71 (normal
    0.674), within 1: 0.67-0.74 (0.683), within 1.96: 0.92-0.945 (0.95), at
    every level and in every experiment. Zero rows tighter in Baseline and
    Null space (block group median 0.16-0.17, PUMA ~0).
  - Tails do not: nonzero rows over 3: 0.9-2.6% (normal 0.27%); over 4:
    0.3-1.2% (0.006%); over 5.9: 0.13-0.33% (3.6e-9); 99.999% point 15-33
    (4.4). Zero block-group rows over 5.9: about 1% (Baseline, Null space).
  - Roll-up's zero rows (merged zero cells) are the worst: median |z| 0.72-0.82
    and 3-4% over 5.9; its nonzero rows have the lightest tails. Not yet
    explained (a merged cell carries several cells' allocations against one
    zero variance -- a guess).
  - Against the published v alone the tails are heavier still (99.999% point
    up to 61): tau^2 absorbs part of the excess.
  - **Correlations do not match:** d / m = e' Sigma^-1 e / m per draw, median
    25.6 (Baseline), 18.2 (Null space), 27.2 (Roll-up); 5-95%: 16-31. If the
    draws' errors had Sigma's correlation pattern this would be of order 1
    (below 1 for the MAP; for posterior draws, by analogy with a linear
    Gaussian model, at most about 2 -- an analogy, not a derivation for this
    model). At alpha 0.01 Sigma allows only D = alpha v + tau^2 outside the
    rank-80 replicate directions, so errors with the right marginal size but
    outside those directions are penalised heavily. To calibrate the scale:
    tr(Sigma^-1 diag(v + tau^2)) / m, the d / m of independent errors of the
    right size -- not yet computed.
  - Reading: per row the posterior is as wide as Sigma says, with much heavier
    tails; jointly its errors are not shaped like Sigma.

### 2. Why do n_eff and D/H rank IPF against the posterior differently?

- **Status:** not started; nothing running. Next: write the share-distribution
  check under **Needed**.
- **Why:** at alpha 0.01, IPF's smallest block-group n_eff (about 12) is above
  the HMC reference's median (about 9), but its smallest D/H (about 0.06) is
  below HMC's median (about 0.11) (Figure B, n_eff and D/H versions). Likewise
  VB's typical draw has a low n_eff but a D/H like HMC's.
- **Reading so far (from the definitions, not checked):** n_eff = 1 / sum s^2 is
  set by the largest shares; D/H counts records with a share around 1/H or
  more, each about 1 however large. So IPF would avoid very large single
  shares but leave few records with a meaningful share; VB's typical draw
  would have one dominant record but many others in use.
- **Figure A (PUMA 4701504, alpha 0.01):** IPF's D/H is not uniformly low: in
  a typical block group it is above HMC's (0.35 against about 0.18-0.31), in
  its least spread one below (0.066 against 0.12-0.17), as Figure B's medians
  (IPF above) and minimums (IPF below) suggest. So IPF's diversity varies more
  across block groups than the posterior's; part of the disagreement may be
  that, rather than n_eff against D/H.
- **Needed:** in a block group or two (e.g. the IPF fit's smallest D/H in each
  PUMA), the distribution of shares under IPF and under a few HMC draws: the
  largest shares, how many records are over 1/H, 0.1/H and 0.01/H, and how much
  of the households the records under 1/H hold together.
- **Answer from:** whether the two measures pick out different features
  (concentration against breadth) and which one each figure and table should
  report.

### 3. Why doesn't IPF converge?

- **Status:** `ipf_diagnosis.py` (commit fecf662) has finished; exp03_rollup 4701504 read (below), the rest not yet. Results in
  `$R/ipf_diagnosis/ipf_diagnosis.txt` and `ipf_cells.csv`.
- **Why:** none of the 12 IPF fits (3 experiments x 4 PUMAs) converged: all
  stopped at 2,000 sweeps. Worst miss |log(m / Y)|: Baseline 1.9-2.8, Null
  space 2.1-2.9, Roll-up 0.53-1.58. Infeasible rows (a published nonzero cell
  with no carrier, or with every carrier removed by the block group's zeros):
  Baseline 5-17, Null space 5-17, Roll-up 0-2 per PUMA; exp03_rollup 4701502
  has none and still does not converge.
- **Not slow convergence:** the worst residual is flat from sweep 100-500 on
  in every fit (saved trace), so the margins cannot all be met.
- **Reading:** Census's tables can be mutually consistent and still not be
  attainable by reweighting the PUMA's records: missing carriers, carriers
  removed by published zeros, or records whose bundles of categories cannot add
  up to every margin at once.
- **Running:** `experiments/ipf_diagnosis.py` (rewritten for paper_v3):
  per experiment and PUMA, on IPF's fitted cells, no-carrier and removed-by-zero
  cells, same-universe table totals that differ (the tables' consistency on the
  data), and an exact LP per block group: can nonnegative weights meet every
  fitted cell, and if not, the least relaxation and which cells; IPF's misses
  attributed to each.
- **Result, exp03_rollup, PUMA 4701504** (80 block groups, 4,827 fitted
  cells): no cell without a carrier; 1 with every carrier removed by its block
  group's zeros (470930061024, B08301.public_transport_bus, published 25: IPF
  gives 0); 15 with only 1-2 carriers left; no same-universe conflict. The
  exact LP meets 75 of 80 block groups; the other 5 need 21 cells relaxed
  (least total relative relaxation median 0.66, max 3.96). IPF misses 90 of
  4,440 cells by more than 5%, **all 90 in those 5 block groups** (18 of them
  cells the LP relaxes, the other 72 not: IPF spreads the conflict over the
  block group, the LP concentrates it). Largest misses in 470930061024,
  470930061021 and 470930035011, on cells with few carriers left after zeros.
  So here non-convergence is infeasibility on the data, not the algorithm.
- **Answer from:** which causes account for IPF's misses, and whether any
  block group's tables are inconsistent on the data.

### 4. A Tweedie error model for small cells and zeros, keeping Sigma?

- **Status:** the published zeros' term is implemented as the dual's CGF
  (`assemble/tweedie.py`, level suffix `+tweedie<w>`, derivation in
  `pmedm_derivation.md`; spec `experiments/tweedie/exp04_tweedie.json`, PUMA
  4701502, alpha 0.01, w = 18). Its exact primal is the square-root-scale
  penalty (sqrt(m) - sqrt(mu0))^2 / w, which is convex; the convexity problem
  below is the Tweedie *likelihood's*, not this term's. Open: VB with the term
  is unstable (ELBO noise); the continuation from 90% of the bound is being
  tested before exp04 is run. Nonzero cells are still Gaussian: #5 would
  extend the term to them with Sigma's correlations.
- **Starting HMC without VB (waiting on job 6459108):** HMC's starts are
  screened (largest cell <= 1% of p, inside the bound) and drawn from the VB
  fit; VB with the term is unstable and Laplace draws are far too
  concentrated. `experiments/start_check.py` tests starts MAP + c (Laplace
  draw - MAP), c from 0 to 1, on PUMA 4701502, alpha 0.01, `+tweedie18`. On
  the synthetic problem the bound never binds (100% inside up to c = 0.7, 96%
  at c = 1) but the share screen does: the median largest share is 0.02 at the
  MAP and 0.82 at c = 1, so nothing passes -- not informative there, as the
  MAP itself holds 2%. The job answers whether the MAP and small c pass on the
  real PUMA. Next: a pilot chain from those starts (divergences, step size),
  re-whitened from the pilot only if needed.
- **Result (job 6459108, PUMA 4701502, alpha 0.01, `+tweedie18`, 16,049
  multipliers):** the MAP converges in 10 Newton iterations (e' S^-1 e / m
  0.46, largest |z| 9.5) and its largest cell holds 0.34% of p. Starts
  MAP + c (Laplace draw - MAP) pass both screens for every c <= 0.1; at
  c = 0.2 only 36% (all lie under the share cap, 64% cross the bound); none at
  c >= 0.3. Here, unlike the synthetic problem, the bound binds first; the
  share screen fails from c = 0.5 (median largest share 0.03, 0.50 at 0.7,
  0.86 at 1). The Laplace Gaussian is several times too wide toward the bound
  and toward concentration; starts near the MAP are usable.
- **VB:** with the continuation from 90% of the bound (`EXTEND = 0.9`,
  commit 7d15dbf) the synthetic Laplace ELBO's standard error is 106 (101
  without the term; 15,869 at 0.99) and the fit climbs at the same rate; its
  draws are 99.8% inside the bound, median largest share 0.29 (0.22 without
  the term). Neither VB fit converged in 300 iterations, with or without the
  term.
- **Why:** for large cells a Gaussian sampling error is plausible; for small
  cells, and most obviously zeros, it cannot be (a published count is
  nonnegative, often exactly zero). The model's error is Gaussian with Sigma
  for every cell.
- **Proposal:** Sigma(alpha) = D + (1 - alpha)(4/80) L L' (`sigma.py`). Keep the
  low-rank, correlated part Gaussian; give the diagonal part a Tweedie error,
  1 < p < 2 (compound Poisson-gamma: Jorgensen 1987, JRSS B; computation Dunn
  and Smyth 2005, Statistics and Computing): mean the allocated count mu,
  variance phi mu^p, positive probability at exactly zero, near Gaussian for
  large mu. phi_i set so the variance at the published value is D_i: second
  moments still Sigma there, correlations untouched. Motivation, not fact: a
  published count is roughly a sum over a random number of sampled people of
  their positive weights (ACS weights are also raked to controls).
- **Effect on the model -- corrected:** the Tweedie penalty is NOT convex in
  the mean mu for 1 < p < 2: psi(mu) = [mu^(2-p)/(2-p) - Y mu^(1-p)/(1-p)] / phi,
  psi'' = mu^(-p-1) [p Y - (p-1) mu] / phi, convex only below p Y / (p-1)
  (3 Y at p = 1.5) and concave everywhere at Y = 0 (Tweedie is concave in its
  canonical parameter, not its mean). PMEDM's posterior over the multipliers
  rests on the convex dual (`pmedm_derivation.md`), so a 1 < p < 2 Tweedie is a
  new model derivation, not a new dual term; an earlier version of this entry
  said otherwise.
- **What fits the framework: p = 1, an overdispersed Poisson** (the Tweedie
  family's convex end): psi(mu) = (mu - Y log mu) / phi, variance phi mu,
  psi'' = Y / (phi mu^2) >= 0, at Y = 0 linear (mu / phi). Its conjugate is in
  closed form but finite on one side only (a one-sided bound on each
  multiplier): HMC must respect it (e.g. a transformed multiplier), MAP and VB
  need checking near it. The low-rank Gaussian part with Sigma's correlations
  stays. Precedent: `rake.py`'s unbalanced Sinkhorn uses this generalised-KL
  penalty (no zeros, no correlations). Calibration phi_i = D_i / Y_i for
  nonzero cells.
- **Caution:** its penalty for over-allocation grows more slowly than the
  Gaussian's (like sqrt(mu) at Y = 0 for p = 1.5), so it would allow more
  over-allocation into small and zero cells, not less. Illustration, 72
  people in a published zero: Gaussian (sd 10.6) about 23 nats; Tweedie with
  the variance at mu = 1 set to 72, about 0.24 nats. A better error model, not
  a way to trim extreme draws (that would act on concentration instead: a
  tighter multiplier prior, the share cap `--share-cap`).
- **Choices to make:** p = 1 (fits the dual) or 1 < p < 2 (a new
  derivation); how to calibrate phi for zero cells (at mu = 1 against the
  modelled zero variance, phi = 72, 72 people in a zero cost 1 nat; or the
  area's typical phi of small nonzero cells); which PUMA (fastest HMC
  reference: `hmc_seconds` in `all_scores.csv`).
- **Answer from:** a fit with the Tweedie diagonal, compared with today's on
  held-out accuracy, small-cell and zero-cell fit, and the draws' tails.

### 5. A k = 0 squared-Gaussian dual term: count-like cells with Sigma exact?

- **Status:** idea. Algebra checked numerically on a 4-cell example with
  negative correlations (scratch, not in the repo); nothing implemented.
- **Why:** #4 makes published zeros count-like but leaves every nonzero cell
  Gaussian, and small nonzero counts have heavy right tails too. Giving each
  cell its own count-like shape *and* Sigma's correlations was the aim. A
  linear mix of independent per-cell Tweedies (a Cholesky factor of Sigma's
  correlation; cf. Cuenin, Jorgensen and Kokonendji 2016, Computational
  Statistics, "full multivariate Tweedie") matches Sigma but not the shapes:
  at alpha 0.01 a cell's own component is about 1% of its variance.
- **The term:** with a = sqrt(Y), A = diag(a), S = A^-1 Sigma A^-1 / 4 and
  T = diag(t), replace the Gaussian CGF Y't + t' Sigma t / 2 in the dual by

      K0(t) = a' T (I - 2 S T)^-1 a,    on  S^-1 - 2 T positive definite.

  It is the k = 0 member of m = sum_r u_r o u_r, u_r ~ N(a_r, S), whose CGF is
  -(k/2) log det(I - 2 S T) + tr[T (I - 2 S T)^-1 Omega]: mean a^2 + k diag S,
  covariance 4 Omega o S + 2 k S o S (k = 1 the noncentral squared Gaussian;
  Omega = 0 the Krishnamoorthy-Parthasarathy 1951 multivariate gamma).
- **Checked (4 cells, correlations to -0.6):** K0'(0) = Y and K0''(0) = Sigma
  exactly, at any alpha; each cell's marginal is the #4 term, Tweedie p = 1.5
  with w_i = v_i / (2 Y_i); K0 is convex (log det concave, matrix inverse
  operator convex; smallest Hessian eigenvalue over 2,000 points in the domain
  0.24); K0 -> Y't + t' Sigma t / 2 for large counts. K0(t) = sup_u
  [t'(u o u) - (u - a)' S^-1 (u - a) / 2], so its primal penalty is the convex
  envelope of the multivariate square-root-scale Gaussian (each root of
  either sign); it departs from that penalty only where a large
  over-allocation meets a negative correlation.
- **Unknown:** whether exp(K0) is a genuine joint MGF when S has negative
  entries (true in one dimension, compound Poisson of exponentials). Not
  needed when the dual is taken as a model in its own right, as in #4.
- **Costs:** the domain is joint (an 80 x 80 positive-definiteness check via
  Woodbury, not a bound per row), so HMC rejects past it and VB needs a
  continuation, as in #4. S keeps Sigma's diagonal-plus-rank-80 structure, so
  K0 and its derivatives cost O(n r^2). Zero cells use a = sqrt(mu0) as in #4.
  The prior's tau^2 stays a Gaussian prior on the multipliers.
- **Variant k > 0:** removes each cell's atom at zero (gamma-like shapes) but
  adds 2 k S o S to the covariance, largest for small cells, so Sigma and Y
  cannot both be matched; Sigma would enter as a loss when fitting (a, S, k)
  (e.g. Stein's loss, the Gaussian KL from N(0, Sigma)).
- **Decision (2026-10-22): k = 0, keeping the atom at zero.** Deferred: a
  covariance of Sigma plus a diagonal without the atom, by adding an
  independent diagonal gamma piece, -(k/2) sum_i log(1 - 2 s_i t_i): covariance
  Sigma + diag(2 k s_i^2), mean kept at Y with a_i^2 = Y_i - k s_i (and S built
  from those a), no atom, but an infinite primal cost at zero allocation (a
  log barrier). Possibly in place of the prior's Gaussian tau^2 diagonal.
- **Depends on:** #4's VB stability, since the same boundary problem returns
  in a joint form.
- **Implemented** (`assemble/k0.py`, level suffix `+k0w<w>`, every model row,
  zero rows a^2 = v / (2 w), tau^2 a separate Gaussian prior; no VB). Checked
  on the synthetic problem (1,672 rows, alpha 0.01): equal to the dense
  formula; gradient and Hessian (relative error 6e-10) match finite
  differences; torch equals numpy; on zero rows without spread it equals the
  Tweedie zero term exactly; value and gradient 0.05 s per 16 draws.
- **Problem: the MAP does not converge.** The Newton decrement stalls at 0.031
  and the step halves each iteration; the full dual Hessian is right there
  (H d = -g to 1e-7 by finite differences). Along the Newton direction the
  iterate meets the joint boundary (R singular) and the K0 term stays finite
  up to it (-0.67822 to -0.67885 as R's smallest eigenvalue falls from 3e-4 to
  6e-9), blowing up only within ~1e-9 of it: its pole scales with the overlap
  of a with R's null direction, which is nearly zero at alpha 0.01 with
  thousands of rows (unlike one row or the 4-cell example). So the dual's
  minimum sits on the boundary -- in primal terms, where the convex envelope is
  a straight chord -- and the posterior would meet a wall with positive
  density there. Options: a small k > 0 (the log det term is a true barrier;
  covariance Sigma + 2 k S o S, no atom), K0 at a larger alpha, or a hard wall
  (reflective HMC, Neal 2011; a constrained MAP).
- **Trying k > 0 (user's choice: a small k and k = 1; a covariance of Sigma
  plus a diagonal is acceptable).** `+k0w<w>k<k>` adds the barrier
  -(k/2) log det(I - 2ST) (commit 4af4b9b): covariance Sigma + 2 k S o S, mean
  a^2 + k diag(S), no atom at zero. Synthetic problem (w = 1.47), all checks
  pass and the MAP converges: k = 0.01 in 53 iterations, 0.1 in 16, 1 in 9;
  variance inflation at most 0.1%, 0.6%, 5.7%; correlation change at most
  0.000, 0.001, 0.011; no sign flips. The synthetic weights understate it: a
  row's variance grows by (k/8) CV^2, a zero row's by k w^2 / (2 v) (2.25 k at
  w = 18, v = 72) and its mean becomes at least sqrt(k v) (8.5 people at
  k = 1, 0.85 at 0.01).
- **Waiting on job 6459604:** `experiments/k0_check.py` (commit 5a9fdfa) on
  PUMA 4701502, alpha 0.01, `puma-rpruned+prior3`, w = 18, k = 0.01, 0.1, 1:
  variance inflation and mean shift (zero / nonzero rows), correlation change
  over 2 million random row pairs, and whether each k's MAP converges
  (iterations, seconds). Output in `slurm-6459604.log`. If it runs out of time,
  rerun with `--no-map`. Next: `start_check.py` on the chosen k, then the pilot
  HMC run.
- **Result (job 6459604, 16,049 rows, 2,893 published zero; S's diagonal median
  4.9 people, 9 on zero rows).** Per k: variance inflation (nonzero median /
  max; zero rows), mean shift in people (nonzero median / max; zero), largest
  correlation change over 2 million pairs (no sign flips among the 388,371 with
  |rho| >= 0.1, at any k), and the MAP:

  | k | variance, nonzero | variance, zero | mean shift, nonzero | zero | max \|d rho\| | MAP |
  |---|---|---|---|---|---|---|
  | 0.01 | 0.04% / 3% | 2.25% | 0.04 / 0.74 | 0.09 | 0.006 | 14 iterations, 94 s |
  | 0.1 | 0.4% / 31% | 22.5% | 0.41 / 7.4 | 0.9 | 0.047 | 13 iterations, 87 s |
  | 1 | 3.8% / 308% | 225% | 4.1 / 74 | 9 | 0.22 | 57 iterations, 359 s |

  **k = 0.01 is close to Sigma plus a small diagonal and its MAP converges:
  chosen for exp04 (`+k0w18k0.01`).** Next: `start_check.py` with it.
- **Answer from:** an implementation checked as #4's was, then a fit against
  Baseline and the #4 fit on the draws' tails and small-cell fit.

## Answered

(none yet)
