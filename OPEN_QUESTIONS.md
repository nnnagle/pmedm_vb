# Open questions

Questions under investigation, so that several can be worked on at once. Each
entry: its status, the question, why it came up, what is running or needed, and
what would answer it. Move an entry to **Answered** with its answer when done.

| # | Question | Status (2026-10-22) |
|---|---|---|
| 1 | HMC draws consistent with Sigma on fitted cells? | per cell at alpha 0.01: bulk yes, tails no; waiting on job 6457912 for pooled and d/m |
| 2 | n_eff against D/H for IPF | not started: the share-distribution check is not written |
| 3 | Why IPF does not converge | waiting: `ipf_diagnosis.py` submitted |
| 4 | Tweedie error for small cells and zeros, keeping Sigma | design: p = 1 fits the dual, 1 < p < 2 does not (corrected); decisions pending |

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
- **Running: `experiments/draw_sigma.py`**, alpha 0.01, job 6457912
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

- **Status:** waiting on `ipf_diagnosis.py` (commit fecf662); results in
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
- **Answer from:** which causes account for IPF's misses, and whether any
  block group's tables are inconsistent on the data.

### 4. A Tweedie error model for small cells and zeros, keeping Sigma?

- **Status:** idea; nothing written or run.
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

## Answered

(none yet)
