# Open questions

Questions under investigation, so that several can be worked on at once. Each
entry: its status, the question, why it came up, what is running or needed, and
what would answer it. Move an entry to **Answered** with its answer when done.

| # | Question | Status (2026-10-22) |
|---|---|---|
| 1 | HMC draws consistent with Sigma on fitted cells? | `draw_sigma.py` written: the model's Sigma (with tau^2), bulk, tails, Mahalanobis; to run |
| 2 | n_eff against D/H for IPF | not started: the share-distribution check is not written |
| 3 | Why IPF does not converge | waiting: `ipf_diagnosis.py` submitted |

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
- **Next: `experiments/draw_sigma.py`** (to run), on the model's own rows
  (PUMA rows, roll-up cells) and its own Sigma at each alpha, as `solve_map`'s
  `mahalanobis`: (1) per row, z = e / sqrt(diag Sigma), pooled by level and
  zero/nonzero: the bulk (median |z|, share within 1 and 1.96) and the tails
  (share over 3, 4, 5, 5.9; 99.9% and 99.999% points) beside a normal's, with
  the published-variance version beside it; (2) per draw d / m = e' Sigma^-1 e
  / m, correlations included. Not a chi-square test: for the MAP the residuals
  are shrunk and d / m is below 1 when the model holds (`map_dual`).
- **Expected (user):** at alpha 0.01, the bulk matches the diagonal of Sigma;
  the tails do not match a normal.

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

## Answered

(none yet)
