# Open questions

Questions under investigation, so that several can be worked on at once. Each
entry: its status, the question, why it came up, what is running or needed, and
what would answer it. Move an entry to **Answered** with its answer when done.

| # | Question | Status (2026-10-22) |
|---|---|---|
| 1 | Tails of single HMC draws' \|z\| | results in: job 6457035, to be read |
| 2 | n_eff against D/H for IPF | not started: the share-distribution check is not written |
| 3 | Why IPF does not converge | waiting: `ipf_diagnosis.py` submitted |

## Open

### 1. How heavy are the tails of single HMC draws against the published tables?

- **Status:** the per-cell rerun finished (job 6457035, commit 6ef931e):
  `$R/draw_z_cells.csv`, and the "Per cell over the draws" part of
  `/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/slurm-6457035.log`; to be read.
- **Why:** in PUMA 4701503 (Baseline, alpha 0.01), the least diverse HMC
  reference draw (draw 1457 of 4,000) puts 72 women aged 70-75 in block group
  470930038014, where the published count is 0 (modelled SE 8.5): z = +8.5 on a
  constraint cell, 99% of it from one record (SERIALNO 2020HU1090019, 55
  copies). Even the largest of the ~39 million (draw, block group, cell) |z|
  values would be about 5.9 under normal tails, so this is either a heavy
  right tail (weights are exponential in the multipliers) or something else.
- **Running:** `experiments/draw_z.py`, slurm job 6456752, writing
  `$R/draw_z.csv`; summary in `/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/slurm-6456752.log`.
- **Answer from:** per experiment, alpha, level (block group, tract), kind
  (constraint, held out) and published zero vs. nonzero: the share of draws
  with a cell over |z| 5.9, and the distribution of each draw's largest |z|.
  Whether zero cells (modelled SE) or counts drive the tail; whether held-out
  cells behave like constraint cells.
- **Result (job 6456752, `$R/draw_z.csv`):** essentially every draw has cells
  over |z| 5.9 in every group (100%; tract zero cells 98-99.7% in Null space
  and Roll-up); the median draw's largest |z| is about 11-47, many such cells
  per draw. Alpha only shrinks correlations (each cell's model variance is its
  published v at every alpha), so this is misfit on the model's own scale.
  Nonzero cells exceed zero cells in every row (e.g. Baseline, alpha 0.01,
  block group constraints: median largest |z| 22.8 against 15.1); held-out
  exceed constraint cells at block group; alpha 1 mostly worse than 0.01.
- **Reading (not verified):** most large |z| look like cells misfit in every
  draw, not excursions: small published counts carry tiny replicate SEs
  (variance 2-3 for counts of 1-2, `floored_variances`), and the runs use no
  floor, so a miss of tens of people is z of 10-20. The 4701503 draw's z = +8.5
  stands out only because the typical draw fits that cell (median 1.0).
- **Running:** per cell, over all draws: how often |z| > 5.9, its median z,
  published value and SE; persistent misfits (median |z| > 5.9) against
  excursions per group, and the most persistent misfits listed. That separates
  persistent misfit from excursions and tests the small-SE reading.

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
