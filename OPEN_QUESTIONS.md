# Open questions

Questions under investigation, so that several can be worked on at once. Each
entry: the question, why it came up, what is running or needed, and what would
answer it. Move an entry to **Answered** with its answer when done.

## Open

### 1. How heavy are the tails of single HMC draws against the published tables?

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

### 2. Why do n_eff and D/H rank IPF against the posterior differently?

- **Why:** at alpha 0.01, IPF's smallest block-group n_eff (about 12) is above
  the HMC reference's median (about 9), but its smallest D/H (about 0.06) is
  below HMC's median (about 0.11) (Figure B, n_eff and D/H versions). Likewise
  VB's typical draw has a low n_eff but a D/H like HMC's.
- **Reading so far (from the definitions, not checked):** n_eff = 1 / sum s^2 is
  set by the largest shares; D/H counts records with a share around 1/H or
  more, each about 1 however large. So IPF would avoid very large single
  shares but leave few records with a meaningful share; VB's typical draw
  would have one dominant record but many others in use.
- **Needed:** in a block group or two (e.g. the IPF fit's smallest D/H in each
  PUMA), the distribution of shares under IPF and under a few HMC draws: the
  largest shares, how many records are over 1/H, 0.1/H and 0.01/H, and how much
  of the households the records under 1/H hold together.
- **Answer from:** whether the two measures pick out different features
  (concentration against breadth) and which one each figure and table should
  report.

## Answered

(none yet)
