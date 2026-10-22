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

## Answered

(none yet)
