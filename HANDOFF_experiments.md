# Handoff: the paper's experiments, from the downloads to the tables

Branch `claude/beautiful-goldberg-jqcrwy`. This replaces the earlier version of
this file. `HANDOFF_paper.md` is older background on the paper; where the two
disagree (for example its "CPU fits on 18 cores"), this file is current.

## 1. Status

- **exp01_baseline** (4 PUMAs x alpha 1, 0.1, 0.01), **exp02_nullspace** and
  **exp03_rollup** are fitted, scored (tag `paper1`, with per-cell detail files)
  and combined on ISAAC. exp02 and exp03 were fitted on the whole grid, but the
  paper now uses them at alpha 1 and 0.01 only (section 3).
- The final tables have been produced per experiment in the old layout. The
  final layout (two table sets, IPF in a no-alpha panel, Sinkhorn left out) has
  **not yet been run on ISAAC**: `run_paper.py tick` will make it (section 4).
- `run_paper.py` (the one-command pipeline) is tested locally against fake
  `sbatch`/`squeue` and fake outputs, stage by stage to the end, but **not yet
  on ISAAC**.
- exp04-07 JSONs in `experiments/paper/` (a share cap on w/N) are **not part of
  the paper** and are not run by the pipeline (section 9).

## 2. Setup on ISAAC

```bash
cd /lustre/isaac24/proj/UTK0496/pmedm_vb
git fetch origin claude/beautiful-goldberg-jqcrwy
git checkout claude/beautiful-goldberg-jqcrwy && git pull
conda activate /lustre/isaac24/proj/UTK0496/envs/pmedm_vb
PY=$CONDA_PREFIX/bin/python
export PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data
PAPER=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/paper
```

A new login loses these variables (an empty `$PAPER` gives paths like
`/exp01_baseline/...`); putting the last three lines in `~/.bashrc` avoids it.
Jobs run the code from the repository on Lustre, so `git pull` before
submitting.

| what | where |
|---|---|
| data directory (downloads, assembled inputs, table trees) | `/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data` |
| results root | `/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/paper` |
| one experiment | `<root>/expNN_<name>/` |
| the driver's own jobs and logs | `<root>/run_paper/` |
| tables | `<root>/tables/`, `<root>/tables_compare/` |

## 3. The experiments

Study area Knox County, ACS 2020-2024 5-year (`knox-2024-5yr`), PUMAs
4701501-4701504. Common model: PUMA rows (`--hierarchy puma`), untapered Sigma
(`--taper none`), zero-count variance floor, all 46 B01001 sex-by-age categories
(`age_sex()` now defaults to every published break; the old collapse is
`age_sex(b01001_boundaries())`).

| experiment | differs from the baseline | run on |
|---|---|---|
| exp01_baseline | nothing | 4 PUMAs x alpha 1, 0.1, 0.01 |
| exp02_nullspace | `--hierarchy nullspace` (removes every consistent direction the data cannot see) | 4 PUMAs x alpha 1, 0.01 |
| exp03_rollup | roll-up at 50 persons / 20 households (`--rollup 50h20`); raking targets the rolled-up cells; **scored on the original cells** | 4 PUMAs x alpha 1, 0.01 |

**Subsets.** An experiment JSON's `pumas`/`alphas` are its grid; an optional
`"subset": {"pumas": [...], "alphas": [...]}` narrows what is run, scored and
tabled. exp02 and exp03 have `"subset": {"alphas": [1.0, 0.01]}`. The grid still
identifies the experiment (a folder fitted on the whole grid is reused, not
refused) and fixes the HMC seeds (`seed + 10 x PUMA index + alpha index` in the
grid), so a subset run uses the same seeds.

**alpha.** `Sigma(alpha) = D^1/2 [(1-alpha) R + alpha I] D^1/2`, R the
replicate-estimated correlation: alpha only shrinks the correlations; alpha = 1
is classic diagonal PMEDM; alpha cannot reach 0 (80 replicates leave R
rank-deficient), so 0.01 is the smallest used.

**Methods** per (PUMA, alpha): IPF and Sinkhorn raking (no alpha: one fit per
PUMA); MAP + Laplace; VB Gaussian and skewed (64 draws per step); both VB
families truncated (`vb_<family>_trunc`, scoring only); short HMC (200 warmup +
1,000 samples, thin 10) and reference HMC (2,000 + 10,000, thin 40), 32 chains,
integration time 3.0, whitened by and started from the skewed VB fit.

## 4. The workflow: one command

`experiments/run_paper.py` runs everything for exp01-03 and resumes from any
point.

```bash
$PY experiments/run_paper.py start     # login node: downloads, first pass, starts the driver job
$PY experiments/run_paper.py status    # anywhere: where every stage stands
$PY experiments/run_paper.py tick      # one pass now, and schedule the driver (no downloads)
$PY experiments/run_paper.py advance --dry-run   # what one pass would submit
```

Options: `--root`, `--data-dir` (new folders = from scratch; existing ones =
resume), `--tag paper1`, `--draws 4000`, `--every 30` (minutes between passes),
`--max-gpu-jobs 6`, `--max-attempts 4`, `--dry-run`.

**Stages.** Each pass submits, per stage, only work that is due and has no job
queued or running; every stage is idempotent.

1. **prefetch** (`start` only; compute nodes are offline): every file assembly,
   held-out tables and table trees read, including the constraint tables'
   national (summary level 010) replicate files.
2. **assemble** (`run_map.sbatch assemble`), then **heldout**
   (`run_map.sbatch heldout`), then **trees** (`table_trees.py --write-inputs`,
   needed by the roll-up) into the data directory. Done when every PUMA has
   `manifest.json`, `heldout/`, `table_trees.json`.
3. **fits**, per experiment: `run_experiment.advance()` submits raking (short),
   MAP + VB bundles (short, 48 CPUs) and HMC bundles (campus-gpu, 24 h; afterok
   on the PUMA's fits), keeping the user's campus-gpu jobs at 6 or fewer.
   Done when every raking, MAP, VB and HMC output (HMC: `_report.txt`) exists.
4. **score**, per experiment once its fits are complete:
   `score_experiment.py --resume --no-combine` over the experiment's subset.
   Only calls without scores are submitted.
5. **combine**, once every call has scores and no scoring job is live.
   Done when `all_scores.csv` is newer than every cell's CSV.
6. **tables**: `tables/` (exp01 alone, whole grid) and `tables_compare/`
   (exp01-03 on the PUMAs and alphas they share: 4 PUMAs x alpha 1, 0.01, so the
   columns pool the same cells). Done when each `tables.md` is newer than the
   `all_scores.csv` files it reads.

**The driver job.** `start`/`tick` end by submitting `run_paper.sbatch` (short,
2 CPUs, 30 min), which runs one pass and resubmits itself `--every` minutes
later while work remains. When everything is done it submits a one-line job
whose only purpose is the "done" e-mail. A stage submitted `--max-attempts`
times without finishing stops the run: the pass prints `NEEDS ATTENTION`, the
driver job fails and Slurm mails the failure. Fix the cause, then `tick`. A
lock (`<root>/run_paper/lock`, stale after 2 h) keeps two drivers apart.

**Queue limits.** campus-gpu takes 6 submitted jobs per user; `short` also
limits submissions per user (this cut off a scoring submission once). A refused
`sbatch` ends that pass's submissions early; the next pass continues.

**Time.** From scratch, at least a day: the null-space HMC bundles (six runs in
sequence per PUMA) are the longest part.

**To produce the final tables now** (everything is already fitted and scored):
`git pull`, `run_paper.py status` (all stages done except the two table sets),
then `run_paper.py tick`.

## 5. The workflow by hand (what the driver calls)

For debugging one stage, or running part of it:

```bash
# data (prefetch on a login node; the rest as jobs)
env -u PMEDM_VB_OFFLINE $PY experiments/run_map.py prefetch
sbatch --export=ALL,PMEDM_VB_DATA=$PMEDM_VB_DATA experiments/run_map.sbatch assemble
sbatch --export=ALL,PMEDM_VB_DATA=$PMEDM_VB_DATA experiments/run_map.sbatch heldout
sbatch --export=ALL,PMEDM_VB_DATA=$PMEDM_VB_DATA experiments/run_python.sbatch table_trees.py \
    --write-inputs knox-2024-5yr --out $PMEDM_VB_DATA/table_trees
$PY experiments/check_inputs.py              # checks the assembly (46 B01001 cells, nesting, totals)

# fits (call again until --status shows every output)
$PY experiments/run_experiment.py experiments/paper/exp01_baseline.json [--dry-run|--status]

# scoring, combining (a tag is a folder under <exp>/scores/)
$PY experiments/score_experiment.py $PAPER/exp01_baseline --tag paper1 [--resume] [--alphas 1 0.01]
$PY experiments/score_experiment.py $PAPER/exp01_baseline --tag paper1 --combine

# tables (a job: it reads ~600 MB of parquet)
J=$(sbatch --parsable --partition=short --qos=short --time=01:00:00 \
  --export=ALL,PMEDM_VB_DATA=$PMEDM_VB_DATA --output=$PAPER/tables-%j.log \
  experiments/run_python.sbatch paper_tables.py --root $PAPER --tag paper1 \
    --experiments exp01_baseline exp02_nullspace exp03_rollup --alphas 1 0.01 \
    --out $PAPER/tables_compare)
```

`sbatch --parsable` prints only the job ID; capture it as above to find the log.

## 6. Scoring

`compare_methods.py` scores one method in one (PUMA, alpha) cell against that
cell's reference HMC; `score_experiment.py` runs one `compare_methods.sbatch`
job per cell (short), each calling every method in turn (`hmc_ref` first, which
caches the reference's summaries).

- **Outputs** under `<exp>/scores/<tag>/`: `scoring.json` (written before the
  first job and after each), `specs/`, `per_cell/<cell>.csv` (long: method,
  subset, metric, stat, value), `detail/<cell>/<method>_cells.parquet` (every
  scored cell's metrics) and `<method>_draws.parquet` (every draw's statistics),
  `all_scores.csv`, `score_summary.txt`, `missing.txt`. About 206 MB of detail
  per experiment.
- **Resuming.** `--resume` submits only calls whose method has no rows in the
  cell's CSV and skips cells with a live job. `--max-jobs`, `--no-combine`,
  `--pumas`, `--alphas`.
- **Per-draw statistics** include: largest cell share of N (walls: > 1% of N),
  household shares and Kish n_eff = 1/sum s^2 in block groups over 100
  published households, D = sum_i 1 - (1 - s_i)^H (expected distinct records
  among H households), and household consistency: the largest |z| of a block
  group's allocated occupied households (B25003 cells summed) against the
  published total, SE from the summed variances with the zero-count floor.
- **arviz race.** Jobs starting together could fail `import arviz` on its daily
  stamp file; `compare.import_arviz()` retries once (fixed).

## 7. The tables (`experiments/paper_tables.py`)

Rows: IPF, MAP + Laplace, VB Gaussian, VB skewed, both truncated, HMC short,
HMC reference. Sinkhorn is still fitted and scored but left out of the tables
(remove from `METHODS` to restore). One panel per alpha; IPF has no alpha and
is shown once in a "Raking (no alpha)" panel, except in Table 4, where it is
compared with each alpha's reference. Outputs: `tables.md`, one `.tex`
(booktabs) and one long `.csv` per table (unrounded, with `n`).

Pooling: every cell (or draw) of every PUMA together, so quantiles are exact
over the pooled cells. z and coverage over *sampled* cells only (nonzero
estimate with a sampling variance). k-hat (one per fit): median [min, max] over
PUMAs. Cost: minutes per PUMA, mean over PUMAs (county totals in the CSV).
`--pumas`/`--alphas` filter; the notes are worded for the PUMAs pooled.

| table | content |
|---|---|
| 1 Cost | fit, sampling, total minutes. HMC's fit is the skewed VB fit that whitens it (VB's time includes its MAP); sampling is HMC warmup + sampling |
| 2 (2t tract) | constrained block-group cells: posterior 90% interval contains published; \|z\| p90, z = (posterior mean - published)/SE; half-width / MOE, median |
| 3 Held-out | the same for held-out tables: related (B25044, C17002, C24030, B12001, B15002) and less related (B25040) |
| 4 Agreement with reference | median diff. in reference sds (p50, p99); 90% width ratio (p10, p50); PSIS k-hat |
| 5 Usability | draws with a wall; smallest n_eff (median); draws with n_eff < 10; draws with households \|z\| > 5; largest households \|z\| (median); IPF's shares as counts of its 4 draws |

**Coverage decisions.** Coverage is computed per cell once, from all draws:
does the 5th-95th percentile interval contain the published value; then
averaged over cells. It is *not* repeated-sampling coverage of a known truth.
Predictive coverage (adding SE^2 to sd^2) was dropped: in-sample it counts the
published SE twice (the posterior was fitted to that value), giving ~100% even
for HMC. In-sample (Table 2) coverage and |z| have no nominal level (normal
example: residual variance B SE^2, posterior variance (1-B) SE^2, B the
shrinkage weight). Held-out (Table 3) coverage below 0.90 is expected, since
the published value also carries sampling error (if sd = SE and the mean is
right, about 0.75); held-out tables share the ACS sample with the constraints,
so their errors may be correlated with them.

## 8. Findings so far

From the per-experiment tables (old layout, before this file's table changes):

- **Short HMC reproduces the reference** in every experiment (median diff. p99
  0.06-0.14, width ratio p10 0.88-0.95) at about a tenth of its cost.
- **MAP + Laplace is unusable**: walls in essentially every draw, collapsed
  intervals, k-hat infinite or huge.
- **VB is not a substitute for HMC's draws**: too narrow in its narrowest cells
  (p10 width ratio 0.19-0.60 null space, 0.45-0.89 roll-up), k-hat 3.8-14 (>>
  0.7), walls (a cell > 1% of N) in 3-43% of draws, and far less within-block-
  group variety than HMC. It matches HMC on typical-cell fit (Tables 2-3).
- **Truncated VB is kept, and is the cheap way to usable VB draws.** It is the
  same VB fit with the draws the reference never makes turned away (any cell
  over 1% of N, or any block group over 100 households with D/H < 0.1):
  `truncated_draws` in `compare_methods.py`, methods `vb_<family>_trunc`,
  options `--trunc-share 0.01`, `--trunc-distinct 0.1`, `--trunc-max-factor 20`;
  acceptance is reported (`trunc_acceptance`). It costs only extra draws (the
  fit is unchanged). It removes every wall, and on held-out and in-sample fit
  it matches HMC (exp01 s04, alpha 0.01: block-group |z| p90 0.705 untruncated,
  0.641 truncated, 0.640 HMC; acceptance 0.61 skewed). It cannot fix VB's
  shape: k-hat and the narrowest widths barely change, draws pile up against
  the cutoff, and within-block-group variety stays below HMC's. So: usable VB
  draws for point summaries and typical-cell uncertainty; HMC for the
  posterior's tails and variety.
- **alpha 0.01** (the replicate correlations) gives the best held-out accuracy
  in every experiment, but the least variety: reference draws with a block
  group under n_eff 10 - exp01 about 22% (diagnostic), null space 40%, roll-up
  20%.
- **Null space** fixes most of the leakage of households into empty block
  groups at alpha 1 (reference draws with households |z| > 5: 10%, against most
  draws in exp01) and the tightest fit at alpha 1; it is 2-3x the baseline's
  cost (skewed VB 37-40 min, reference HMC 2.6-5 h per PUMA).
- **Roll-up** is the cheapest (skewed VB 5-9 min, short HMC 8-16 min total per
  PUMA), gives a much better VB and more usable draws, at a small cost in
  fine-cell fit and held-out accuracy (less-related |z| p90 2.13-2.64 against
  2.00-2.25 null space).
- **On held-out point accuracy the posterior mean only matches IPF** (null
  space related |z| p90 1.52 vs IPF 1.54); the Bayesian methods add calibrated
  uncertainty and usable draws, not a better point estimate.
- **In-sample**, half-width / MOE ~ 1: each cell's own estimate dominates its
  posterior. The fit loosens at alpha 1 (exp01 |z| p90 1.48 vs 0.60 at 0.01).

## 9. What was tried and set aside

- **Walls.** VB draws where one (record x block group) cell takes > 1% of N;
  driven by published-zero cells in rare categories (zero-count SE 8.49), not by
  large households; local, not low-rank (`wall_subspace.py`); mostly
  over-allocation of a block group's total rather than crowding out. HMC never
  produces them.
- **Ratio cap, then share cap** (w/N, `--share-cap`, `sharecap.py`): planned as
  exp04-07; never run. Bounded penalties of p are not convex in lambda.
- **Changing the truncation rule** (n_eff floors instead of the wall and D/H
  rules, or truncating the HMC reference and short HMC as well, a posterior
  conditioned on usable draws): explored with `usability_diagnostic.py`, not
  adopted. Truncated VB itself stays as above.
- **D/H floor**: nearly redundant with the wall rule; D ignores dominance.
- **Usability rules** (`usability_diagnostic.py`): an n_eff >= 5 floor keeps
  99-100% of the posterior and catches 76-86% of walls; n_eff >= 10 binds on the
  posterior at alpha 0.01; a household-count |z| rule fails the posterior at
  alpha >= 0.1 (leakage into empty block groups), so it is a model diagnostic,
  not a truncation rule.
- **Dirichlet-form prior** kappa KL(d || s_b) (convex in lambda): rejected; its
  sum is dominated by legitimate exclusions (median block group ~5 nats at
  alpha 0.01) and does not separate wall draws. The diagnostic's KL section
  remains in `usability_diagnostic.py`.
- **Ideas not pursued**: hinge priors on n_eff and household |z| with a
  curvature check; tighter B25003 variances (convex); VB draws moved by
  annealed/SMC HMC (Neal 2001; Del Moral, Doucet & Jasra 2006), which would
  handle walls by reweighting; a representation layer (factor prior on lambda_B).

## 10. Diagnostic scripts added on this branch

`check_inputs.py` (assembly checks), `ratio_diagnostic.py` (largest cells,
drivers, n_eff/D), `wall_subspace.py` (are walls low-rank), `usability_diagnostic.py`
(n_eff and household-|z| rules, KL calibration; one job per PUMA, then
`--combine`), `paper_tables.py`, `run_paper.py` + `run_paper.sbatch`.

## 11. Computation, for the paper

- **MAP + VB**: one job per PUMA on 48 CPU cores of one `short` node. The three
  stand-alone MAP solves (one per alpha) run in parallel, 16 threads each; then
  the two VB families run concurrently, 24 cores each, each fitting its three
  alphas in parallel at 8 threads per fit. A VB fit: its own MAP (damped Newton:
  per-tract Cholesky + Woodbury), the Gaussian stage started at the Laplace
  approximation (Adam, 64 reparameterised draws per step, window-stall
  learning-rate halving, at most 1,000 steps), for skewed a sinh-arcsinh stage,
  ELBOs from 400 draws. CPU only. Whether a Slurm CPU on ISAAC is a core or a
  hyperthread is unchecked (`scontrol show node`).
- **HMC**: one NVIDIA V100S per job, 4 CPU cores; the 32 chains are advanced as
  one batch; a PUMA's six runs run in sequence in one job; checkpoints every 500
  iterations. Leapfrog steps per iteration = jittered integration time / step
  size (tuned by dual averaging toward acceptance 0.8), capped at 1,000.
- **Times (exp01, s04 medians over PUMAs)**: VB Gaussian 7-10 min, skewed 12-17
  min per fit including MAP; short HMC 5-16 min, reference 51 min - 2.6 h of GPU
  time, plus the skewed VB fit.

## 12. Open items

- Run `run_paper.py tick` on ISAAC and read `tables/` and `tables_compare/`.
- exp03 Sinkhorn: median diff. p99 ~22 reference sds, far above IPF's 2.5;
  check which cells (query in the exp03 detail files) before any Sinkhorn
  result is cited.
- IPF in PUMA 4701502 gives a block group a zero total (`max_bg_share`
  undefined there); cosmetic, scoring summary only.
- exp04-07 JSONs and the share-cap code are unused; delete or keep as a record.
- `usability_diagnostic.py`'s KL section documents a rejected idea.
- `HANDOFF_paper.md` predates this work (e.g. 18 cores).
