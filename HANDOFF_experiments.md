# Handoff: the paper's final experiments

**Everything here runs from scratch, in clean folders.** A new data directory
(inputs and tables rebuilt from the downloads) and a new results root. Nothing
reads the earlier runs under `pmedm_vb_runs/<jobid>`, `rollup50_grid`,
`nullspace_grid`, `cap_4701502` or the old `scores/`; they stay as they are,
for reference only.

Two drivers do the work:

- `experiments/run_experiment.py EXP.json`: submits the fits of one experiment
  (raking, MAP + VB, HMC) into `<root>/<name>/`.
- `experiments/score_experiment.py <root>/<name> --tag sNN`: scores them into
  `<root>/<name>/scores/sNN/`, a new folder for each scoring, so the scoring
  can be redone as its code changes without touching the fits.

## Fixed for every experiment

Study area Knox County, ACS 2020–2024 5-year (`knox-2024-5yr`), PUMAs 4701501–4701504,
alphas 1, 0.1, 0.01. The model:

- PUMA rows (`--hierarchy puma`), untapered (`--taper none`), the zero-count
  variance floor (`--variance-floor zero`);
- methods: IPF and Sinkhorn raking; MAP + Laplace; VB Gaussian and skewed (64
  draws per step); HMC short (200 warmup + 1,000 samples, thin 10) and
  reference (2,000 + 10,000, thin 40, checkpoint every 500); 32 chains,
  trajectory 3.0, whitened by the skewed VB fit;
- left out: VB sum-difference, `--collapse`.

HMC seeds: `seed + 10 × (PUMA index) + (alpha index)`, from 2000 (short) and
1000 (reference), the same in every experiment.

## The experiments

The JSON files are in `experiments/paper/`.

| name | differs from the baseline |
|---|---|
| `exp01_baseline` | nothing |
| `exp02_nullspace` | `--hierarchy nullspace`: also removes every consistent direction the data cannot see |
| `exp03_rollup` | roll-up at 50 persons / 20 households (`--rollup 50h20`, level `puma-r50h20`); raking targets the rolled-up cells |
| `exp04_cap1000x1` | ratio cap at 1,000, strength 1 (level `puma+cap1000x1`) |
| `exp05_cap1000x10` | ratio cap at 1,000, strength 10 |
| `exp06_cap5000x1` | ratio cap at 5,000, strength 1 |
| `exp07_cap5000x10` | ratio cap at 5,000, strength 10 |

The caps have fixed strength and are applied to the plain `puma` level, with no
roll-up and no null space. Raking does not depend on the hierarchy or the cap, so
its results are identical in every experiment except exp03. Each experiment
still rakes for itself, which keeps each folder self-contained.

## Folder layout

```
<root>/                                  default /lustre/isaac24/proj/UTK0496/pmedm_vb_runs/paper
  expNN_<name>/
    experiment.json                      the JSON + commit, data directory, level, start time
    jobs.jsonl                           every job submitted: id, stage, PUMA, commit, command
    logs/slurm-<jobid>.log
    <puma>/
      ipf/  sinkhorn/                    raking
      map/  vb_gaussian/  vb_skewed/     every alpha in each (+ vb_<family>.log)
      hmc_short/  hmc_ref/               every alpha in each; hmc_spec.txt beside them
    scores/
      s01/
        scoring.json                     commit, date, draws, jobs, what was missing
        specs/<puma>_a<alpha>.txt        the compare_methods.py calls
        per_cell/<puma>_a<alpha>.csv, <puma>_a<alpha>_tables.csv
        all_scores.csv  all_tables.csv
        score_summary.txt  table_summary.txt  distribution_summary.txt
        calibration_summary.txt  missing.txt
        logs/
```

Result files are named as before: `<puma>_none_a<alpha>_h<level>.npz` (MAP, VB),
`..._trace.npz` / `..._report.txt` (HMC), and `<puma>_<method>[_r50h20].npz` (raking).

## Step 0: code and folders

On a login node, from the repository at `/lustre/isaac24/proj/UTK0496/pmedm_vb`
(the job scripts run the code from there):

```bash
cd /lustre/isaac24/proj/UTK0496/pmedm_vb
git fetch origin claude/practical-volta-ybxa07
git checkout claude/practical-volta-ybxa07 && git pull
git status            # should be clean: experiment.json records the commit (with -dirty if not)
conda activate /lustre/isaac24/proj/UTK0496/envs/pmedm_vb
PY=$CONDA_PREFIX/bin/python

export PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data   # the JSONs' data_dir
mkdir -p $PMEDM_VB_DATA /lustre/isaac24/proj/UTK0496/pmedm_vb_runs/paper
```

`data_dir` in every JSON is `/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data`.
If you change it, change it in all seven files before the first submission. A
folder refuses a JSON that differs from the one it was started with.

Optional, to skip downloading again: the downloads in `raw/` are never edited in
place, so hard links to the old cache are safe and take no space on the same
file system. Leave out `interim/` and `processed/`, which are rebuilt:

```bash
cp -al /lustre/isaac24/scratch/$USER/pmedm_vb_data/raw $PMEDM_VB_DATA/raw
```

## Step 1: rebuild the inputs and tables

Steps 1a and 1d need the network, so run them on a login node. The rest run as jobs.

```bash
# a. download what the cache lacks (login node; compute nodes are offline)
env -u PMEDM_VB_OFFLINE $PY experiments/run_map.py prefetch

# b. assemble every PUMA from scratch, then c. the held-out tables
a=$(sbatch --parsable --export=ALL,PMEDM_VB_DATA=$PMEDM_VB_DATA \
      experiments/run_map.sbatch assemble --rebuild)
sbatch --dependency=afterok:$a --export=ALL,PMEDM_VB_DATA=$PMEDM_VB_DATA \
      experiments/run_map.sbatch heldout --rebuild

# d. after both finish: the table trees from the national files, checked on Knox
#    and written into each PUMA's inputs (login node: it downloads the US files)
env -u PMEDM_VB_OFFLINE $PY experiments/table_trees.py \
      --write-inputs knox-2024-5yr --out $PMEDM_VB_DATA/table_trees
```

Step 1d refuses to write if any table fails its check on Knox. Read
`$PMEDM_VB_DATA/table_trees/` before going on. Only exp03 needs the trees, and
`run_experiment.py` will not submit exp03 without them.

Optional record of what the roll-up does, cells before and after, per table:

```bash
sbatch --export=ALL,PMEDM_VB_DATA=$PMEDM_VB_DATA experiments/run_python.sbatch rollup_report.py \
      --puma 4701501 4701502 4701503 4701504 --rollup 50h20 \
      --out /lustre/isaac24/proj/UTK0496/pmedm_vb_runs/paper/exp03_rollup/rollup_report
```

## Step 2: run the experiments

For each experiment, on a login node:

```bash
$PY experiments/run_experiment.py experiments/paper/exp01_baseline.json --dry-run   # look first
$PY experiments/run_experiment.py experiments/paper/exp01_baseline.json
$PY experiments/run_experiment.py experiments/paper/exp01_baseline.json --status
```

Each call submits, per PUMA, whatever is neither finished nor already queued or
running:

| stage | job | partition | outputs |
|---|---|---|---|
| rake | `run_map.sbatch rake`, one per (PUMA, method) | short, 1 h, 4 CPUs | `ipf/`, `sinkhorn/` |
| fits | `run_map_bundle.sbatch`, one per PUMA: MAP, then VB Gaussian and skewed, every alpha | short, 1 h, 48 CPUs | `map/`, `vb_*/` |
| hmc | `run_mcmc_bundle.sbatch`, one per PUMA: three short runs, then three references | campus-gpu, 24 h | `hmc_short/`, `hmc_ref/` |

So the way to run an experiment is to **call the same command again** until
`--status` shows every output:

- **HMC jobs wait on the fits.** An HMC job depends (`afterok`) on its PUMA's
  fits job while that job is live. If the fits job fails, the HMC job stays
  pending (DependencyNeverSatisfied): cancel it, then call again.
- **The GPU limit.** campus-gpu takes 6 submitted jobs per user, so the driver
  submits only `--max-gpu-jobs` (default 6) less what you already have there.
  The other PUMAs are submitted on later calls. 7 experiments × 4 PUMAs make 28
  HMC jobs in all.
- **Jobs that run out of time resume.** A rake or fits job that times out leaves
  its finished fits, and the next call resubmits only what is missing. An HMC
  bundle that times out resumes from its checkpoints when it is resubmitted.
- **Nothing is checked yet.** None of these time limits has been checked against
  these settings. Read the first rake, fits and HMC jobs' logs before
  submitting the rest. To change a limit or the CPUs, edit `RAKE_SLURM`,
  `FITS_SLURM` or `HMC_SLURM` at the top of `run_experiment.py`.

`--stage rake fits` (or `hmc`) limits a call to some stages. A sensible order is
to submit rake and fits for all seven experiments first (CPU, 56 + 28 jobs), then
call again with `--stage hmc` as the GPU queue frees up.

## Step 3: score

When an experiment's `--status` is complete:

```bash
$PY experiments/score_experiment.py /lustre/isaac24/proj/UTK0496/pmedm_vb_runs/paper/exp01_baseline \
      --tag s01 --dry-run
$PY experiments/score_experiment.py /lustre/isaac24/proj/UTK0496/pmedm_vb_runs/paper/exp01_baseline --tag s01
```

This submits:

- one `compare_methods.sbatch` job per (PUMA, alpha). Each job scores, in order,
  the HMC reference, short HMC, MAP + Laplace, both VB families and both raking
  fits. Every one is scored against that cell's HMC reference, and the HMC runs
  are also scored with the skewed VB fit that whitened them. Each job uses
  4,000 draws.
- one combine job after them (`afterany`). It concatenates the cells' results
  into `all_scores.csv` and `all_tables.csv`, writes the four `score_summary.py`
  views, and lists in `missing.txt` any (method, PUMA, alpha) without scores.

It submits nothing if any result is missing, unless you pass
`--allow-missing`. A cell without its HMC reference is left out whole. It also
refuses a tag that already exists. When the scoring code changes, score again
under the next tag (`s02`, ...); `scoring.json` records the commit of each.

## Not yet done

- No cross-experiment folder: comparing experiments is the `all_*.csv` of
  each, read side by side (in R, or a later `compare/` script).
- The cube and TRS integerisation comparison is on hold.
