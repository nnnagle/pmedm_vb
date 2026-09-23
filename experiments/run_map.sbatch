#!/bin/bash
#SBATCH --job-name=pmedm-map
#SBATCH --account=acf-utk0011
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# MAP or VB sweep for Knox County: assemble one problem per PUMA, then fit every
# (PUMA, taper, alpha). See experiments/run_map.py for what each step does.
#
# Compute nodes cannot reach census.gov, so download first, from a login or
# data transfer node (cheap to repeat; cached files are skipped):
#
#   $CONDA_PREFIX/bin/python experiments/run_map.py prefetch
#
# then, from the repository root:
#
#   mkdir -p /lustre/isaac24/proj/UTK0496/pmedm_vb_runs   # once; Slurm needs it for --output
#   sbatch experiments/run_map.sbatch            # MAP (the solve step)
#   sbatch experiments/run_map.sbatch vb         # VB, each fit started from MAP
#
# short/short caps a job at one hour. That is enough for the solves (about a
# minute each, run in parallel), but first-time assembly is still untimed. If a
# job runs out of time, resubmit: PUMAs already assembled are saved and skipped.
# A PUMA whose assembly alone takes over an hour needs a longer partition.
# Resources can be overridden at submission, e.g. sbatch --cpus-per-task=48.
# Arguments after the optional step are passed to it, e.g.
#
#   sbatch experiments/run_map.sbatch --alpha 1.0 0.9 0.8 0.7 0.5
#   sbatch experiments/run_map.sbatch vb --alpha 1.0 0.5 --max-iter 2000
#
# Each fit logs its iterations to $OUT/logs/<puma>_<taper>_a<alpha>.log; this
# job's own log has one line per finished fit. To finish a sweep that ran out
# of time, point a new job at the old results, which are skipped:
#
#   sbatch experiments/run_map.sbatch vb --out /lustre/isaac24/proj/UTK0496/pmedm_vb_runs/<jobid>
#
# Assembled problems are reused across jobs; after changing the constraint
# tables or assembly code, run assemble with --rebuild (edit ASSEMBLE_ARGS).

# -u (unset variables are errors) is switched on only after conda activates:
# the environment's activate.d hooks read variables such as
# MKL_INTERFACE_LAYER that are normally unset, and abort under -u.
set -eo pipefail

REPO=/lustre/isaac24/proj/UTK0496/pmedm_vb
RUNS=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs
OUT=$RUNS/${SLURM_JOB_ID}
ASSEMBLE_ARGS=""
STEP=solve
case "${1:-}" in
    solve|vb) STEP=$1; shift ;;
esac
# An --out among the arguments (resuming an earlier sweep) replaces the default.
args=("$@")
for ((i = 0; i < ${#args[@]} - 1; i++)); do
    [ "${args[i]}" = "--out" ] && OUT=${args[i + 1]}
done

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /lustre/isaac24/proj/UTK0496/envs/pmedm_vb
set -u
export PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_data

# No network here: a file the prefetch missed fails at once, by name.
export PMEDM_VB_OFFLINE=1

# One BLAS thread for this driver process; run_map.py sets each worker's
# thread count itself from --cpus-per-task and the number of solves.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=$CONDA_PREFIX/bin/python
case "$($PY -c 'import sys; print(sys.executable)')" in
    "$CONDA_PREFIX"/*) ;;
    *) echo "ERROR: python is not the pmedm_vb environment's" >&2; exit 1 ;;
esac

mkdir -p "$OUT"
cd "$REPO"
echo "node $(hostname), ${SLURM_CPUS_PER_TASK} cores, commit $(git rev-parse --short HEAD), $(date)"
echo "step $STEP, results -> $OUT"

# -u: unbuffered, so the log shows progress as it happens.
$PY -u experiments/run_map.py assemble $ASSEMBLE_ARGS
$PY -u experiments/run_map.py "$STEP" --out "$OUT" "$@"

cp "$OUT/summary.csv" "$RUNS/summary-${STEP}-${SLURM_JOB_ID}.csv"
echo "done $(date)"
