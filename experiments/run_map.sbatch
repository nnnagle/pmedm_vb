#!/bin/bash
#SBATCH --job-name=pmedm-map
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
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
# campus/campus is the standard CPU partition. The four-hour limit here is a
# placeholder until the timing test sets it; override at submission with
# sbatch --time=HH:MM:SS. Resources too, e.g. sbatch --cpus-per-task=48.
#
# The first argument picks the step: assemble, heldout, solve (MAP, the
# default), vb or rake. assemble and heldout run only when asked for, so fit
# jobs never race to build the same inputs; run them once, first:
#
#   sbatch experiments/run_map.sbatch assemble --rebuild
#   sbatch experiments/run_map.sbatch heldout
#
# One fit per job, for clean timing (see run_map.py):
#
#   sbatch experiments/run_map.sbatch vb --puma 4701501 --alpha 0.01 --taper tract \
#       --family skewed --draws 64 --variance-floor zero --seed 3000
#   sbatch experiments/run_map.sbatch rake --puma 4701501 --method ipf
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
# --epsilon E builds and solves the statewide-support problems instead, in their
# own inputs directory (see run_map.py), e.g.
#
#   sbatch experiments/run_map.sbatch --epsilon 0.01 --alpha 1.0 --taper tract
#
# Assembled problems are reused across jobs; after changing the constraint
# tables or assembly code, run the assemble step with --rebuild.

# -u (unset variables are errors) is switched on only after conda activates:
# the environment's activate.d hooks read variables such as
# MKL_INTERFACE_LAYER that are normally unset, and abort under -u.
set -eo pipefail

REPO=/lustre/isaac24/proj/UTK0496/pmedm_vb
RUNS=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs
OUT=$RUNS/${SLURM_JOB_ID}
STEP=solve
case "${1:-}" in
    assemble|heldout|solve|vb|rake) STEP=$1; shift ;;
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
case "$STEP" in
    assemble|heldout)
        $PY -u experiments/run_map.py "$STEP" "$@"
        ;;
    *)
        $PY -u experiments/run_map.py "$STEP" --out "$OUT" "$@"
        cp "$OUT/summary.csv" "$RUNS/summary-${STEP}-${SLURM_JOB_ID}.csv"
        ;;
esac
echo "done $(date)"
