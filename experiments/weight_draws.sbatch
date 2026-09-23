#!/bin/bash
#SBATCH --job-name=pmedm-wdraws
#SBATCH --account=acf-utk0011
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# How often VB posterior draws give one household an implausible weight, and
# which households -- see experiments/weight_draws.py. Every (PUMA, taper,
# alpha) runs in its own process; reports go to <run>/weight_draws/ as
# <puma>_<taper>_a<alpha>.txt plus a _households.csv. For example
#
#   RUNS=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs
#   sbatch experiments/weight_draws.sbatch --alpha 1.0 0.1 --run $RUNS/<vb jobid>
#
# --alpha and --run are required. --puma defaults to every assembled PUMA,
# --taper to tract and --draws to 4000.

# -u only after conda activates; see run_map.sbatch.
set -eo pipefail

REPO=/lustre/isaac24/proj/UTK0496/pmedm_vb

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /lustre/isaac24/proj/UTK0496/envs/pmedm_vb
set -u
export PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_data
export PMEDM_VB_OFFLINE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

PY=$CONDA_PREFIX/bin/python
case "$($PY -c 'import sys; print(sys.executable)')" in
    "$CONDA_PREFIX"/*) ;;
    *) echo "ERROR: python is not the pmedm_vb environment's" >&2; exit 1 ;;
esac

cd "$REPO"
echo "node $(hostname), ${SLURM_CPUS_PER_TASK} cores, commit $(git rev-parse --short HEAD), $(date)"
$PY -u experiments/weight_draws.py "$@"
echo "done $(date)"
