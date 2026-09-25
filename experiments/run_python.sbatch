#!/bin/bash
#SBATCH --job-name=pmedm-py
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# Run one experiments/ script on a CPU node:
#
#   sbatch experiments/run_python.sbatch wall_layers.py --puma 4701502 --alpha 1.0 ...
#
# The first argument is the script under experiments/; the rest go to it.

set -eo pipefail
REPO=/lustre/isaac24/proj/UTK0496/pmedm_vb

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /lustre/isaac24/proj/UTK0496/envs/pmedm_vb
set -u
export PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_data
export PMEDM_VB_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

PY=$CONDA_PREFIX/bin/python
case "$($PY -c 'import sys; print(sys.executable)')" in
    "$CONDA_PREFIX"/*) ;;
    *) echo "ERROR: python is not the pmedm_vb environment's" >&2; exit 1 ;;
esac

cd "$REPO"
echo "node $(hostname), ${SLURM_CPUS_PER_TASK} cores, commit $(git rev-parse --short HEAD), $(date)"
SCRIPT=${1:?usage: sbatch run_python.sbatch SCRIPT.py [args]}; shift
$PY -u "experiments/$SCRIPT" "$@"
echo "done $(date)"
