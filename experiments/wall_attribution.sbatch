#!/bin/bash
#SBATCH --job-name=pmedm-walls
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

# Wall attribution over several VB fits in one CPU job -- see
# experiments/wall_attribution.py. Arguments go straight to the script, e.g.
#
#   sbatch experiments/wall_attribution.sbatch --variance-floor zero \
#       --fit 4701502 1.0 $RUNS/6277113 --fit 4701502 0.1 $RUNS/6277117 \
#       --out $RUNS/wall_attribution

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
$PY -u experiments/wall_attribution.py "$@"
echo "done $(date)"
