#!/bin/bash
#SBATCH --job-name=embed-fit
#SBATCH --account=isaac-utk0533
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --gpus=v100s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/isaac24/proj/UTK0533/pmedm_vb_runs/embed/slurm-%j.log

# Fit the encoder-free baseline on one GPU; arguments go to fit_baseline.py:
#
#   sbatch experiments/embed/fit_baseline.sbatch --config experiments/embed/stage1_v2.json \
#       --out /lustre/isaac24/proj/UTK0533/pmedm_vb_runs/embed/baseline --d 8 --gamma 0 0.5 1 \
#       --ridge 1e-4 1e-3 1e-2
#
# Completed fits are skipped, so a timed-out job can be resubmitted as it was.
# PMEDM_VB_REPO, PMEDM_VB_ENV and PMEDM_VB_DATA override the defaults below.

set -eo pipefail
REPO=${PMEDM_VB_REPO:-/lustre/isaac24/proj/UTK0533/pmedm_vb_embed}

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate "${PMEDM_VB_ENV:-/lustre/isaac24/proj/UTK0533/envs/pmedm_vb}"
# After activation, and essential: the environment's editable install points at
# the other clone, so the package must come from this one's src/.
export PYTHONPATH=$REPO/src
set -u
export PMEDM_VB_DATA=${PMEDM_VB_DATA:-/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data}
export PMEDM_VB_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

PY=$CONDA_PREFIX/bin/python
case "$($PY -c 'import sys; print(sys.executable)')" in
    "$CONDA_PREFIX"/*) ;;
    *) echo "ERROR: python is not the pmedm_vb environment's" >&2; exit 1 ;;
esac
case "$($PY -c 'import pmedm_vb; print(pmedm_vb.__file__)')" in
    "$REPO"/src/*) ;;
    *) echo "ERROR: pmedm_vb is not imported from $REPO/src" >&2; exit 1 ;;
esac

cd "$REPO"
echo "node $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), commit $(git rev-parse --short HEAD), $(date)"
$PY -u experiments/embed/fit_baseline.py --device cuda "$@"
echo "done $(date)"
