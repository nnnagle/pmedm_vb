#!/bin/bash
#SBATCH --job-name=pmedm-gpu-py
#SBATCH --account=isaac-utk0496
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --gpus=v100s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# Run one experiments/ script on a V100 (see run_mcmc.sbatch on why the type is pinned):
#
#   sbatch --export=ALL,PMEDM_VB_DATA=$PMEDM_VB_DATA experiments/run_gpu_python.sbatch \
#       bench_chains.py --pumas 4701501 --alpha 0.01 --vb-root $PAPER/exp01_baseline ...
#
# The first argument is the script under experiments/; the rest go to it.

set -eo pipefail
REPO=/lustre/isaac24/proj/UTK0496/pmedm_vb

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /lustre/isaac24/proj/UTK0496/envs/pmedm_vb
set -u
export PMEDM_VB_DATA=${PMEDM_VB_DATA:-/lustre/isaac24/scratch/$USER/pmedm_vb_data}
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
echo "node $(hostname), commit $(git rev-parse --short HEAD), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
SCRIPT=${1:?usage: sbatch run_gpu_python.sbatch SCRIPT.py [args]}; shift
$PY -u "experiments/$SCRIPT" "$@"
echo "done $(date)"
