#!/bin/bash
#SBATCH --job-name=pmedm-puma-bundle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# Many PUMAs in one job, one per GPU at a time: MAP, the VB fits and their HMC
# runs (experiments/run_puma_bundle.py). Submitted by run_experiment.py when the
# experiment has a "bundle" block, with the account, partition, GPUs, cores,
# memory and time from its slurm.bundle options -- or, for a pilot
# (run_experiment.py --pilot), from slurm.pilot:
#
#   sbatch --account=... --partition=... --gpus=4 --cpus-per-task=64 ... \
#       --signal=B:USR1@600 experiments/run_puma_bundle.sbatch BUNDLE.json
#
# Resubmitting the same file finishes a bundle that ran out of time. The last
# line exec-s Python, so that the batch shell's USR1 (--signal=B:...) and the
# time limit's SIGTERM reach the driver itself, which then stops taking PUMAs
# and releases its claims.

set -eo pipefail
BUNDLE=${1:?usage: sbatch run_puma_bundle.sbatch BUNDLE.json}
# PMEDM_VB_REPO and PMEDM_VB_ENV (environment) replace the UTK0496 defaults.
REPO=${PMEDM_VB_REPO:-/lustre/isaac24/proj/UTK0496/pmedm_vb}

module purge
unset PYTHONPATH  # a login profile's PYTHONPATH would shadow the environment's packages
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate "${PMEDM_VB_ENV:-/lustre/isaac24/proj/UTK0496/envs/pmedm_vb}"
set -u
# A caller (experiments/run_experiment.py) may point the job at another data directory.
export PMEDM_VB_DATA=${PMEDM_VB_DATA:-/lustre/isaac24/scratch/$USER/pmedm_vb_data}
export PMEDM_VB_OFFLINE=1

PY=$CONDA_PREFIX/bin/python
case "$($PY -c 'import sys; print(sys.executable)')" in
    "$CONDA_PREFIX"/*) ;;
    *) echo "ERROR: python is not the pmedm_vb environment's" >&2; exit 1 ;;
esac

cd "$REPO"
echo "node $(hostname), ${SLURM_CPUS_PER_TASK:-?} cores, GPUs ${CUDA_VISIBLE_DEVICES:-none}, commit $(git rev-parse --short HEAD), $(date)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    $PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
fi
exec $PY -u experiments/run_puma_bundle.py "$BUNDLE"
