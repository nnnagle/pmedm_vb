#!/bin/bash
#SBATCH --job-name=pmedm-mcmc
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# HMC reference posterior for one fit, then its report -- see
# experiments/run_mcmc.py. Arguments go to both steps, e.g.
#
#   RUNS=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs
#   sbatch experiments/run_mcmc.sbatch --puma 4701502 --alpha 1.0 \
#       --variance-floor zero --vb-run $RUNS/<vb jobid>
#
# Results go to $RUNS/<this jobid>/mcmc unless --out is given. The sampler
# checkpoints, so a job that hits its time limit is finished by resubmitting
# with --out pointing at the first job's directory and the same settings.
#
# Keep the campus-gpu partition and QoS: they give a V100. Submitted to another
# partition (short, say) the job can land on a T4, which runs this float64
# sampler about 30x slower (runs 6273419 and 6275306).
#
# One GPU of any size will do: at 32 chains the largest tensor is tens of MB and
# the dense whitening matrix under 1 GB, so the 16 GB cards are fine.

# -u only after conda activates; see run_map.sbatch.
set -eo pipefail

REPO=/lustre/isaac24/proj/UTK0496/pmedm_vb
RUNS=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs
# An --out among the arguments (resuming an earlier job) replaces the default.
OUT=""
args=("$@")
for ((i = 0; i < ${#args[@]} - 1; i++)); do
    [ "${args[i]}" = "--out" ] && OUT=${args[i + 1]}
done
if [ -z "$OUT" ]; then
    OUT=$RUNS/${SLURM_JOB_ID}/mcmc
    set -- "$@" --out "$OUT"
fi

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
echo "node $(hostname), commit $(git rev-parse --short HEAD), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Fail fast if this environment's torch cannot see the GPU (a CPU-only build).
$PY -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
mkdir -p "$OUT"

$PY -u experiments/run_mcmc.py sample "$@"
$PY -u experiments/run_mcmc.py report "$@"
echo "done $(date)"
