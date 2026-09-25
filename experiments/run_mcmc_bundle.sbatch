#!/bin/bash
#SBATCH --job-name=pmedm-mcmc-bundle
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus-gpu
#SBATCH --qos=campus-gpu
#SBATCH --gpus=v100s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# Several HMC runs in one GPU job, one after another -- for when the campus-gpu
# QoS (6 jobs submitted, 3 running per user) will not take one job per run.
#
#   sbatch experiments/run_mcmc_bundle.sbatch SPECFILE
#
# Each non-blank, non-# line of SPECFILE is one run: an output directory, then
# the arguments to experiments/run_mcmc.py (without the step and --out), e.g.
#
#   /lustre/.../pmedm_vb_runs/hmc/4701502_a1_ref  --puma 4701502 --alpha 1.0 ...
#
# Each run does sample, then report, exactly as run_mcmc.sbatch does. The
# sampler checkpoints and resumes, so a bundle that hits its time limit is
# finished by resubmitting the same SPECFILE: completed runs only redo their
# report. Each run's wall time is in its own trace (seconds, warmup_seconds),
# not this job's elapsed time. A run that fails does not stop the rest.

set -eo pipefail
SPEC=${1:?usage: sbatch run_mcmc_bundle.sbatch SPECFILE}
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
echo "node $(hostname), commit $(git rev-parse --short HEAD), $(date); spec $SPEC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"

failed=0
while read -r out rest; do
    [ -z "${out:-}" ] && continue
    case "$out" in \#*) continue ;; esac
    mkdir -p "$out"
    echo "=== $(date +%T) run -> $out"
    if $PY -u experiments/run_mcmc.py sample $rest --out "$out" < /dev/null \
        && $PY -u experiments/run_mcmc.py report $rest --out "$out" < /dev/null > /dev/null; then
        echo "=== $(date +%T) done $out"
    else
        echo "=== $(date +%T) FAILED $out"; failed=$((failed + 1))
    fi
done < "$SPEC"
echo "done $(date); $failed run(s) failed"
[ "$failed" -eq 0 ]
