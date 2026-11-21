#!/bin/bash
#SBATCH --job-name=latent-s4
#SBATCH --account=isaac-utk0533
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/lustre/isaac24/proj/UTK0533/pmedm_vb_runs/latent/slurm-%j.log

# Step 4 of the latent-population PMEDM (stage 2 posterior and its calibration)
# on CPU; arguments go to experiments/latent/step4_posterior.py:
#
#   R=/lustre/isaac24/proj/UTK0533/pmedm_vb_runs
#   sbatch experiments/latent/step4_posterior.sbatch \
#       --config experiments/embed/stage1_v2.json \
#       --regions experiments/latent/regions_tn.json \
#       --run $R/embed/stage1_v2/d16_g0.5_r0.01 --step1 $R/latent/step1_sigma \
#       --step3 $R/latent/step3 --step3b $R/latent/step3b \
#       --out $R/latent/step4 --workers 16 --threads 1
#
# Finished units (simulation x region x replicate) are skipped, so a timed-out
# job can be resubmitted unchanged. The output directory for the log
# (pmedm_vb_runs/latent) must exist before submitting.
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
# Several processes (--workers) of a few threads each (--threads): BLAS gets
# the same number of threads as the engine's sparse products.
THREADS=1
for ((i = 1; i <= $#; i++)); do
    if [ "${!i}" = "--threads" ]; then j=$((i + 1)); THREADS=${!j}; fi
done
export OMP_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS

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
echo "node $(hostname), ${SLURM_CPUS_PER_TASK} cores, $THREADS threads per process, commit $(git rev-parse --short HEAD), $(date)"
lscpu | grep -E "^Model name|^CPU\(s\)" || true
$PY -u experiments/latent/step4_posterior.py "$@"
echo "done $(date)"
