#!/bin/bash
#SBATCH --job-name=latent-s5p
#SBATCH --account=isaac-utk0533
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/isaac24/proj/UTK0533/pmedm_vb_runs/latent/slurm-%j.log

# Step 5, phase 2: the pilot of the realized-population sampler (numba) on
# simulation S, model version 5's cut form, replicate 0, at step 4c's chosen
# pairs; arguments go to experiments/latent/step5_pilot.py:
#
#   R=/lustre/isaac24/proj/UTK0533/pmedm_vb_runs
#   export PMEDM_VB_ENV=/lustre/isaac24/proj/UTK0533/envs/pmedm_vb_numba
#   sbatch experiments/latent/step5_pilot.sbatch \
#       --config experiments/embed/stage1_v2.json \
#       --regions experiments/latent/regions_tn.json \
#       --run $R/embed/stage1_v2/d16_g0.5_r0.01 --step1 $R/latent/step1_sigma \
#       --step3 $R/latent/step3 --step3b $R/latent/step3b --step4 $R/latent/step4 \
#       --step4c $R/latent/step4c --out $R/latent/step5_pilot --threads 16
#
# PMEDM_VB_ENV MUST point to the numba environment (pmedm_vb_numba: numba,
# arviz, torch); the default environment below has no numba, and the job
# stops if numba cannot be imported. sbatch passes the submitting shell's
# environment to the job, so the export above is enough.
#
# One process; the sampler's numba threads (--threads, default 16 here)
# should equal --cpus-per-task. NUMBA_NUM_THREADS (the most numba may use)
# and the BLAS threads of the unit's set-up (the MAP of U, the conditional
# PMEDM) are set from --threads below. Regions run one after another;
# finished regions (their pilot.json) are skipped, so a timed-out job can be
# resubmitted unchanged. The first run compiles the kernels (a minute or
# so) and caches them next to the source (src/pmedm_vb/latent/__pycache__).
# The output directory for the log (pmedm_vb_runs/latent) must exist before
# submitting.
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
THREADS=${SLURM_CPUS_PER_TASK:-16}
for ((i = 1; i <= $#; i++)); do
    if [ "${!i}" = "--threads" ]; then j=$((i + 1)); THREADS=${!j}; fi
done
export NUMBA_NUM_THREADS=$THREADS
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
if ! $PY -c 'import numba, arviz' 2>/dev/null; then
    echo "ERROR: numba or arviz missing: set PMEDM_VB_ENV to the numba environment" >&2
    exit 1
fi

cd "$REPO"
echo "node $(hostname), ${SLURM_CPUS_PER_TASK} cores, $THREADS numba threads, numba $($PY -c 'import numba; print(numba.__version__, numba.config.THREADING_LAYER)'), commit $(git rev-parse --short HEAD), $(date)"
lscpu | grep -E "^Model name|^CPU\(s\)" || true
# the "--threads" default of the script is 1: pass it through explicitly
case " $* " in
    *" --threads "*) $PY -u experiments/latent/step5_pilot.py "$@" ;;
    *) $PY -u experiments/latent/step5_pilot.py "$@" --threads "$THREADS" ;;
esac
echo "done $(date)"
