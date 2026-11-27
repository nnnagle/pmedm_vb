#!/bin/bash
#SBATCH --job-name=latent-s5r
#SBATCH --account=isaac-utk0533
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/isaac24/proj/UTK0533/pmedm_vb_runs/latent/slurm-%j.log

# Step 5, phase 3: the regional run of the realized-population sampler (arm
# C3; PLAN_latent_step5.md §4 item 3, §8c's settings, which are the script's
# defaults) on simulation S at step 4c's chosen pairs; arguments go to
# experiments/latent/step5_regional.py. One job per region, each running that
# region's 10 replicates one after another (13 jobs; they can run at the same
# time, one node each):
#
#   R=/lustre/isaac24/proj/UTK0533/pmedm_vb_runs
#   export PMEDM_VB_ENV=/lustre/isaac24/proj/UTK0533/envs/pmedm_vb_numba
#   for r in $(python3 -c 'import json; print(" ".join(x["id"] for x in
#           json.load(open("experiments/latent/regions_tn.json"))["regions"]))'); do
#       sbatch --job-name=s5r-$r experiments/latent/step5_regional.sbatch \
#           --config experiments/embed/stage1_v2.json \
#           --regions experiments/latent/regions_tn.json \
#           --run $R/embed/stage1_v2/d16_g0.5_r0.01 --step1 $R/latent/step1_sigma \
#           --step3 $R/latent/step3 --step3b $R/latent/step3b --step4 $R/latent/step4 \
#           --step4c $R/latent/step4c --out $R/latent/step5 --threads 16 --region $r
#   done
#
# and, once they are done (any time, for a partial summary):
#
#   python experiments/latent/collect_step5.py --root $R/latent/step5 \
#       --step4e $R/latent/step4e --step4d $R/latent/step4d --step4c $R/latent/step4c
#
# Resources, projected from the third pilot (r_hat arm, 16 threads): per
# unit, 4 MAP chains x 1,000 sweeps (0.73 s per sweep in nashville_core,
# 0.35 s in knoxville_core), 50 draw chains x 130 sweeps (about 950-1,200 s
# and 570-640 s), and the set-up with 50 r_hat refits (71 s with 4 refits in
# the pilot; at most about 900 s with 50 if the refits were all of it):
# nashville_core (6 zones, the largest region) about 30-50 minutes per unit,
# so its 10 replicates in about 5-8 hours; knoxville_core (4 zones) about
# 17-32 minutes per unit. Memory: the draw chains' kernel tables are about
# 74 MB per (draw, zone) in nashville_core, 11.2 GB for the default
# --draw-batch 25, plus about 3 GB of other per-draw arrays, on top of the
# shared data the pilot ran with under 64G. Finished units
# (their done.json) are skipped, so a timed-out job can be resubmitted
# unchanged.
#
# PMEDM_VB_ENV MUST point to the numba environment (pmedm_vb_numba: numba,
# arviz, torch); the default environment below has no numba, and the job
# stops if numba cannot be imported. sbatch passes the submitting shell's
# environment to the job, so the export above is enough.
#
# One process; the sampler's numba threads (--threads, default 16) should
# equal --cpus-per-task. NUMBA_NUM_THREADS (the most numba may use) and the
# BLAS threads of the unit's set-up (the MAP of U, the conditional PMEDM and
# its 50 refits) are set from --threads below. CPU only. The first run
# compiles the kernels (a minute or so) and caches them next to the source
# (src/pmedm_vb/latent/__pycache__). The output directory for the log
# (pmedm_vb_runs/latent) must exist before submitting. Offline: the Census
# API is never called (PMEDM_VB_OFFLINE=1).
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
echo "environment: $CONDA_PREFIX (PMEDM_VB_ENV=${PMEDM_VB_ENV:-unset})"
if ! $PY -c 'import numba, arviz; print("numba", numba.__version__, "arviz", arviz.__version__)'; then
    echo "ERROR: importing numba or arviz failed in $CONDA_PREFIX (the error is above)." >&2
    echo "       If this is not the numba environment, export PMEDM_VB_ENV before sbatch." >&2
    exit 1
fi

cd "$REPO"
echo "node $(hostname), ${SLURM_CPUS_PER_TASK} cores, $THREADS numba threads, numba $($PY -c 'import numba; print(numba.__version__, numba.config.THREADING_LAYER)'), commit $(git rev-parse --short HEAD), $(date)"
lscpu | grep -E "^Model name|^CPU\(s\)" || true
# pass the thread count through explicitly (the script's default is 16)
case " $* " in
    *" --threads "*) $PY -u experiments/latent/step5_regional.py "$@" ;;
    *) $PY -u experiments/latent/step5_regional.py "$@" --threads "$THREADS" ;;
esac
echo "done $(date)"
