#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nnagle@utk.edu

# The queue driver's job: one call of experiments/run_queue.py (--tick), which
# submits the next driver job while work remains. Started, with the account,
# partition, time and output set there, by
#
#   $CONDA_PREFIX/bin/python experiments/run_queue.py <experiment JSONs> --root $R --start
#
# The arguments are run_queue.py's. A call that fails exits non-zero, which mails
# the failure; the next driver job is still submitted unless calls keep failing.

set -eo pipefail
# PMEDM_VB_REPO and PMEDM_VB_ENV (environment) replace the UTK0496 defaults.
REPO=${PMEDM_VB_REPO:-/lustre/isaac24/proj/UTK0496/pmedm_vb}

module purge
unset PYTHONPATH  # a login profile's PYTHONPATH would shadow the environment's packages
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate "${PMEDM_VB_ENV:-/lustre/isaac24/proj/UTK0496/envs/pmedm_vb}"
set -u

PY=$CONDA_PREFIX/bin/python
case "$($PY -c 'import sys; print(sys.executable)')" in
    "$CONDA_PREFIX"/*) ;;
    *) echo "ERROR: python is not the pmedm_vb environment's" >&2; exit 1 ;;
esac

cd "$REPO"
echo "node $(hostname), commit $(git rev-parse --short HEAD), $(date)"
exec $PY -u experiments/run_queue.py "$@"
