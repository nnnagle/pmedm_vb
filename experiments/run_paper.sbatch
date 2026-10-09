#!/bin/bash
#SBATCH --job-name=pmedm-paper
#SBATCH --account=acf-utk0011
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=nnagle@utk.edu

# The paper's driver: one pass of experiments/run_paper.py over every stage,
# which submits the next driver job while work remains. Started by
#
#   $CONDA_PREFIX/bin/python experiments/run_paper.py start
#
# with the output path set there; the arguments are run_paper.py's options.
# A pass that needs attention exits non-zero, which mails the failure.

set -eo pipefail
REPO=/lustre/isaac24/proj/UTK0496/pmedm_vb

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /lustre/isaac24/proj/UTK0496/envs/pmedm_vb
set -u
export PMEDM_VB_OFFLINE=1

PY=$CONDA_PREFIX/bin/python
case "$($PY -c 'import sys; print(sys.executable)')" in
    "$CONDA_PREFIX"/*) ;;
    *) echo "ERROR: python is not the pmedm_vb environment's" >&2; exit 1 ;;
esac

cd "$REPO"
echo "node $(hostname), commit $(git rev-parse --short HEAD), $(date)"
status=0
$PY -u experiments/run_paper.py tick "$@" || status=$?
echo "done $(date), status $status"
# 3 is "waiting on jobs": the next driver job is already submitted.
[ "$status" -eq 0 ] || [ "$status" -eq 3 ]
