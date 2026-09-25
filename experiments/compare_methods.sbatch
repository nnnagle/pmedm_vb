#!/bin/bash
#SBATCH --job-name=pmedm-score
#SBATCH --account=acf-utk0011
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# Score several (method, PUMA, alpha) cells with experiments/compare_methods.py,
# one after another, in one CPU job:
#
#   sbatch experiments/compare_methods.sbatch SPECFILE
#
# Each non-blank, non-# line of SPECFILE is the argument list of one
# compare_methods.py call (--method, --puma, --alpha, --run, --out, ...). Every
# call appends to the CSV its --out names. A call that fails does not stop the
# rest; the job fails at the end if any did.

set -eo pipefail
SPEC=${1:?usage: sbatch compare_methods.sbatch SPECFILE}
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
echo "node $(hostname), ${SLURM_CPUS_PER_TASK} cores, commit $(git rev-parse --short HEAD), $(date); spec $SPEC"
failed=0
while read -r line; do
    case "$line" in ''|\#*) continue ;; esac
    echo "=== $(date +%T) $line"
    if $PY -u experiments/compare_methods.py $line < /dev/null; then
        echo "=== $(date +%T) done"
    else
        echo "=== $(date +%T) FAILED"; failed=$((failed + 1))
    fi
done < "$SPEC"
echo "done $(date); $failed call(s) failed"
[ "$failed" -eq 0 ]
