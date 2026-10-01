#!/bin/bash
#SBATCH --job-name=pmedm-map-bundle
#SBATCH --account=acf-utk0011
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/slurm-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nnagle@utk.edu

# MAP and all three VB families for one study area and PUMA, in one job -- for
# when the queue will not take one job per fit.
#
#   sbatch experiments/run_map_bundle.sbatch NAME PUMA [run_map.py options...]
#
# e.g.
#
#   sbatch experiments/run_map_bundle.sbatch knox-min 4701502 \
#       --alpha 1 0.1 0.01 --taper none --variance-floor zero --hierarchy puma
#
# First MAP (the solve step) for every alpha, then the gaussian, skewed and
# sumdiff VB fits at once, each run_map.py process given a third of the cores
# and running its alphas in parallel. Results go to $RUNS/<jobid>/map and
# $RUNS/<jobid>/vb_<family>, each with its own summary.csv and logs/; each VB
# process's own output is in $RUNS/<jobid>/vb_<family>.log. The job fails at
# the end if any step did. --draws defaults to 64 here, as in the grid.
# FAMILIES (environment) picks the VB families, default "gaussian skewed
# sumdiff"; e.g. sbatch --export=ALL,FAMILIES="gaussian skewed" ... (sumdiff
# does not run under --collapse).

set -eo pipefail
NAME=${1:?usage: sbatch run_map_bundle.sbatch NAME PUMA [options]}
PUMA=${2:?usage: sbatch run_map_bundle.sbatch NAME PUMA [options]}
shift 2

REPO=/lustre/isaac24/proj/UTK0496/pmedm_vb
RUNS=/lustre/isaac24/proj/UTK0496/pmedm_vb_runs
# OUT_DIR (environment) replaces the default results directory.
OUT=${OUT_DIR:-$RUNS/${SLURM_JOB_ID}}

module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /lustre/isaac24/proj/UTK0496/envs/pmedm_vb
set -u
# A caller (experiments/run_experiment.py) may point the job at another data directory.
export PMEDM_VB_DATA=${PMEDM_VB_DATA:-/lustre/isaac24/scratch/$USER/pmedm_vb_data}
export PMEDM_VB_OFFLINE=1
# One BLAS thread per driver; run_map.py sets each worker's own from --cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PY=$CONDA_PREFIX/bin/python
case "$($PY -c 'import sys; print(sys.executable)')" in
    "$CONDA_PREFIX"/*) ;;
    *) echo "ERROR: python is not the pmedm_vb environment's" >&2; exit 1 ;;
esac

mkdir -p "$OUT"
cd "$REPO"
CORES=${SLURM_CPUS_PER_TASK}
read -r -a FAMS <<< "${FAMILIES:-gaussian skewed sumdiff}"
PER=$((CORES / ${#FAMS[@]}))
echo "node $(hostname), $CORES cores ($PER per VB family: ${FAMS[*]}), commit $(git rev-parse --short HEAD), $(date)"
echo "name $NAME, puma $PUMA, options: $*"

failed=0
if $PY -u experiments/run_map.py solve --name "$NAME" --puma "$PUMA" --cores "$CORES" \
        --out "$OUT/map" "$@"; then
    echo "=== $(date +%T) map done"
else
    echo "=== $(date +%T) map FAILED"; failed=$((failed + 1))
fi

pids=()
for family in "${FAMS[@]}"; do
    $PY -u experiments/run_map.py vb --name "$NAME" --puma "$PUMA" --cores "$PER" \
        --family "$family" --draws 64 --out "$OUT/vb_$family" "$@" \
        > "$OUT/vb_$family.log" 2>&1 &
    pids+=($!)
done
for i in "${!FAMS[@]}"; do
    if wait "${pids[i]}"; then
        echo "=== $(date +%T) vb ${FAMS[i]} done"
    else
        echo "=== $(date +%T) vb ${FAMS[i]} FAILED (see $OUT/vb_${FAMS[i]}.log)"; failed=$((failed + 1))
    fi
done
echo "done $(date); $failed step(s) failed"
[ "$failed" -eq 0 ]
