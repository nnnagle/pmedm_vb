#!/bin/bash
# Submit the baseline grid: the target arrays on CPU, then five GPU jobs that
# wait for it -- one per d running its 9 (gamma, ridge) fits in turn, and the
# unweighted diagnostic (d 8, gamma 0.5, ridge 1e-3). Five jobs keep within
# campus-gpu's 6 submitted per user; at most 3 run at once and the rest queue.
#
#   bash experiments/embed/submit_baseline_grid.sh [CONFIG] [OUT]

set -euo pipefail
CONFIG=${1:-experiments/embed/stage1_v1.json}
OUT=${2:-/lustre/isaac24/proj/UTK0533/pmedm_vb_runs/embed/baseline}
mkdir -p /lustre/isaac24/proj/UTK0533/pmedm_vb_runs/embed "$OUT"

build=$(sbatch --parsable experiments/embed/build_targets.sbatch "$CONFIG")
echo "targets: job $build"
for d in 4 8 16 32; do
    job=$(sbatch --parsable --dependency=afterok:$build --job-name=embed-d$d \
        experiments/embed/fit_baseline.sbatch --config "$CONFIG" --out "$OUT" \
        --d $d --gamma 0 0.5 1 --ridge 1e-4 1e-3 1e-2)
    echo "d=$d: job $job"
done
job=$(sbatch --parsable --dependency=afterok:$build --job-name=embed-unw \
    experiments/embed/fit_baseline.sbatch --config "$CONFIG" --out "$OUT" \
    --d 8 --gamma 0.5 --ridge 1e-3 --unweighted)
echo "unweighted: job $job"
