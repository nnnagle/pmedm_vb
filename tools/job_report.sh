#!/bin/bash
# Collect what a set of Slurm jobs produced into one text file, for review
# off-cluster: sacct states, each job's summary CSV, the tail of its Slurm log,
# the tail of every per-fit log, and an HMC job's report.
#
#   tools/job_report.sh OUTFILE JOBID [JOBID ...]
#   tools/job_report.sh OUTFILE $(tail -n +2 $RUNS/grid_cpu_jobs.csv | cut -d, -f1)
RUNS=${RUNS:-/lustre/isaac24/proj/UTK0496/pmedm_vb_runs}
out=$1; shift
ids=("$@")
{
  echo "== sacct"
  sacct -j "$(IFS=,; echo "${ids[*]}")" -X \
      --format=JobID,JobName%24,Partition,State,Elapsed,Timelimit,ExitCode,NodeList
  for J in "${ids[@]}"; do
    echo; echo "################ job $J"
    for S in "$RUNS"/summary-*-"$J".csv; do
      [ -e "$S" ] && { echo "== $(basename "$S")"; cat "$S"; }
    done
    echo "== slurm log (last 25 lines)"
    tail -n 25 "$RUNS/slurm-$J.log" 2>/dev/null || echo "(no log yet)"
    for L in "$RUNS/$J"/logs/*.log; do
      [ -e "$L" ] && { echo "== fit log $(basename "$L") (last 15 lines)"; tail -n 15 "$L"; }
    done
    for R in "$RUNS/$J"/mcmc/*_report.txt; do
      [ -e "$R" ] && { echo "== HMC report $(basename "$R")"; cat "$R"; }
    done
  done
} > "$out" 2>&1
echo "written to $out"
