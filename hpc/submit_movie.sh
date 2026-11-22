#!/bin/bash
# Analyse one movie on a SLURM cluster: prepare (CPU) -> segment on N GPUs -> track and growth (CPU).
#
#   bash hpc/submit_movie.sh MOVIE RUN_DIR [N_GPUS] [extra bsd prepare options, e.g. --position 2]
#
# Each step waits for the previous one. Logs (bsd_<step>_<job>_<task>.out) go to the
# current directory. If a segment task runs out of time, rerun the script: finished frames
# are skipped. Set BSD_GPU_PARTITION / BSD_CPU_PARTITION / BSD_GPU_TIME for your cluster.
set -euo pipefail
MOVIE=${1:?movie file}; RUN=${2:?run folder}; N=${3:-4}; shift $(( $# < 3 ? $# : 3 ))
HERE=$(cd "$(dirname "$0")" && pwd)
GPU_P=${BSD_GPU_PARTITION:-gpu}; CPU_P=${BSD_CPU_PARTITION:-short}; GPU_T=${BSD_GPU_TIME:-01:30:00}
prep=$(sbatch --parsable -J prepare -p "$CPU_P" -t 01:00:00 -o "bsd_%x_%j.out" "$HERE/bsd_job.sbatch" prepare "$MOVIE" --out "$RUN" "$@")
seg=$(sbatch --parsable -J segment -p "$GPU_P" --gres=gpu:1 -t "$GPU_T" --array=0-$((N - 1)) \
      --dependency=afterok:$prep "$HERE/bsd_job.sbatch" segment "$RUN" --block slurm)
trk=$(sbatch --parsable -J track -p "$CPU_P" -t 03:00:00 --mem=64G -o "bsd_%x_%j.out" --dependency=afterok:$seg \
      "$HERE/bsd_job.sbatch" track "$RUN")
grw=$(sbatch --parsable -J growth -p "$CPU_P" -t 00:30:00 -o "bsd_%x_%j.out" --dependency=afterok:$trk \
      "$HERE/bsd_job.sbatch" growth "$RUN")
echo "submitted: prepare $prep -> segment $seg (x$N) -> track $trk -> growth $grw"
echo "watch with: squeue -u $USER"
