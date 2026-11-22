#!/bin/bash
#SBATCH -J bsd
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -o bsd_%x_%A_%a.out
# Runs one `bsd` command inside the environment from hpc/setup_env.sh. Usually submitted by
# hpc/submit_movie.sh, which picks partitions and chains the steps. In an array job,
# `--block slurm` becomes this task's block (task K of N).
set -euo pipefail
BSD_ENV=${BSD_ENV:-/scratch/$USER/bsd-env}
BSD_PYTHON_MODULE=${BSD_PYTHON_MODULE-python/3.13.5}
[ -n "$BSD_PYTHON_MODULE" ] && module load "$BSD_PYTHON_MODULE"
source "$BSD_ENV/bin/activate"
args=()
for a in "$@"; do
  [ "$a" = slurm ] && a="${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}"
  args+=("$a")
done
python -c "import torch; print('torch', torch.__version__, 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
echo "bsd ${args[*]}"
bsd "${args[@]}"
