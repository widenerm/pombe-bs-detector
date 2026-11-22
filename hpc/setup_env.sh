#!/bin/bash
# One-time setup on a SLURM cluster: a Python environment with CUDA PyTorch and BS-Detector.
#
#   bash hpc/setup_env.sh                      # installs from GitHub into $BSD_ENV
#   BSD_REPO=/path/to/checkout bash hpc/setup_env.sh
#
# BSD_ENV        where the environment goes          (default /scratch/$USER/bsd-env)
# BSD_REPO       what to install                     (default the GitHub repository, main branch)
# BSD_PYTHON_MODULE  module providing Python >= 3.10 (default python/3.13.5; "" to skip)
# BSD_TORCH_INDEX    PyTorch wheel index for your CUDA (default cu126)
set -euo pipefail
BSD_ENV=${BSD_ENV:-/scratch/$USER/bsd-env}
BSD_REPO=${BSD_REPO:-git+https://github.com/widenerm/pombe-bs-detector}
BSD_PYTHON_MODULE=${BSD_PYTHON_MODULE-python/3.13.5}
BSD_TORCH_INDEX=${BSD_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}
[ -n "$BSD_PYTHON_MODULE" ] && module load "$BSD_PYTHON_MODULE"
python -m venv "$BSD_ENV"
source "$BSD_ENV/bin/activate"
pip install -q --upgrade pip
# CUDA builds of torch and torchvision together, so they match; cellpose then keeps them
pip install -q torch torchvision --index-url "$BSD_TORCH_INDEX"
if [ -d "$BSD_REPO" ]; then pip install -q "$BSD_REPO"       # a local checkout
else pip install -q "bs-detector @ $BSD_REPO"; fi          # a git URL
bsd --help >/dev/null && echo "ready: source $BSD_ENV/bin/activate"
