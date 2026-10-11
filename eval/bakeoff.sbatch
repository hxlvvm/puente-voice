#!/bin/bash
#SBATCH --job-name=pv_asr_bakeoff
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=runs/logs/%x_%j.log
set -euo pipefail
V=${PUENTE_WORKDIR:?set PUENTE_WORKDIR to the models and data directory}
module load devel/python/3.12.3-gnu-14.2
export PYTHONPATH=$V/pylibs-asr HF_HOME=$V/hf HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=$V/pylibs/nvidia/cublas/lib:$V/pylibs/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
.venv/bin/python eval/${2:-asr_bakeoff.py} "$V"
