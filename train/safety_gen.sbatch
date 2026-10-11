#!/bin/bash
# Generate synthetic safety training turns (train/safety_gen.py). usage: sbatch train/safety_gen.sbatch OUT_DIR
#SBATCH --job-name=pv_safety_gen
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=runs/logs/%x_%j.log
set -euo pipefail
module load devel/python/3.12.3-gnu-14.2
unset PYTHONPATH
export HF_HOME=${HF_HOME:?set HF_HOME to the Qwen2.5-7B-Instruct cache} HF_HUB_OFFLINE=1
# torch + transformers come from a separate environment: training adds no dependency to this project
${PUENTE_TRAIN_PYTHON:?set PUENTE_TRAIN_PYTHON to a python with torch and transformers} train/safety_gen.py "${1:?out dir}" ${2:-}
