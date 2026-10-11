#!/bin/bash
# Clean generated emergencies with zero-shot NLI (train/safety_clean.py). usage: sbatch train/safety_clean.sbatch GEN OUT [REPLACE CATEGORY]
#SBATCH --job-name=pv_safety_clean
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=runs/logs/%x_%j.log
set -euo pipefail
module load devel/python/3.12.3-gnu-14.2
unset PYTHONPATH
export PYTHONPATH=$PWD
export HF_HOME=${PUENTE_WORKDIR:?set PUENTE_WORKDIR to the models and data directory}/hf HF_HUB_OFFLINE=1
# torch + transformers come from a separate environment: training adds no dependency to this project
${PUENTE_TRAIN_PYTHON:?set PUENTE_TRAIN_PYTHON to a python with torch and transformers} train/safety_clean.py "$@"
