#!/bin/bash
# Zero-shot NLI safety spike (eval/nli_safety_spike.py). usage: sbatch [-p cpu_il --gres=] eval/nli_spike.sbatch
#SBATCH --job-name=pv_nli_spike
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=runs/logs/%x_%j.log
set -euo pipefail
module load devel/python/3.12.3-gnu-14.2
unset PYTHONPATH
export PYTHONPATH=$PWD
export HF_HOME=${PUENTE_WORKDIR:?set PUENTE_WORKDIR to the models and data directory}/hf HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
# torch + transformers come from a separate environment: the spike adds no dependency to this project
${PUENTE_TRAIN_PYTHON:?set PUENTE_TRAIN_PYTHON to a python with torch and transformers} eval/nli_safety_spike.py
