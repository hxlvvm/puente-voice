#!/bin/bash
# Meaning (or safety) evaluation spoken through the 8 kHz phone line.
# Usage: sbatch eval/meaning_eval.sbatch x <dev|test> [meaning|safety] [CORPUS.yaml] [SAFETY_MODEL_DIR]
#SBATCH --job-name=pv_meaning_eval
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=runs/logs/%x_%j.log
set -euo pipefail
V=${PUENTE_WORKDIR:?set PUENTE_WORKDIR to the models and data directory}
module load devel/python/3.12.3-gnu-14.2
unset PYTHONPATH
export HF_HOME=$V/hf HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=$V/pylibs/nvidia/cublas/lib:$V/pylibs/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
.venv/bin/python eval/${3:-meaning}_eval.py --split "${2:-dev}" --asr "$V" ${4:+--corpus "$4"} ${5:+--model "$5"}
