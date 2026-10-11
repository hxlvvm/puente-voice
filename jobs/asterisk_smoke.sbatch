#!/bin/bash
# Smoke test through a real Asterisk 22 (eval/asterisk_smoke.py).
# Usage: sbatch jobs/asterisk_smoke.sbatch <models-dir> [cuda|cpu]   (cpu hides the GPU: the compose default)
#SBATCH --job-name=pv_asterisk_smoke
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=runs/logs/%x_%j.log
set -euo pipefail
V=${1:?models dir}
module load devel/python/3.12.3-gnu-14.2
export HF_HOME=$V/hf HF_HUB_OFFLINE=1
if [ "${2:-cuda}" = cpu ]; then
  unset PYTHONPATH; export CUDA_VISIBLE_DEVICES=  # the venv's CPU onnxruntime, Whisper int8 on 4 cores
else
  module load devel/cuda/12.8  # CUDA runtime for the GPU build of onnxruntime (Kokoro)
  export PYTHONPATH=$V/ortgpu  # onnxruntime-gpu shadows the venv's CPU build
  export LD_LIBRARY_PATH=$V/pylibs/nvidia/cublas/lib:$V/pylibs/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
fi
.venv/bin/python eval/asterisk_smoke.py "$V/containers/asterisk22.sif" "$V" "runs/asterisk_smoke/$SLURM_JOB_ID" "${2:-cuda}"
