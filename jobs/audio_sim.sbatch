#!/bin/bash
# Audio simulation: every scenario as a phone call through the real speech pipeline.
# Usage: sbatch --array=0-3 jobs/audio_sim.sbatch <models-dir> [audio_sim args, default: scenarios scenarios/audio]
# then: .venv/bin/python -m puente_voice.sim.audio_sim --merge runs/audio/calls-<array-job-id>-s*of4.jsonl
#SBATCH --job-name=pv_audio_sim
#SBATCH --partition=gpu_h100_short,gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G  # + the 1.1 GB safety classifier when --safety-model is given
#SBATCH --time=00:15:00
#SBATCH --output=runs/logs/%x_%A_%a.log
set -euo pipefail
V=${1:?models dir}
shift
[ $# -gt 0 ] || set -- scenarios scenarios/audio
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8  # CUDA runtime for the GPU build of onnxruntime (Kokoro)
export PYTHONPATH=$V/ortgpu  # onnxruntime-gpu shadows the venv's CPU build
export HF_HOME=$V/hf HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=$V/pylibs/nvidia/cublas/lib:$V/pylibs/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name --format=csv,noheader
.venv/bin/python -m puente_voice.sim.audio_sim --models "$V" --shard "${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}" "$@"
