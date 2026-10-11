#!/bin/bash
# Safety-model gates (eval/safety_model_eval.py) on CPU and GPU, plus the text sim with the model.
# usage: sbatch eval/safety_model_eval.sbatch MODEL_DIR <dev|test>
#SBATCH --job-name=pv_safety_eval
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=runs/logs/%x_%j.log
set -euo pipefail
V=${PUENTE_WORKDIR:?set PUENTE_WORKDIR to the models and data directory}
module load devel/python/3.12.3-gnu-14.2
unset PYTHONPATH
export OMP_NUM_THREADS=4
echo "== CPU (4 threads)"
.venv/bin/python eval/safety_model_eval.py "$1" --split "$2"
echo "== text sim with the model"
.venv/bin/python -m puente_voice.sim.text_sim scenarios "$1"
echo "== GPU"
module load devel/cuda/12.8
PYTHONPATH=$V/ortgpu LD_LIBRARY_PATH=$V/pylibs/nvidia/cublas/lib:$V/pylibs/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-} \
  .venv/bin/python eval/safety_model_eval.py "$1" --split "$2" --gpu
