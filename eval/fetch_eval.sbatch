#!/bin/bash
#SBATCH --job-name=pv_fetch_eval
#SBATCH --partition=cpu_il,cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:40:00
#SBATCH --output=runs/logs/%x_%j.log
set -euo pipefail
V=${PUENTE_WORKDIR:?set PUENTE_WORKDIR to the models and data directory}
E=$V/eval; mkdir -p $E/mswc $E/fleurs $V/asr $V/pylibs-asr
HF=https://huggingface.co/datasets
cd $E/mswc
curl -sSfLO $HF/MLCommons/ml_spoken_words/resolve/main/data/splits/es/splits.tar.gz
for i in 0 1 2; do curl -sSfL -o test_$i.tar.gz $HF/MLCommons/ml_spoken_words/resolve/main/data/wav/es/test/audio/$i.tar.gz; done
cd $E/fleurs
curl -sSfLO $HF/google/fleurs/resolve/main/data/es_419/test.tsv
curl -sSfL -o test.tar.gz $HF/google/fleurs/resolve/main/data/es_419/audio/test.tar.gz
cd $V/asr
curl -sSfL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2 | tar -xj
module load devel/python/3.12.3-gnu-14.2
unset PYTHONPATH
PIP_CACHE_DIR=$V/pipcache .venv/bin/pip install -q --target $V/pylibs-asr sherpa-onnx
du -sh $E/* $V/asr/* $V/pylibs-asr; ls $V/asr/*/
