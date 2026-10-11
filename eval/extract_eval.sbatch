#!/bin/bash
#SBATCH --job-name=pv_extract_eval
#SBATCH --partition=cpu_il,cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=runs/logs/%x_%j.log
set -euo pipefail
E=${PUENTE_WORKDIR:?set PUENTE_WORKDIR to the models and data directory}/eval
mkdir -p $E/words $E/sentences
cd $E/words
for i in 0 1 2; do
  tar -xzf $E/mswc/test_$i.tar.gz --wildcards '*claro_*' '*vale_*' '*bueno_*' '*hola_*' '*gracias_*' '*correcto_*' '*primera_*' '*segunda_*' 2>/dev/null || true
done
find . -name '*.wav' | wc -l
cd $E/sentences
head -121 $E/fleurs/test.tsv | tail -120 | cut -f2 > names.txt
tar -xzf $E/fleurs/test.tar.gz -T <(sed 's|^|test/|' names.txt) 2>/dev/null || tar -xzf $E/fleurs/test.tar.gz -T names.txt
find . -name '*.wav' | wc -l
