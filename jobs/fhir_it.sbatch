#!/bin/bash
#SBATCH --job-name=pv_fhir_it
#SBATCH --partition=<cpu-partition>
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=00:10:00
#SBATCH --output=runs/logs/%x_%j.log
# FHIR integration tests on a compute node: HAPI in Apptainer + pytest -m fhir. Submit from the repo root:
#   sbatch jobs/fhir_it.sbatch <dir-with-or-for-hapi.sif>
set -euo pipefail
module load devel/python/3.12.3-gnu-14.2
unset PYTHONPATH
export PYTHONNOUSERSITE=1
cd "$SLURM_SUBMIT_DIR"
bash scripts/hapi_apptainer.sh "$1"
trap 'kill $(cat "$1/hapi.pid") 2>/dev/null || true' EXIT
PUENTE_FHIR_URL=http://localhost:8080/fhir .venv/bin/pytest -m fhir -q
