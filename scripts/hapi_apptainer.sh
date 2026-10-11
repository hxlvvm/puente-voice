#!/bin/bash
# Pull (once) and start HAPI FHIR with Apptainer on the current node; waits until /fhir/metadata answers.
# usage: scripts/hapi_apptainer.sh <dir-for-sif>
set -euo pipefail
DIR=${1:?directory for hapi.sif}
SIF=$DIR/hapi-v8.12.0-2.sif
[ -f "$SIF" ] || apptainer pull "$SIF" docker://hapiproject/hapi:v8.12.0-2
# --pwd /app: the image starts main.war by a path relative to Docker's WORKDIR; --writable-tmpfs holds the small
# test H2 database under /app/target. Cap the heap: the JVM sizes it from node RAM, not the job's memory limit.
JAVA_TOOL_OPTIONS="-Xmx2g" apptainer run --pwd /app --writable-tmpfs "$SIF" > "$DIR/hapi.log" 2>&1 &
echo $! > "$DIR/hapi.pid"
for i in $(seq 120); do
  curl -sf http://localhost:8080/fhir/metadata > /dev/null && { echo "HAPI up"; exit 0; }
  sleep 2
done
echo "HAPI did not start" >&2; tail -50 "$DIR/hapi.log" >&2; exit 1
