#!/bin/sh
# Download the speech models into $1 (default ./models). Kokoro-82M: Apache-2.0. Whisper turbo: MIT.
set -eu
d=${1:-models}
mkdir -p "$d/kokoro"
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  [ -s "$d/kokoro/$f" ] || curl -sSfL -o "$d/kokoro/$f" \
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$f"
done
echo "Kokoro in $d/kokoro; Whisper downloads itself on first start (or set PUENTE_WHISPER to a local path)"
