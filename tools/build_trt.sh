#!/usr/bin/env bash
# 지정 버전의 ONNX에서 AGX용 TensorRT FP16 엔진을 생성한다.
set -euo pipefail
PACKAGE="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-python3}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
VERSION="${2:-}"
FOLDER="$($PYTHON - "$PACKAGE" "$VERSION" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from vision_model.versioning import load_version
print(load_version(sys.argv[1],sys.argv[2] or None)[0])
PY
)"
for model in seg det; do
  "$TRTEXEC" --onnx="$FOLDER/$model.onnx" --saveEngine="$FOLDER/$model.fp16.engine" --fp16
done
# Apply the version's NumPy decode/NMS/finish after network outputs.
