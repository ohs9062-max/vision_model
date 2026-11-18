#!/usr/bin/env bash
# 패키지 루트에서 카메라 런타임을 실행한다.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python3}" -m vision_model.runtime "$@"
