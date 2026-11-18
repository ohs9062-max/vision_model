#!/usr/bin/env bash
# 검증된 버전을 확인한 뒤 활성 모델 설정을 되돌린다.
# 활성 버전은 config/model.yaml의 active_model 한 줄만 교체한다.
set -euo pipefail
PACKAGE="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
exec "$PYTHON" "$PACKAGE/tools/manage_versions.py" rollback "$@"
