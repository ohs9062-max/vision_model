#!/usr/bin/env bash
# 체크포인트 복사·내보내기·엄격 FP32 검증을 버전 관리 도구에 위임한다.
# 승인 기준은 verify_version.py의 엄격 FP32(b); AMP FP16(a) 미달은 경고만 남긴다.
set -euo pipefail
PACKAGE="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/home/hs/rang/X-AnyLabeling-Server/.venv/bin/python}"
exec "$PYTHON" "$PACKAGE/tools/manage_versions.py" update "$@"
