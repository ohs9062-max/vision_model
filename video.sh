#!/usr/bin/env bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
python=/home/hs/rang/X-AnyLabeling-Server/.venv/bin/python
if [[ ! -x "$python" ]]; then python=python3; fi
exec "$python" -m vision_model.player_gui
