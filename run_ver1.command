#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv가 없습니다. 먼저 https://docs.astral.sh/uv/ 에서 설치하세요."
  exit 1
fi

export PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --group nlp --group asr --group onnx python -m app.main "$@"
