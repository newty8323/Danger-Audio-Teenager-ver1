#!/usr/bin/env bash
# Download the compact Korean language-gate artifacts published with this repository.
# The CED-mini and KoELECTRA runtime models remain in the original data-v1 release and are
# fetched separately by scripts/fetch_data.sh --models.
set -euo pipefail

cd "$(dirname "$0")/.."

REPO="${LANGUAGE_GATE_REPO:-newty8323/Danger-Audio-Teenager-Korean-Gate}"
TAG="${1:-v0.1.0}"
ASSET="language_gate_models_macos.tar.gz"
DEST="artifacts/language_gate"
DL="data_dl/language_gate_release"

if ! command -v gh >/dev/null 2>&1; then
  echo "[language-gate] GitHub CLI (gh) is required. Run: gh auth login" >&2
  exit 1
fi

mkdir -p "$DL" "$DEST"
echo "[language-gate] downloading $ASSET from $REPO release $TAG ..."
gh release download "$TAG" -R "$REPO" -D "$DL" -p "$ASSET" --clobber
tar -xzf "$DL/$ASSET" -C "$DEST"

for file in silero_vad.jit whisper_tiny_encoder_lid.pt; do
  if [ ! -f "$DEST/$file" ]; then
    echo "[language-gate] expected artifact is missing after extraction: $DEST/$file" >&2
    exit 1
  fi
done

du -sh "$DEST"
echo "[language-gate] ready: use --language-gate-vad $DEST/silero_vad.jit"
