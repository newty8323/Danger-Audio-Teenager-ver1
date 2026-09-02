#!/usr/bin/env bash
# Fetch the full experiment data bundle from the GitHub Release of this (private) repo
# and restore it into the working layout the scripts expect.
#
# Requirements:
#   - GitHub CLI (`gh`) logged in with an account that has access to the repo
#     (https://cli.github.com -> `gh auth login`)
#   - ~11 GB free disk (5.4 GB download + unpacked copies)
#
# Usage: bash scripts/fetch_data.sh [tag]            full bundle (~5.4 GB, for training/eval)
#        bash scripts/fetch_data.sh --models [tag]   models only (~0.2 GB, to RUN src/app)
#
# Restored layout:
#   data_dl/clips/*.wav            raw 10s audio clips (8,648)
#   data_dl/features/*.npy         precomputed log-mel features (8,409)
#   data_dl/manifests/*.jsonl      label manifests (v2.0-vio taxonomy)
#   data_dl/artifacts/*.npz        eval outputs (probs_*, norm stats, calibration)
#   data_dl/asr/                   ASR CER results + listen samples
#   data_dl/weights/BEATs_iter3_plus_AS2M.pt   BEATs backbone (fine-tune input)
#   ckpt_ced_mini_vio/best.ckpt    adopted violence trigger (CED-mini)
#   artifacts/koelectra_small_harm*/           adopted text classifier (KoELECTRA-small)
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_ONLY=0
if [ "${1:-}" = "--models" ]; then
  MODELS_ONLY=1
  shift
fi
TAG="${1:-data-v1}"
REPO="soysaucecrab/Danger-Audio-Teenager"
DL="data_dl/release_download"

mkdir -p "$DL"

# Client mode (a laptop running src/app): inference needs only the adopted checkpoints —
# clips/features/manifests are for training and evaluation.
if [ "$MODELS_ONLY" = 1 ]; then
  echo "[fetch_data] models-only: downloading ckpt_final.tar from release '$TAG' ..."
  # --clobber, NOT --skip-existing: the asset is re-uploaded whenever a model is retrained,
  # so skipping an existing local copy silently keeps stale weights (hit on 2026-07-30).
  gh release download "$TAG" -R "$REPO" -D "$DL" -p ckpt_final.tar --clobber
  tar -xf "$DL"/ckpt_final.tar     # -> ckpt_ced_mini_vio/ + artifacts/koelectra_*
  echo "[fetch_data] done:"
  du -sh ckpt_ced_mini_vio artifacts/koelectra_small_harm_asraug_slang
  echo "[fetch_data] (Moonshine-KR and the CED-mini base weights download from HF on first run.)"
  exit 0
fi

echo "[fetch_data] downloading release '$TAG' from $REPO ..."
gh release download "$TAG" -R "$REPO" -D "$DL" --skip-existing

echo "[fetch_data] unpacking (this can take a few minutes) ..."
cat "$DL"/clips.tar.part*    | tar -xf -    # -> data_dl/clips/
cat "$DL"/features.tar.part* | tar -xf -    # -> data_dl/features/
tar -xzf "$DL"/meta.tar.gz                  # -> data_dl/{manifests,artifacts,asr}
tar -xf  "$DL"/ckpt_final.tar               # -> ckpt_ced_mini_vio/ + artifacts/koelectra_*
mkdir -p data_dl/weights
cp -f "$DL"/BEATs_iter3_plus_AS2M.pt data_dl/weights/

echo "[fetch_data] done. Restored:"
du -sh data_dl/clips data_dl/features data_dl/manifests data_dl/artifacts \
       data_dl/weights ckpt_ced_mini_vio artifacts/koelectra_small_harm_asraug
echo "[fetch_data] you can delete $DL to reclaim the 5.4 GB download."
