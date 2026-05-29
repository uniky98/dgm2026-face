#!/usr/bin/env bash
# Verify that re-running inference.py produces a bit-identical 1000-image set.
#
# Usage:
#   ./verify_reproducibility.sh                       # uses defaults below
#   ./verify_reproducibility.sh WEIGHTS ORIG_ZIP SG3_REPO
#
# Defaults (relative to this script's dir):
#   WEIGHTS  = weights/network-snapshot-001360.pkl
#   ORIG_ZIP = submission.zip
#   SG3_REPO = stylegan3            (cloned by Dockerfile or git-clone manually)
#
# Exit code 0 iff every regenerated PNG sha256-matches the corresponding entry
# of the original submission zip. Prints first few divergences if any.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

WEIGHTS="${1:-weights/network-snapshot-001360.pkl}"
ORIG_ZIP="${2:-submission.zip}"
SG3_REPO="${3:-stylegan3}"

[ -f "$WEIGHTS" ]  || { echo "ERROR: weights not found at $WEIGHTS"; exit 1; }
[ -f "$ORIG_ZIP" ] || { echo "ERROR: original zip not found at $ORIG_ZIP"; exit 1; }
[ -d "$SG3_REPO" ] || { echo "ERROR: NVlabs/stylegan3 source not found at $SG3_REPO"; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "[verify] regenerating into $WORK/regen"
python inference.py \
    --weights "$WEIGHTS" \
    --seeds   seeds.json \
    --outdir  "$WORK/regen" \
    --stylegan3-repo "$SG3_REPO"

echo "[verify] unpacking original zip"
mkdir -p "$WORK/orig"
unzip -q "$ORIG_ZIP" -d "$WORK/orig"

echo "[verify] comparing sha256 of each PNG"
fail=0; checked=0; mismatches=0; missing=0
for p in "$WORK/regen"/seed*.png; do
    name=$(basename "$p")
    orig="$WORK/orig/$name"
    if [ ! -f "$orig" ]; then
        missing=$((missing+1))
        [ "$missing" -le 3 ] && echo "  MISSING in original: $name"
        fail=1; continue
    fi
    a=$(sha256sum "$p"    | awk '{print $1}')
    b=$(sha256sum "$orig" | awk '{print $1}')
    if [ "$a" != "$b" ]; then
        mismatches=$((mismatches+1))
        [ "$mismatches" -le 3 ] && { echo "  MISMATCH: $name"; echo "    regen: $a"; echo "    orig : $b"; }
        fail=1
    fi
    checked=$((checked+1))
done
echo "[verify] checked $checked files  match=$((checked-mismatches-missing))  mismatch=$mismatches  missing=$missing"
if [ "$fail" -eq 0 ]; then
    echo "REPRO OK — all $checked PNGs match the original submission"
else
    echo "REPRO FAILED — see mismatches above"
    exit 2
fi
