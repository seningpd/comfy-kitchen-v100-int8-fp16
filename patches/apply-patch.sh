#!/usr/bin/env bash
# Apply the V100 INT8 -> FP16 tensor-core fallback patch to a comfy-kitchen checkout.
#
# Usage:
#   bash apply-patch.sh /path/to/comfy-kitchen
#
# The checkout must be at the same revision the patch was generated against
# (comfy-kitchen v0.2.40, commit f8b7e97). If it is not, the patch may not
# apply cleanly -- in that case regenerate it against your revision:
#
#   git -C /path/to/comfy-kitchen diff --no-color HEAD > v100-int8-fp16-fallback.patch
#
set -euo pipefail

TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
  echo "usage: bash apply-patch.sh /path/to/comfy-kitchen" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/v100-int8-fp16-fallback.patch"

if [[ ! -d "$TARGET/.git" ]]; then
  echo "error: $TARGET is not a git checkout" >&2
  exit 1
fi

echo "==> checking out a clean tree"
git -C "$TARGET" stash push --include-untracked --quiet || true

echo "==> applying $PATCH"
git -C "$TARGET" apply --3way "$PATCH"

echo
echo "Applied. Files changed:"
git -C "$TARGET" diff --stat HEAD
echo
echo "Next:"
echo "  cd $TARGET && pip install -e ."
echo "  python -m pytest tests/test_int8_fp16_fallback.py -q"
