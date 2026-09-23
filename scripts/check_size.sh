#!/usr/bin/env bash
# Fails when the footprint a consumer inherits grows past its budget.
#
# The figure that matters is the stripped size of a program that links the
# library and does nothing else: almost all of it is ONNX Runtime, so it moves
# when ONNX Runtime's build flags move. An operator set re-enabled by accident,
# or RTTI turned back on, shows up here rather than in someone's deployment.
#
# Budgets are the measured size plus room for ordinary drift. Raise one only
# with a reason, because raising it is the only way this check ever passes again.
set -euo pipefail

binary=${1:-build/size_probe}
[ -f "$binary" ] || { echo "no binary at $binary; build the size_probe target"; exit 1; }

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)       budget_mb=19 ;;
  Linux-aarch64)      budget_mb=19 ;;
  Linux-x86_64)       budget_mb=20 ;;
  *) echo "no budget recorded for $(uname -s)-$(uname -m); skipping"; exit 0 ;;
esac

stripped=$(mktemp)
trap 'rm -f "$stripped"' EXIT
cp "$binary" "$stripped"
# -x on Darwin and --strip-all elsewhere: both leave what a consumer would ship.
if [ "$(uname -s)" = "Darwin" ]; then strip -x "$stripped"; else strip --strip-all "$stripped"; fi

bytes=$(wc -c < "$stripped" | tr -d ' ')
mb=$((bytes / 1048576))
echo "stripped consumer footprint: ${mb} MB (budget ${budget_mb} MB)"
[ "$mb" -le "$budget_mb" ] || {
  echo "footprint grew past its budget; check what changed in the archives"
  exit 1; }
