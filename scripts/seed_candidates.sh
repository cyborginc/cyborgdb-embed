#!/usr/bin/env bash
# Places unpublished archives where the CMake fetch looks first, so a build links
# them instead of downloading the pinned release. Nothing checks their digests:
# this is for verifying a candidate before it is published, never for shipping.
#
#   seed_candidates.sh <dir holding libonnxruntime-*.a and/or libtokenizer-*.a>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
from=${1:?usage: seed_candidates.sh <dir>}
seeded=0

seed() {
  local pattern=$1 tag=$2 archive
  for archive in "$from"/$pattern; do
    [ -f "$archive" ] || continue
    mkdir -p "$ROOT/.artifacts/$tag"
    # A stripped copy left from the published archive would otherwise win.
    rm -f "$ROOT/.artifacts/$tag/$(basename "$archive" .a)-local-stripped.a"
    cp "$archive" "$ROOT/.artifacts/$tag/"
    echo "seeded $tag/$(basename "$archive")"
    seeded=$((seeded + 1))
  done
}

seed 'libonnxruntime-*.a' "$(python3 "$ROOT/scripts/version.py" onnxruntime --tag)"
seed 'libtokenizer-*.a' "$(python3 "$ROOT/scripts/version.py" tokenizers --tag)"

# Seeding nothing would leave the build quietly testing the published archives.
[ "$seeded" -gt 0 ] || { echo "no candidate archives in $from"; exit 1; }
