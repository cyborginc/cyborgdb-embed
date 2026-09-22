#!/usr/bin/env bash
# Fails when any object in a Mach-O archive requires a newer macOS than the
# deployment target in versions.json. A wheel built for that target cannot link
# such an object without the linker warning and delocate refusing the wheel.
#
#   check_minos.sh <archive>...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="$(python3 "$ROOT/scripts/version.py" macos)"
[ $# -gt 0 ] || { echo "usage: check_minos.sh <archive>..."; exit 2; }

status=0
for archive in "$@"; do
  # LC_BUILD_VERSION reports "minos", the older LC_VERSION_MIN_MACOSX "version".
  found=$(otool -l "$archive" | awk '
    /cmd LC_BUILD_VERSION/      { want = "minos" }
    /cmd LC_VERSION_MIN_MACOSX/ { want = "version" }
    want != "" && $1 == want    { print $2; want = "" }' | sort -u)
  [ -n "$found" ] || { echo "$archive: no load commands read"; status=1; continue; }

  newest=$(python3 -c '
import sys
versions = sys.argv[1].split()
print(max(versions, key=lambda v: tuple(int(p) for p in v.split("."))))' "$found")
  if python3 -c '
import sys
key = lambda v: tuple(int(p) for p in v.split("."))
sys.exit(key(sys.argv[1]) > key(sys.argv[2]))' "$newest" "$target"; then
    echo "$archive: minos $(paste -sd, - <<< "$found") (target $target): ok"
  else
    echo "$archive: requires macOS $newest, above the $target target"
    status=1
  fi
done
exit "$status"
