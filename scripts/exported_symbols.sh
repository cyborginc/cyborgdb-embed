#!/usr/bin/env bash
# Prints the strong symbols an archive exports to whoever links it, one per line
# and without Mach-O's leading underscore.
#
# Weak symbols are left out: they are COMDAT instantiations that deduplicate
# safely. So are Mach-O private externs, which resolve within the consumer's link
# but are never exported from it; nm -g lists them alongside real exports.
#
#   exported_symbols.sh <archive>
set -euo pipefail

archive=${1:?usage: exported_symbols.sh <archive>}
if [ "$(uname)" = Darwin ]; then
  nm -m "$archive" 2>/dev/null \
    | awk '!/undefined/ && /\) external / {print $NF}'
else
  nm -g --defined-only "$archive" 2>/dev/null | awk '$2 ~ /^[TDBR]$/ {print $3}'
fi | sed 's/^_//' | sort -u
