#!/usr/bin/env bash
#
# Builds instrumented, runs the tests that need no network, and fails below a
# line-coverage floor.
#
# The golden test is included when a model cache is present: it is the only
# thing that exercises a real session, and without it the inference path is
# unmeasured. Its absence lowers the number rather than breaking the run.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${COVERAGE_BUILD_DIR:-$ROOT/build-coverage}"
FLOOR="${COVERAGE_FLOOR:-90}"

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Debug -DCYBORGDB_EMBED_COVERAGE=ON
cmake --build "$BUILD" -j

cd "$BUILD"
rm -f ./*.profraw ./*.profdata coverage.info

binaries=(./pooling_test ./error_test)
LLVM_PROFILE_FILE=pooling.profraw ./pooling_test > /dev/null
LLVM_PROFILE_FILE=error.profraw ./error_test > /dev/null
if ./golden_test "$ROOT/tests/data" BAAI/bge-small-en-v1.5 > /dev/null 2>&1; then
  LLVM_PROFILE_FILE=golden.profraw ./golden_test "$ROOT/tests/data" \
    BAAI/bge-small-en-v1.5 > /dev/null
  binaries+=(./golden_test)
else
  echo "note: golden test skipped; the inference path will read as uncovered" >&2
fi

sources=("$ROOT"/src/*.cpp "$ROOT"/src/pooling.hpp)

if [[ "$(uname)" == "Darwin" ]]; then
  # Must be the toolchain that compiled the code: a profile written by Apple
  # clang is a format version ahead of whatever Homebrew has installed, and the
  # merge fails rather than degrading.
  profdata="xcrun llvm-profdata"; cov="xcrun llvm-cov"
elif command -v llvm-profdata > /dev/null; then
  profdata="llvm-profdata"; cov="llvm-cov"
fi

if [[ -n "${profdata:-}" ]]; then

  $profdata merge -sparse ./*.profraw -o all.profdata
  local_args=()
  for b in "${binaries[@]:1}"; do local_args+=(-object "$b"); done
  $cov report "${binaries[0]}" "${local_args[@]}" -instr-profile=all.profdata \
    "${sources[@]}"
  percent="$($cov export "${binaries[0]}" "${local_args[@]}" \
    -instr-profile=all.profdata -summary-only "${sources[@]}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["totals"]["lines"]["percent"])')"
else
  # gcov path, for a GCC build.
  lcov --capture --directory . --output-file coverage.info --quiet
  lcov --extract coverage.info "$ROOT/src/*" --output-file coverage.info --quiet
  lcov --list coverage.info
  percent="$(lcov --summary coverage.info 2>&1 | sed -n 's/.*lines\.*: \([0-9.]*\)%.*/\1/p')"
fi

printf '\nline coverage: %.2f%% (floor %s%%)\n' "$percent" "$FLOOR"
python3 -c "import sys; sys.exit(0 if float('$percent') >= float('$FLOOR') else 1)" || {
  echo "coverage below the floor" >&2; exit 1; }
