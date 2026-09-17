#!/usr/bin/env bash
#
# Builds the dependencies cyborgdb-core vendors, at core's pins and with core's
# compile settings, so a symbol comparison against ONNX Runtime reflects what an
# actual consumer link would contain.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${CORE_DEPS_PREFIX:-$ROOT/.ort/core-deps}"
SRC="$PREFIX/src"

ABSL_TAG=20250814.2
FLATBUFFERS_TAG=v25.12.19
RE2_TAG=2025-11-05

if [[ "$(uname)" == "Darwin" ]]; then
  JOBS="$(sysctl -n hw.ncpu)"
else
  JOBS="$(nproc)"
fi

# Mirrors cyborgdb-core: C++17, Release, hidden visibility, position independent.
COMMON_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_STANDARD=17
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
  -DCMAKE_CXX_VISIBILITY_PRESET=hidden
  -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON
  -DCMAKE_INSTALL_PREFIX="$PREFIX"
  -DCMAKE_PREFIX_PATH="$PREFIX"
  -DBUILD_SHARED_LIBS=OFF
)

fetch() {
  local name="$1" url="$2" tag="$3"
  [[ -d "$SRC/$name" ]] && return 0
  mkdir -p "$SRC"
  git clone --depth 1 --branch "$tag" "$url" "$SRC/$name"
}

build() {
  local name="$1"; shift
  cmake -S "$SRC/$name" -B "$PREFIX/build/$name" "${COMMON_ARGS[@]}" "$@"
  cmake --build "$PREFIX/build/$name" -j "$JOBS" --target install
}

fetch abseil https://github.com/abseil/abseil-cpp.git "$ABSL_TAG"
fetch flatbuffers https://github.com/google/flatbuffers.git "$FLATBUFFERS_TAG"
fetch re2 https://github.com/google/re2.git "$RE2_TAG"

build abseil -DABSL_PROPAGATE_CXX_STD=ON -DABSL_ENABLE_INSTALL=ON -DABSL_BUILD_TESTING=OFF
build flatbuffers -DFLATBUFFERS_BUILD_TESTS=OFF -DFLATBUFFERS_BUILD_FLATC=OFF
build re2 -DRE2_BUILD_TESTING=OFF

echo "core-equivalent deps installed under $PREFIX"
find "$PREFIX/lib" -name '*.a' | wc -l
