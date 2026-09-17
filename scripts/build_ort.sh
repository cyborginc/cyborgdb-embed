#!/usr/bin/env bash
#
# Builds ONNX Runtime as static archives, merges them into one library, and
# verifies the result links. Output lands in .ort/install.
#
# Phases run independently:  build_ort.sh [build|merge|isolate|verify|all]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/.ort/src"
BUILD="${ORT_BUILD_DIR:-$ROOT/.ort/build}"
INSTALL="${ORT_INSTALL_DIR:-$ROOT/.ort/install}"
CONFIG="${CONFIG:-Release}"
PHASE="${1:-all}"

if [[ "$(uname)" == "Darwin" ]]; then
  JOBS="$(sysctl -n hw.ncpu)"
else
  JOBS="$(nproc)"
fi

build() {
  local defines=(
    # --skip_tests only skips running them; this is what stops them building.
    onnxruntime_BUILD_UNIT_TESTS=OFF
    # Without this CMake prefers a host protobuf over the vendored protoc and
    # fails later with unreadable generated-code errors.
    FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER
    # ORT defaults Release to -O3. Size matters more here than the last few
    # percent of throughput, which lives in hand-written kernels regardless.
    "CMAKE_CXX_FLAGS_RELEASE=-Os -DNDEBUG"
    "CMAKE_C_FLAGS_RELEASE=-Os -DNDEBUG"
  )

  # Point ORT at an already-built dependency tree instead of its own pin.
  if [[ -n "${ABSL_SOURCE_DIR:-}" ]]; then
    defines+=("FETCHCONTENT_SOURCE_DIR_ABSEIL_CPP=$ABSL_SOURCE_DIR")
  fi
  if [[ -n "${FLATBUFFERS_SOURCE_DIR:-}" ]]; then
    defines+=("FETCHCONTENT_SOURCE_DIR_FLATBUFFERS=$FLATBUFFERS_SOURCE_DIR")
  fi
  if [[ -n "${RE2_SOURCE_DIR:-}" ]]; then
    defines+=("FETCHCONTENT_SOURCE_DIR_RE2=$RE2_SOURCE_DIR")
  fi
  if [[ -n "${PROTOBUF_SOURCE_DIR:-}" ]]; then
    defines+=("FETCHCONTENT_SOURCE_DIR_PROTOBUF=$PROTOBUF_SOURCE_DIR")
  fi

  # Omitting --build_shared_lib is what selects a static build; the flag is
  # store-true and passing it a value is a parse error.
  #
  # Deliberately absent:
  #   --disable_rtti            RTTI must match the consuming build, since both
  #                             feed abseil's ABI options
  #   --minimal_build           drops ONNX-protobuf loading, forcing .ort files
  #   --include_ops_by_config   pins the build to the opset versions seen in one
  #                             corpus, so ordinary models fail on ordinary ops
  #   LTO                       measured larger, not smaller
  python3 "$SRC/tools/ci_build/build.py" \
    --build_dir "$BUILD" \
    --config "$CONFIG" \
    --parallel "$JOBS" \
    --cmake_generator "Unix Makefiles" \
    --skip_tests \
    --compile_no_warning_as_error \
    --no_telemetry \
    --disable_ml_ops \
    --disable_contrib_ops \
    --cmake_extra_defines "${defines[@]}"

  # build.py logs failures and still exits 0, so check for output rather than
  # trusting the status.
  [[ -d "$BUILD/$CONFIG" ]] || {
    echo "ONNX Runtime build produced no $BUILD/$CONFIG directory" >&2; exit 1; }

  # re2 is only reachable from the test targets. With tests off it is never
  # built, and the gap first shows up as an undefined symbol at downstream link.
  cmake --build "$BUILD/$CONFIG" --target re2 -j "$JOBS"

  local built
  built="$(find "$BUILD/$CONFIG" -name '*.a' | wc -l)"
  [[ "$built" -gt 1 ]] || { echo "only $built archives built" >&2; exit 1; }
}

# The static target is an INTERFACE library, so no single archive exists.
#
# Merge archives rather than extracted objects: several archives hold two
# different members under one name (a generic env.cc.o and a platform-specific
# one), and extracting flattens them onto each other, silently dropping code that
# only fails to link much later.
#
# Archives fully contained in another are dropped first: a whole-archive partial
# link rejects the duplicate symbols they would contribute.
merge() {
  local out="$INSTALL/lib/onnxruntime_merged.a"
  mkdir -p "$INSTALL/lib" "$INSTALL/include"

  local archives=()
  while IFS= read -r archive; do
    archives+=("$archive")
  done < <(find "$BUILD/$CONFIG" -name '*.a' \
    ! -path '*gtest*' ! -path '*gmock*' ! -path '*benchmark*' ! -path '*test*' \
    | sort -u)
  [[ ${#archives[@]} -gt 0 ]] || { echo "no archives under $BUILD/$CONFIG" >&2; exit 1; }

  local redundant keep=()
  redundant="$(python3 "$ROOT/scripts/redundant_archives.py" "${archives[@]}")"
  local archive
  for archive in "${archives[@]}"; do
    if ! grep -qxF "$archive" <<< "$redundant"; then
      keep+=("$archive")
    fi
  done
  archives=("${keep[@]}")
  echo "merging ${#archives[@]} archives"

  rm -f "$out"
  if [[ "$(uname)" == "Darwin" ]]; then
    libtool -static -o "$out" "${archives[@]}" 2>/dev/null
  else
    { echo "create $out"
      printf 'addlib %s\n' "${archives[@]}"
      echo save; echo end; } | ar -M
  fi

  cp "$SRC"/include/onnxruntime/core/session/*.h "$INSTALL/include/"
  ls -lh "$out"
}

# Demotes everything except the public C API to private external, so ONNX
# Runtime's vendored abseil, re2 and flatbuffers cannot bind against a consumer's
# own copies. A partial link resolves ORT's internal references first, which is
# what makes hiding the rest safe.
isolate() {
  local merged="$INSTALL/lib/onnxruntime_merged.a"
  local out="$INSTALL/lib/onnxruntime_isolated.a"
  local keep="$INSTALL/lib/exported_symbols.txt"
  local object="$BUILD/ort_isolated.o"

  if [[ "$(uname)" == "Darwin" ]]; then
    # Mach-O symbol names carry a leading underscore.
    printf '_OrtGetApiBase\n_OrtSessionOptionsAppendExecutionProvider_CPU\n' > "$keep"
    ld -r -arch "$(uname -m)" \
      -platform_version macos "$(sw_vers -productVersion | cut -d. -f1).0" "$(xcrun --show-sdk-version)" \
      -all_load -exported_symbols_list "$keep" \
      -o "$object" "$merged" 2>&1 | grep -v "not 4-byte aligned" || true
  else
    printf 'OrtGetApiBase\nOrtSessionOptionsAppendExecutionProvider_CPU\n' > "$keep"
    ld -r --whole-archive "$merged" --no-whole-archive -o "$object"
    objcopy --keep-global-symbols="$keep" "$object"
  fi

  [[ -f "$object" ]] || { echo "partial link produced no object" >&2; exit 1; }
  rm -f "$out"
  ar rcs "$out" "$object"
  ls -lh "$out"
}

# A successful link against the merged archive is the check that the archive set
# is complete; undefined symbols here mean merge() missed something.
verify() {
  local tmp; tmp="$(mktemp -d)"
  cat > "$tmp/smoke.cpp" <<'EOF'
#include <onnxruntime_c_api.h>
#include <cstdio>

int main() {
  const OrtApiBase* base = OrtGetApiBase();
  if (base == nullptr) return 1;
  std::printf("%s\n", base->GetVersionString());
  return base->GetApi(ORT_API_VERSION) != nullptr ? 0 : 1;
}
EOF
  local libs=(-framework Foundation -framework Accelerate)
  [[ "$(uname)" == "Darwin" ]] || libs=(-lpthread -ldl -lm -lstdc++)

  c++ -std=c++17 -I"$INSTALL/include" "$tmp/smoke.cpp" \
    "$INSTALL/lib/onnxruntime_merged.a" "${libs[@]}" -o "$tmp/smoke"
  "$tmp/smoke"
  rm -rf "$tmp"
}

case "$PHASE" in
  build)   build ;;
  merge)   merge ;;
  isolate) isolate ;;
  verify)  verify ;;
  all)     build && merge && isolate && verify ;;
  *)       echo "usage: $0 [build|merge|isolate|verify|all]" >&2; exit 2 ;;
esac
