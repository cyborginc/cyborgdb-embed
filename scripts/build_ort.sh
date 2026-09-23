#!/usr/bin/env bash
#
# Builds ONNX Runtime as static archives, merges them into one library, and
# verifies the result links. Output lands in .ort/install.
#
# Phases run independently:  build_ort.sh [fetch|build|merge|isolate|verify|vendor|all]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/.ort/src"
BUILD="${ORT_BUILD_DIR:-$ROOT/.ort/build}"
INSTALL="${ORT_INSTALL_DIR:-$ROOT/.ort/install}"

# Committed per platform so a consumer never builds ONNX Runtime. Microsoft's
# releases cannot be used: they are shared libraries, built without RTTI to match
# a consumer's ABI and without contrib ops, and are not symbol-isolated.
PLATFORM="${ORT_PLATFORM:-$(uname -s | tr 'A-Z' 'a-z')-$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/')}"
VENDOR="${ORT_VENDOR_DIR:-$ROOT/onnxruntime}"
CONFIG="${CONFIG:-Release}"
# versions.json is the single place either upstream version is written; this
# script, the CMake fetch and the release workflow all read it.
ORT_VERSION="${ORT_VERSION:-v$(python3 -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
print(json.loads((root / "versions.json").read_text())["onnxruntime"]["version"])
' "$ROOT")}"
PHASE="${1:-all}"

if [[ "$(uname)" == "Darwin" ]]; then
  JOBS="$(sysctl -n hw.ncpu)"
  # CMake reads this from the environment. Left unset, every object claims the
  # build host's macOS version, and a wheel targeting an older one refuses them.
  MACOSX_DEPLOYMENT_TARGET="$(python3 "$ROOT/scripts/version.py" macos)"
  export MACOSX_DEPLOYMENT_TARGET
else
  JOBS="$(nproc)"
fi

# Clones the pinned source if it is absent. Nothing else in the repository
# depends on it; the committed archives are what consumers build against.
fetch() {
  [[ -d "$SRC" ]] && return 0
  mkdir -p "$(dirname "$SRC")"
  git clone --depth 1 --branch "$ORT_VERSION" --recurse-submodules \
    --shallow-submodules https://github.com/microsoft/onnxruntime.git "$SRC"
}

build() {
  fetch

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

  # The environment only seeds a fresh cache; this also holds for a reused one.
  if [[ -n "${MACOSX_DEPLOYMENT_TARGET:-}" ]]; then
    defines+=("CMAKE_OSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET")
  fi

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
# Archives fully contained in another are dropped first, so the partial link
# never sees two definitions of one symbol.
merge() {
  local out="$INSTALL/lib/libonnxruntime_merged.a"
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
#
# The C API is the link's only root: members nothing reaches from it are left
# out. A kernel reachable only through code that was left out fails at inference
# rather than at link, so the golden and parity runs gate this, not the smoke link.
isolate() {
  local merged="$INSTALL/lib/libonnxruntime_merged.a"
  local out="$INSTALL/lib/libonnxruntime_isolated.a"
  local keep="$INSTALL/lib/exported_symbols.txt"
  local object="$BUILD/ort_isolated.o"
  local roots=() symbol

  if [[ "$(uname)" == "Darwin" ]]; then
    # Mach-O symbol names carry a leading underscore.
    printf '_OrtGetApiBase\n_OrtSessionOptionsAppendExecutionProvider_CPU\n' > "$keep"
    while read -r symbol; do roots+=(-u "$symbol"); done < "$keep"
    ld -r -arch "$(uname -m)" \
      -platform_version macos "$MACOSX_DEPLOYMENT_TARGET" "$(xcrun --show-sdk-version)" \
      "${roots[@]}" -exported_symbols_list "$keep" \
      -o "$object" "$merged" 2>&1 | grep -v "not 4-byte aligned" || true
  else
    printf 'OrtGetApiBase\nOrtSessionOptionsAppendExecutionProvider_CPU\n' > "$keep"
    while read -r symbol; do roots+=(-u "$symbol"); done < "$keep"
    # Some hand-written assembly carries no .note.GNU-stack marker, so the
    # linker conservatively marks the stack executable and every consumer
    # inherits it. Assert non-executable here rather than leaving it to them.
    ld -r -z noexecstack "${roots[@]}" -o "$object" "$merged"

    # Localize strong symbols only, never weak ones.
    #
    # Weak symbols are the COMDAT template and inline instantiations. They
    # deduplicate safely against a consumer's identical copies, and demoting one
    # makes the linker discard its group while references stay live — a failure
    # that appears only when a program reaches that code, not at a smoke link.
    # The collisions that matter are strong: 1,748 of them against a consuming
    # project on this platform, and none weak.
    nm -g --defined-only "$object" \
      | awk '$2 ~ /^[TDBR]$/ {print $3}' | sort -u > "$BUILD/defined.txt"
    comm -23 "$BUILD/defined.txt" <(sort -u "$keep") > "$BUILD/localize.txt"
    echo "  localizing $(wc -l < "$BUILD/localize.txt") strong symbols"
    objcopy --localize-symbols="$BUILD/localize.txt" "$object"
  fi

  [[ -f "$object" ]] || { echo "partial link produced no object" >&2; exit 1; }
  rm -f "$out"
  ar rcs "$out" "$object"
  ls -lh "$out"
}

# A successful link against the isolated archive is the check that the archive
# set is complete and the partial link kept what the C API needs; undefined
# symbols here mean merge() or isolate() dropped something.
verify() {
  local tmp; tmp="$(mktemp -d)"
  cat > "$tmp/smoke.cpp" <<'EOF'
#include <onnxruntime_c_api.h>
#include <cstdio>

int main() {
  const OrtApiBase* base = OrtGetApiBase();
  if (base == nullptr) return 1;
  const OrtApi* api = base->GetApi(ORT_API_VERSION);
  if (api == nullptr) return 1;

  OrtEnv* env = nullptr;
  if (api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "verify", &env) != nullptr) return 1;
  OrtSessionOptions* options = nullptr;
  if (api->CreateSessionOptions(&options) != nullptr) return 1;

  // Deliberately loads a file that is not a graph: the failure path still pulls
  // in protobuf parsing and the logger, which a version query does not.
  OrtSession* session = nullptr;
  OrtStatus* status = api->CreateSession(env, "/nonexistent.onnx", options, &session);
  if (status != nullptr) api->ReleaseStatus(status);

  api->ReleaseSessionOptions(options);
  api->ReleaseEnv(env);
  std::printf("%s\n", base->GetVersionString());
  return 0;
}
EOF
  local libs=(-framework Foundation -framework Accelerate)
  [[ "$(uname)" == "Darwin" ]] || libs=(-lpthread -ldl -lm -lstdc++)

  # Loading a graph reaches protobuf, the logger and the session machinery. A
  # program that only asks for the API version links even when those are broken.

  c++ -std=c++17 -I"$INSTALL/include" "$tmp/smoke.cpp" \
    "$INSTALL/lib/libonnxruntime_isolated.a" "${libs[@]}" -o "$tmp/smoke"
  "$tmp/smoke"
  rm -rf "$tmp"
}

# Copies the isolated archive and headers into the tree. Headers are identical
# across platforms, so only the archive is per-platform.
vendor() {
  mkdir -p "$VENDOR/prebuilt/$PLATFORM" "$VENDOR/include"
  cp "$INSTALL/lib/libonnxruntime_isolated.a" \
     "$VENDOR/prebuilt/$PLATFORM/libonnxruntime.a"
  cp "$INSTALL"/include/*.h "$VENDOR/include/"

  # Only strong exports are checked. Weak symbols are COMDAT template and inline
  # instantiations, which deduplicate safely against a consumer's identical
  # copies; localizing them discards their groups and breaks the link. The
  # collisions that matter are strong.
  local archive="$VENDOR/prebuilt/$PLATFORM/libonnxruntime.a"
  local strong weak
  strong="$("$ROOT/scripts/exported_symbols.sh" "$archive")"
  weak="$(nm -g --defined-only "$archive" 2>/dev/null \
    | awk '$2 ~ /^[WVS]$/' | wc -l | tr -d ' ')"
  echo "$archive: $(wc -c < "$archive") bytes, $weak weak"
  printf '  %s\n' $strong
  [[ "$(echo "$strong" | wc -l | tr -d ' ')" -le 2 ]] || {
    echo "archive exports strong symbols beyond the public C API" >&2; exit 1; }
}

case "$PHASE" in
  fetch)   fetch ;;
  build)   build ;;
  merge)   merge ;;
  isolate) isolate ;;
  verify)  verify ;;
  vendor)  vendor ;;
  all)     build && merge && isolate && verify && vendor ;;
  *)       echo "usage: $0 [fetch|build|merge|isolate|verify|vendor|all]" >&2; exit 2 ;;
esac
