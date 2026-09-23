#!/usr/bin/env bash
#
# Builds the tokenizer shim as a static archive exporting only its C API.
#
# The HuggingFace tokenizers library is the reference implementation, and no C++
# alternative reproduces it: ONNX Runtime Extensions turns every accented
# character into [UNK], and the rest cover only half the model registry.
#
# Phases:  build_tokenizer.sh [build|isolate|all]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE="$ROOT/tokenizer"
# Separate target trees per platform; one shared tree would have each build
# overwrite the last.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/tokenizer/target}"
# Prebuilt archives are committed per platform so consumers never need cargo.
PLATFORM="${TOKENIZER_PLATFORM:-$(uname -s | tr 'A-Z' 'a-z')-$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/')}"
VENDOR="${TOKENIZER_VENDOR_DIR:-$ROOT/tokenizer/prebuilt/$PLATFORM}"
PHASE="${1:-all}"

EXPORTS=(cyborgdb_tokenizer_new cyborgdb_tokenizer_free cyborgdb_tokenizer_encode)

# rustc and the C compiler behind the onig crate both read this. Left unset,
# objects claim the build host's macOS version, and an older wheel refuses them.
if [[ "$(uname)" == "Darwin" ]]; then
  MACOSX_DEPLOYMENT_TARGET="$(python3 "$ROOT/scripts/version.py" macos)"
  export MACOSX_DEPLOYMENT_TARGET
fi

# Cargo needs the version in Cargo.toml, but versions.json is what a person
# edits, so the manifest is brought into line here rather than being a second
# place to remember.
sync_manifest() {
  local want
  want="$(python3 -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
print(json.loads((root / "versions.json").read_text())["tokenizers"]["version"])
' "$ROOT")"
  local have
  have="$(sed -n 's/.*tokenizers = { version = "\([^"]*\)".*/\1/p' "$CRATE/Cargo.toml")"
  if [[ "$want" != "$have" ]]; then
    echo "syncing Cargo.toml to tokenizers $want (was $have)"
    sed -i.bak "s|tokenizers = { version = \"[^\"]*\"|tokenizers = { version = \"$want\"|" \
      "$CRATE/Cargo.toml"
    rm -f "$CRATE/Cargo.toml.bak"
  fi
}

build() {
  sync_manifest

  # Neither LTO nor strip may be enabled in the release profile: LTO emits
  # bitcode that the partial link cannot read, and strip leaves it with no
  # symbol table to filter.
  cargo build --release --manifest-path "$CRATE/Cargo.toml"
  local archive="$CARGO_TARGET_DIR/release/libcyborgdb_tokenizer.a"
  [[ -f "$archive" ]] || { echo "cargo produced no archive" >&2; exit 1; }
  ls -lh "$archive"
}

# Demotes the Rust runtime to local symbols so it cannot collide with a consuming
# project. Objects are extracted and linked individually: passing the archive
# instead makes the linker discard the exported symbols, since nothing inside it
# references them. Rust names every member uniquely, so extraction is safe here.
isolate() {
  local archive="$CARGO_TARGET_DIR/release/libcyborgdb_tokenizer.a"
  local staging="$CARGO_TARGET_DIR/isolate"
  local out="$VENDOR/libcyborgdb_tokenizer.a"
  local keep="$VENDOR/exported_symbols.txt"

  mkdir -p "$VENDOR"
  rm -rf "$staging"; mkdir -p "$staging"
  ( cd "$staging" && ar x "$archive" )

  local total unique
  total="$(ar t "$archive" | wc -l | tr -d ' ')"
  unique="$(ar t "$archive" | sort -u | wc -l | tr -d ' ')"
  [[ "$total" == "$unique" ]] || {
    echo "archive has duplicate member names; extraction would lose code" >&2; exit 1; }

  local object="$staging/tokenizer_isolated.o"
  if [[ "$(uname)" == "Darwin" ]]; then
    printf '_%s\n' "${EXPORTS[@]}" > "$keep"
    # The unwind tables name the personality routine rather than pointing at
    # it, and linkers before Xcode 15 (conda's ld64 among them) resolve that name
    # only against a non-local symbol. So it leaves the first pass global and the
    # second makes it private extern: resolvable within the consumer's link, and
    # still never exported from it.
    local first="$CARGO_TARGET_DIR/isolate-first.o"
    { cat "$keep"; echo _rust_eh_personality; } > "$CARGO_TARGET_DIR/isolate-first.txt"
    local platform=(-arch "$(uname -m)"
      -platform_version macos "$MACOSX_DEPLOYMENT_TARGET" "$(xcrun --show-sdk-version)")
    ld -r "${platform[@]}" -exported_symbols_list "$CARGO_TARGET_DIR/isolate-first.txt" \
      -o "$first" "$staging"/*.o 2>&1 | grep -viE "was built for newer" || true
    ld -r "${platform[@]}" -exported_symbols_list "$keep" -keep_private_externs \
      -o "$object" "$first"
  else
    printf '%s\n' "${EXPORTS[@]}" > "$keep"
    # Mach-O's partial link drops unreachable code on its own; ELF needs to be
    # told, with the C API as the roots. Without this the archive is mostly
    # relocations and section names for code nothing can reach.
    local roots=()
    for symbol in "${EXPORTS[@]}"; do roots+=(-u "$symbol"); done
    ld -r --gc-sections "${roots[@]}" -o "$object" "$staging"/*.o
    objcopy --keep-global-symbols="$keep" "$object"
    # rustc embeds LLVM bitcode so consumers can do cross-crate LTO. We never
    # will, and it is over a third of the archive. The Mach-O partial link drops
    # it on its own; ELF keeps it.
    objcopy --remove-section=.llvmbc --remove-section=.llvmcmd "$object"
    # ELF partial linking keeps everything else the Mach-O path drops too. Safe
    # only after the symbols are localised.
    strip --strip-unneeded "$object"
  fi

  rm -f "$out"
  ar rcs "$out" "$object"

  # The failure that matters is the Rust runtime leaking out as strong exports.
  local found
  found="$("$ROOT/scripts/exported_symbols.sh" "$out")"
  echo "$out: $(wc -c < "$out") bytes"
  printf '  %s\n' $found
  [[ "$(echo "$found" | wc -l | tr -d ' ')" == "${#EXPORTS[@]}" ]] || {
    echo "expected exactly ${#EXPORTS[@]} exported symbols" >&2; exit 1; }
}

case "$PHASE" in
  --sync-only) sync_manifest ;;
  build)   build ;;
  isolate) isolate ;;
  all)     build && isolate ;;
  *)       echo "usage: $0 [build|isolate|all]" >&2; exit 2 ;;
esac
