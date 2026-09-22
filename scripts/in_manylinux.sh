#!/usr/bin/env bash
# Runs a command in the manylinux build image for this machine's architecture,
# with the repository mounted at /work.
#
#   scripts/in_manylinux.sh ./scripts/build_ort.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$(uname -m)" in
  x86_64|amd64)  arch=x86_64 ;;
  arm64|aarch64) arch=aarch64 ;;
  *) echo "no manylinux image for $(uname -m)" >&2; exit 1 ;;
esac

image="cyborgdb-embed-manylinux:$arch"
docker build -q -t "$image" --build-arg ARCH="$arch" "$ROOT/scripts/docker" >/dev/null

# The invoking user's ids, so everything written into the checkout stays theirs.
# Build-script overrides pass through.
env_args=()
while IFS= read -r name; do env_args+=(-e "$name"); done \
  < <(compgen -e | grep -E '^(ORT|TOKENIZER|CYBORGDB)_' || true)

tty_args=()
[ -t 0 ] && tty_args=(-it)

exec docker run --rm ${tty_args[@]+"${tty_args[@]}"} --user "$(id -u):$(id -g)" \
  ${env_args[@]+"${env_args[@]}"} -v "$ROOT:/work" -w /work "$image" "$@"
