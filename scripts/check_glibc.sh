#!/usr/bin/env bash
# Fails when a Linux binary binds a newer glibc, libstdc++ or libgcc symbol than
# manylinux_2_28 allows, which is what auditwheel would reject in a wheel that
# links the archives.
#
# The caps are auditwheel's manylinux_2_28 policy, i.e. AlmaLinux 8's system
# libraries. gcc-toolset links its newer libstdc++ parts statically, so a newer
# toolset is fine as long as nothing binds past these.
set -euo pipefail

binary=${1:?usage: check_glibc.sh <binary>}
declare -A cap=([GLIBC]=2.28 [GLIBCXX]=3.4.24 [CXXABI]=1.3.11 [GCC]=7.0.0)

refs=$(objdump -T "$binary" \
  | grep -oE '\b(GLIBC|GLIBCXX|CXXABI|GCC)_[0-9]+(\.[0-9]+)*\b' | sort -u)
# A binary with no versioned references at all means objdump read nothing.
[ -n "$refs" ] || { echo "no versioned symbols in $binary"; exit 1; }

status=0
for lib in GLIBC GLIBCXX CXXABI GCC; do
  newest=$(grep -E "^${lib}_" <<< "$refs" | sed "s/^${lib}_//" | sort -V | tail -1)
  [ -n "$newest" ] || continue
  if [ "$(printf '%s\n%s\n' "$newest" "${cap[$lib]}" | sort -V | tail -1)" = "${cap[$lib]}" ]; then
    echo "${lib}_${newest} (cap ${cap[$lib]}): ok"
  else
    echo "${lib}_${newest} exceeds manylinux_2_28's ${lib}_${cap[$lib]}:"
    objdump -T "$binary" | grep -E "\b${lib}_" | sort -V -k5 | tail -5 | sed 's/^/  /'
    status=1
  fi
done
exit "$status"
