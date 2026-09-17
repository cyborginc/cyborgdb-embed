#!/usr/bin/env python3
"""Report symbols defined by two sets of static archives that would collide.

Duplicate weak symbols do not produce a link error; the linker keeps one
definition and discards the rest. Which one survives depends on link order, so
the failure is silent and can differ between build configurations.
"""

import argparse
import collections
import platform
import subprocess
import sys

# Mangled names carry the namespace text, so matching does not need demangling.
FAMILIES = {
    "abseil": ("absl",),
    "flatbuffers": ("flatbuffers",),
    "re2": ("re2", "RE2"),
    "protobuf": ("protobuf",),
}

# GNU nm type letters for symbols this object defines and exports.
ELF_BINDING = {"T": "strong", "D": "strong", "B": "strong", "R": "strong",
               "W": "weak", "V": "weak", "C": "common"}


# Per-object compiler artifacts. Every object emits these under the same names
# and the linker handles them; counting them as collisions is noise.
ARTIFACT_PREFIXES = ("GCC_except_table", "l_.str", "_OBJC_")


def _is_artifact(name):
    return name.startswith(ARTIFACT_PREFIXES)


def _run(argv):
    return subprocess.run(argv, capture_output=True, text=True, check=False).stdout


def _macho_symbols(paths):
    """Binding per symbol, from nm -m.

    Plain nm collapses weak definitions and hidden symbols into the same type
    letters, and those decide whether a duplicate is a link error, a silent
    substitution, or already invisible to the other side.
    """
    symbols = {}
    for path in paths:
        for line in _run(["nm", "-m", path]).splitlines():
            # "non-external" contains "external": test it first, or every local
            # symbol is misread as a global one.
            if "(undefined)" in line or "non-external" in line:
                continue
            if "private external" in line:
                binding = "hidden"
            elif "weak external" in line:
                binding = "weak"
            elif "external" in line:
                binding = "strong"
            else:
                continue
            name = line.split()[-1]
            if not _is_artifact(name):
                symbols[name] = binding
    return symbols


def _elf_symbols(paths):
    symbols = {}
    for path in paths:
        for line in _run(["nm", "-g", "--defined-only", path]).splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            binding = ELF_BINDING.get(parts[-2])
            if binding and not _is_artifact(parts[-1]):
                symbols[parts[-1]] = binding
    return symbols


def defined_symbols(paths):
    if platform.system() == "Darwin":
        return _macho_symbols(paths)
    return _elf_symbols(paths)


def classify(name):
    for family, needles in FAMILIES.items():
        if any(needle in name for needle in needles):
            return family
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", nargs="+", required=True, metavar="ARCHIVE")
    parser.add_argument("--b", nargs="+", required=True, metavar="ARCHIVE")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--show", type=int, default=0, help="print N example symbols per family")
    args = parser.parse_args()

    a = defined_symbols(args.a)
    b = defined_symbols(args.b)
    shared = sorted(set(a) & set(b))

    print(f"{args.label_a}: {len(a)} defined   {args.label_b}: {len(b)} defined")
    print(f"overlap: {len(shared)}\n")

    if not shared:
        return 0

    counts = collections.defaultdict(collections.Counter)
    for name in shared:
        counts[classify(name)][a[name]] += 1

    print(f"{'family':<14}{'total':>8}{'strong':>9}{'weak':>7}{'hidden':>8}")
    for family, breakdown in sorted(counts.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(breakdown.values())
        print(
            f"{family:<14}{total:>8}{breakdown['strong']:>9}"
            f"{breakdown['weak']:>7}{breakdown['hidden']:>8}"
        )

    if args.show:
        for family in counts:
            print(f"\n{family}:")
            examples = [n for n in shared if classify(n) == family][: args.show]
            for name in examples:
                readable = _run(["c++filt", name]).strip() or name
                print(f"  [{a[name]:<6}] {readable}")

    return 1


if __name__ == "__main__":
    sys.exit(main())
