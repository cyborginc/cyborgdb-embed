#!/usr/bin/env python3
"""Writes archive digests into cmake/artifacts.cmake.

    record_digests.py digest-linux-amd64.txt digest-linux-arm64.txt ...

Each file holds `shasum -a 256` output for archives named lib<name>-<platform>.a,
as release-artifacts.yml writes them.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PIN = ROOT / "cmake" / "artifacts.cmake"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    text = PIN.read_text()
    for digest_file in sys.argv[1:]:
        for line in pathlib.Path(digest_file).read_text().splitlines():
            digest, archive = line.split()
            name = pathlib.Path(archive).stem.removeprefix("lib")
            entry = re.compile(rf'("{re.escape(name)} +)(?:[0-9a-fA-F]{{64}}|PENDING)"')
            text, count = entry.subn(rf'\g<1>{digest}"', text)
            if count != 1:
                sys.exit(f"{name}: expected one entry in {PIN.name}, found {count}")
            print(f"{name}  {digest}")
    PIN.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
