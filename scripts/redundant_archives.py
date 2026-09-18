#!/usr/bin/env python3
"""Print archives whose every member is already present in another archive.

A whole-archive partial link rejects duplicate definitions, and static library
sets routinely ship a "lite" or "proto" archive whose objects are also compiled
into a larger one. Dropping the redundant archive is safe; dropping the wrong one
removes code that only fails to link much later.
"""

import hashlib
import subprocess
import sys


# Archive symbol tables listed as members: "/" and "//" from GNU ar, "__.SYMDEF"
# from BSD ar. They differ between archives by construction, so treating them as
# content makes every comparison fail.
INDEX_MEMBERS = ("/", "//", "__.SYMDEF", "__.SYMDEF SORTED")


def members(archive):
    """Member names, without archive index entries or GNU ar's trailing slashes."""
    out = subprocess.run(["ar", "t", archive], capture_output=True, text=True).stdout
    return {line.rstrip("/") for line in out.splitlines()
            if line.strip() and line not in INDEX_MEMBERS}


def digest(archive, member):
    data = subprocess.run(
        ["ar", "p", archive, member], capture_output=True
    ).stdout
    return hashlib.sha256(data).hexdigest() if data else None


def main(paths):
    names = {p: members(p) for p in paths}
    redundant = set()
    # Two archives with identical member sets each contain the other, so dropping
    # every archive that is contained somewhere drops both and deletes the code.
    # Keeping one requires an order: never fold into an archive already dropped,
    # and on a tie keep the first by name.
    for a in sorted(names):
        ma = names[a]
        if not ma:
            continue
        for b in sorted(names):
            if b == a or b in redundant or not ma <= names[b]:
                continue
            if len(ma) == len(names[b]) and b > a:
                continue
            # Name containment is only a candidate; confirm the bytes match.
            if all(digest(a, m) == digest(b, m) for m in sorted(ma)):
                redundant.add(a)
                break
    for path in sorted(redundant):
        print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
