#!/usr/bin/env python3
"""Derive the binary corpus fixtures the C++ golden test reads.

Texts are NUL-separated rather than JSON: the corpus deliberately contains
newlines, empty strings and control characters, and a C++ test should not be
parsing escapes to get at them.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTDATA = ROOT / "tests" / "data"

rows = [json.loads(line) for line in (TESTDATA / "corpus.jsonl").open()]
(TESTDATA / "corpus.bin").write_bytes(b"".join(r["text"].encode() + b"\0" for r in rows))
(TESTDATA / "corpus.groups").write_text("".join(r.get("group", "real") + "\n" for r in rows))
print(f"corpus.bin, corpus.groups: {len(rows)} rows")
