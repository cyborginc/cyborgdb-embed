#!/usr/bin/env python3
"""Reads versions.json, the single place upstream versions are written.

    version.py onnxruntime          -> 1.30.0
    version.py tokenizers --tag     -> tokenizers-0.23.2-1
    version.py macos                -> 14.0
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFIX = {"onnxruntime": "ort", "tokenizers": "tokenizers"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=sorted(PREFIX) + ["macos"])
    parser.add_argument("--tag", action="store_true",
                        help="print the release tag rather than the version")
    args = parser.parse_args()

    versions = json.loads((ROOT / "versions.json").read_text())
    if args.component == "macos":
        print(versions["macos_deployment_target"])
        return 0

    entry = versions[args.component]
    if args.tag:
        print(f"{PREFIX[args.component]}-{entry['version']}-{entry['build']}")
    else:
        print(entry["version"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
