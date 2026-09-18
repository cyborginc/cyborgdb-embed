#!/usr/bin/env python3
"""Resolve upstream revisions and file digests into registry.yaml.

Pins each model to a commit and a content hash so a repository that changes
underneath us fails loudly instead of returning different vectors. Digests come
from the LFS object id the API already exposes, so nothing is downloaded.

Edits the pinned lines in place, leaving comments and ordering untouched.
"""

import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request

API = "https://huggingface.co/api/models"
REGISTRY = os.path.join(os.path.dirname(os.path.dirname(__file__)), "registry.yaml")


def fetch(url):
    request = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def digest_of(model_id, commit, path):
    """SHA-256 of a repository file, by downloading it.

    Only for files small enough that HuggingFace keeps them in git rather than
    LFS, so no content hash is exposed through the API.
    """
    url = f"https://huggingface.co/{model_id}/resolve/{commit}/{path}"
    request = urllib.request.Request(url)
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return hashlib.sha256(response.read()).hexdigest()


def resolve(model_id, onnx_path):
    """Return (commit, sha256, size) for one model's graph."""
    commit = fetch(f"{API}/{model_id}")["sha"]

    directory, _, filename = onnx_path.rpartition("/")
    tree = fetch(f"{API}/{model_id}/tree/{commit}/{directory}")
    entry = next((f for f in tree if f["path"] == onnx_path), None)
    if entry is None:
        raise LookupError(f"{onnx_path} is not in {model_id} at {commit[:12]}")

    lfs = entry.get("lfs") or {}
    if not lfs.get("oid"):
        # Small files are stored in git rather than LFS, so the API exposes a
        # blob sha1 instead of a content hash. Nothing that size is a real model.
        raise LookupError(f"{model_id}: {onnx_path} is not LFS-backed")

    sidecar = next((f for f in tree if f["path"] == onnx_path + "_data"), None)
    tokenizer = digest_of(model_id, commit, "tokenizer.json")
    return commit, lfs["oid"], entry.get("size", 0), sidecar is not None, tokenizer


def main():
    lines = open(REGISTRY).read().splitlines(keepends=True)
    current = None
    failures = []
    pinned = 0

    for index, line in enumerate(lines):
        model = re.match(r"\s*-\s*id:\s*(\S+)", line)
        if model:
            current = {"id": model.group(1), "onnx_path": None, "rev": None,
                       "sha": None, "tokenizer": None}
            # The path is needed before the pin lines, and it always precedes
            # them, so collect it by scanning ahead within this block.
            for ahead in lines[index:]:
                if ahead.strip().startswith("- id:") and ahead is not line:
                    break
                path = re.match(r"\s*onnx_path:\s*(\S+)", ahead)
                if path:
                    current["onnx_path"] = path.group(1)
                    break
            try:
                commit, digest, size, sidecar, tokenizer = resolve(
                    current["id"], current["onnx_path"])
            except (urllib.error.URLError, LookupError, KeyError) as error:
                failures.append(f"{current['id']}: {error}")
                current = None
                continue
            current["rev"], current["sha"] = commit, digest
            current["tokenizer"] = tokenizer
            note = "  (has external-data sidecar)" if sidecar else ""
            print(f"  {current['id']:<48} {commit[:12]} {size / 1e6:8.1f} MB{note}")
            pinned += 1
            continue

        if current:
            if re.match(r"\s*revision:", line):
                lines[index] = re.sub(r'"[^"]*"', f'"{current["rev"]}"', line, count=1)
            elif re.match(r"\s*onnx_sha256:", line):
                lines[index] = re.sub(r'"[^"]*"', f'"{current["sha"]}"', line, count=1)
            elif re.match(r"\s*tokenizer_sha256:", line):
                lines[index] = re.sub(r'"[^"]*"', f'"{current["tokenizer"]}"', line, count=1)

    open(REGISTRY, "w").write("".join(lines))
    print(f"\npinned {pinned} models")
    for failure in failures:
        print(f"FAILED {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
