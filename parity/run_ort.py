"""Middle arm: ONNX Runtime and the tokenizers library, driven from Python.

Runs the same graph and the same tokenizer as the C++ library, so the gap
between this arm and the C++ one is attributable to the C++ alone. It also
establishes the achievable ceiling per model: where the published ONNX graph
does not reproduce the PyTorch model exactly, this arm shows by how much, and
the C++ arm matching it is a pass rather than a failure.
"""

import argparse
import os
import pathlib
import sys
import urllib.request

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

import common


def cache_dir():
    if env := os.environ.get("CYBORGDB_EMBED_CACHE"):
        return pathlib.Path(env)
    if env := os.environ.get("XDG_CACHE_HOME"):
        return pathlib.Path(env) / "cyborgdb-embed"
    return pathlib.Path.home() / ".cache" / "cyborgdb-embed"


def fetch(entry, repo_path, destination):
    """Mirrors the library's cache layout so both read the same bytes."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://huggingface.co/{entry['id']}/resolve/"
           f"{entry['revision']}/{repo_path}")
    urllib.request.urlretrieve(url, destination)
    return destination


def pool(hidden, mask, mode):
    if mode == "cls_token":
        return hidden[:, 0]
    if mode == "mean_tokens":
        weights = mask[..., None].astype(np.float32)
        return (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
    if mode == "max_tokens":
        return np.where(mask[..., None].astype(bool), hidden, -np.inf).max(axis=1)
    raise ValueError(f"unsupported pooling: {mode}")


def encode(entry, texts, prefix, batch_size=32):
    root = cache_dir() / common.slug(entry) / entry["revision"]
    graph = fetch(entry, entry["onnx_path"], root / pathlib.Path(entry["onnx_path"]).name)
    tokenizer = Tokenizer.from_file(str(fetch(entry, "tokenizer.json", root / "tokenizer.json")))

    # A fixed padding strategy in tokenizer.json pads every short sentence to a
    # constant length: same output, many times the work. Pad to the batch instead.
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=entry["max_seq_length"])

    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    expected = {i.name for i in session.get_inputs()}

    out = []
    for start in range(0, len(texts), batch_size):
        chunk = [prefix + t for t in texts[start:start + batch_size]]
        encodings = [tokenizer.encode(t) for t in chunk]
        width = max((len(e.ids) for e in encodings), default=1) or 1

        ids = np.zeros((len(chunk), width), dtype=np.int64)
        mask = np.zeros((len(chunk), width), dtype=np.int64)
        for row, enc in enumerate(encodings):
            ids[row, :len(enc.ids)] = enc.ids
            mask[row, :len(enc.attention_mask)] = enc.attention_mask

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in expected:
            feed["token_type_ids"] = np.zeros_like(ids)
        feed = {k: v for k, v in feed.items() if k in expected}

        hidden = session.run(None, feed)[0]
        vectors = pool(hidden, mask, entry["pooling"])
        if entry["normalize"]:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.clip(norms, 1e-12, None)
        out.append(vectors.astype(np.float32))

    return np.vstack(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    entry = common.model(args.model)
    texts, _ = common.corpus()

    for kind, prefix in (("document", entry.get("doc_prefix") or ""),
                         ("query", entry.get("query_prefix") or "")):
        vectors = encode(entry, texts, prefix, args.batch)
        out = common.GOLDEN / common.slug(entry) / f"ort-{kind}.f32"
        common.write_vectors(out, vectors, {
            "arm": "ort-python", "kind": kind, "model": entry["id"],
            "revision": entry["revision"], "prefix": prefix,
        })
        print(f"  {out.relative_to(common.ROOT)}  {vectors.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
