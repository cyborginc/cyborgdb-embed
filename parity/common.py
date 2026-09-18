"""Shared pieces of the parity harness: corpus, registry, and vector files."""

import json
import pathlib

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "testdata" / "corpus.jsonl"
REGISTRY = ROOT / "registry.yaml"
GOLDEN = ROOT / "testdata" / "golden"


def corpus():
    """Frozen evaluation corpus: (texts, groups).

    Groups separate real prose from deliberately degenerate inputs so the two can
    be scored apart.
    """
    rows = [json.loads(line) for line in CORPUS.open()]
    return [r["text"] for r in rows], [r.get("group", "real") for r in rows]


def models():
    return yaml.safe_load(REGISTRY.read_text())["models"]


def model(model_id):
    for entry in models():
        if entry["id"] == model_id or entry["enum"] == model_id:
            return entry
    raise KeyError(f"{model_id} is not in the registry")


def slug(entry):
    return entry["id"].replace("/", "_")


def write_vectors(path, matrix, meta):
    """Raw little-endian float32 plus a JSON sidecar.

    Every arm writes the same format, so the comparison never needs to know which
    one produced a file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix.astype("<f4").tofile(path)
    path.with_suffix(".json").write_text(json.dumps(
        {**meta, "rows": int(matrix.shape[0]), "dim": int(matrix.shape[1])}, indent=2))


def read_vectors(path):
    meta = json.loads(path.with_suffix(".json").read_text())
    flat = np.fromfile(path, dtype="<f4")
    return flat.reshape(meta["rows"], meta["dim"]), meta
