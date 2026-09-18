"""Reference arm: sentence-transformers on PyTorch.

Whatever this produces is ground truth. Its output is committed as the golden
vectors every other arm is measured against.
"""

import argparse
import sys

import numpy as np
import common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="registry id or enum name")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    entry = common.model(args.model)
    texts, _ = common.corpus()

    from sentence_transformers import SentenceTransformer

    # Pinned to the same revision the library resolves, so the reference and the
    # graph under test come from one commit.
    model = SentenceTransformer(entry["id"], revision=entry["revision"])

    prefixes = {"document": entry.get("doc_prefix") or "",
                "query": entry.get("query_prefix") or ""}
    for kind, prefix in prefixes.items():
        vectors = model.encode([prefix + t for t in texts],
                               batch_size=args.batch,
                               convert_to_numpy=True,
                               normalize_embeddings=bool(entry["normalize"]))
        out = common.GOLDEN / common.slug(entry) / f"reference-{kind}.f32"
        common.write_vectors(out, np.asarray(vectors), {
            "arm": "reference", "kind": kind, "model": entry["id"],
            "revision": entry["revision"], "prefix": prefix,
        })
        print(f"  {out.relative_to(common.ROOT)}  {vectors.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
