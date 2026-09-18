#!/usr/bin/env python3
"""Run every arm over every model and report the verdicts.

The reference arm needs torch and downloads full weight trees, so this is the
nightly job rather than the per-PR one. Results are written back into the
registry: a model's parity verdict is generated, never hand-written.
"""

import argparse
import json
import pathlib
import subprocess
import sys

import common
import compare

HERE = pathlib.Path(__file__).resolve().parent


def run(command, env=None):
    result = subprocess.run(command, cwd=HERE, capture_output=True, text=True,
                            env=env)
    return result.returncode, (result.stdout + result.stderr)


def corpus_blob(path):
    texts, _ = common.corpus()
    path.write_bytes(b"".join(t.encode() + b"\0" for t in texts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", help="registry ids; default all")
    parser.add_argument("--reference-python", required=True,
                        help="interpreter with sentence-transformers installed")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter with onnxruntime and tokenizers")
    parser.add_argument("--embed-corpus", default=str(common.ROOT / "build" / "embed_corpus"))
    parser.add_argument("--skip-reference", action="store_true",
                        help="reuse committed reference vectors")
    args = parser.parse_args()

    blob = pathlib.Path("/tmp/cyborgdb-embed-corpus.bin")
    corpus_blob(blob)

    entries = common.models()
    if args.models:
        wanted = set(args.models)
        entries = [e for e in entries if e["id"] in wanted or e["enum"] in wanted]

    summary = {}
    for entry in entries:
        model_id = entry["id"]
        directory = common.GOLDEN / common.slug(entry)
        print(f"\n=== {model_id}")

        if not args.skip_reference:
            code, out = run([args.reference_python, "run_reference.py", "--model", model_id])
            if code != 0:
                print(f"  reference FAILED: {out.strip().splitlines()[-1][:90]}")
                summary[model_id] = {"error": "reference"}
                continue

        code, out = run([args.python, "run_ort.py", "--model", model_id])
        if code != 0:
            print(f"  ort FAILED: {out.strip().splitlines()[-1][:90]}")
            summary[model_id] = {"error": "ort"}
            continue

        code, out = run([args.embed_corpus, str(blob), str(directory / "embed"), model_id])
        if code != 0:
            print(f"  embed FAILED: {out.strip().splitlines()[-1][:90]}")
            summary[model_id] = {"error": "embed"}
            continue

        _, groups = common.corpus()
        verdicts = {}
        for arm in ("ort", "embed"):
            for kind in ("document", "query"):
                reference_path = directory / f"reference-{kind}.f32"
                candidate_path = directory / f"{arm}-{kind}.f32"
                if not (reference_path.exists() and candidate_path.exists()):
                    continue
                reference, _ = common.read_vectors(reference_path)
                candidate, _ = common.read_vectors(candidate_path)
                verdicts[f"{arm}-{kind}"] = compare.compare(reference, candidate, groups)

        for name, result in verdicts.items():
            print(f"  {name:<16} real={result['verdict_real']:<24} "
                  f"cos_min_real={result['cosine_min_real']:.6f} "
                  f"whole={result['verdict']}")
        summary[model_id] = {k: {"verdict_real": v["verdict_real"],
                                 "verdict": v["verdict"],
                                 "cosine_min_real": v["cosine_min_real"]}
                             for k, v in verdicts.items()}

    (common.ROOT / "tests" / "data" / "parity-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote tests/data/parity-summary.json ({len(summary)} models)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
