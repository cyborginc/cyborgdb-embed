"""Score an arm against the reference and print the verdict."""

import argparse
import json
import sys

import common
import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", default="ort")
    parser.add_argument("--reference", default="reference")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    entry = common.model(args.model)
    _, groups = common.corpus()
    directory = common.GOLDEN / common.slug(entry)

    results = {}
    for kind in ("document", "query"):
        reference_path = directory / f"{args.reference}-{kind}.f32"
        candidate_path = directory / f"{args.arm}-{kind}.f32"
        if not (reference_path.exists() and candidate_path.exists()):
            continue
        reference, _ = common.read_vectors(reference_path)
        candidate, _ = common.read_vectors(candidate_path)
        results[kind] = compare.compare(reference, candidate, groups)

    if not results:
        print(f"no vectors for {entry['id']}: run both arms first", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps({"model": entry["id"], "arm": args.arm, "results": results}, indent=2))
        return 0

    print(f"{entry['id']}  ({args.arm} vs {args.reference})")
    for kind, r in results.items():
        print(f"  {kind:<9} whole-corpus: {r['verdict']:<24} cos_min={r['cosine_min']:.6f}")
        print(f"  {'':<9} real prose:   {r['verdict_real']:<24} cos_min={r['cosine_min_real']:.6f} "
              f"recall@10={r['recall_at_10_real']:.4f}")
        print(f"  {'':<9} edge cases:   cos_min={r.get('cosine_min_edge', float('nan')):.6f} "
              f"max_abs={r.get('max_abs_diff_edge', 0):.2e}")
        worst = r["worst_rows"][0]
        print(f"  {'':<9} worst row {worst['row']} at cosine {worst['cosine']:.6f}")
    # Real prose carries the verdict, so that is what gates.
    return 0 if all(r["verdict_real"] in ("identical", "numerically-equivalent",
                                          "retrieval-equivalent") for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
