"""Throughput and memory of sentence-transformers, for comparison with bench.cpp.

Mirrors the C++ benchmark: closed loop, fixed batch, count sentences completed
in a fixed wall window, over the same corpus file.
"""
import argparse
import json
import pathlib
import resource
import sys
import time


def peak_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", default="tests/data/corpus.bin")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--threads", type=int, default=0)
    # Left to itself sentence-transformers selects the GPU through MPS on Apple
    # silicon, which does not compare against a CPU-only library.
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    from sentence_transformers import SentenceTransformer

    # The NUL-separated blob the C++ benchmark reads, so both arms see one input.
    blob = pathlib.Path(args.corpus).read_bytes()
    texts = [t.decode() for t in blob.split(b"\0") if t]

    model = SentenceTransformer(args.model, device=args.device)
    model.eval()
    rss_after_load = peak_rss_bytes()

    def encode(batch):
        model.encode(batch, batch_size=args.batch, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)

    # The first call pays lazy initialisation that steady state should not.
    encode(texts[:args.batch])

    latencies = []
    sentences = index = 0
    start = time.perf_counter()
    while time.perf_counter() - start < args.seconds:
        batch = texts[index % len(texts):][:args.batch]
        if len(batch) < args.batch:
            batch = (batch + texts)[:args.batch]
        call_start = time.perf_counter()
        encode(batch)
        latencies.append((time.perf_counter() - call_start) * 1000)
        sentences += len(batch)
        index += args.batch
    wall = time.perf_counter() - start

    latencies.sort()
    print(json.dumps({
        "model": args.model,
        "device": str(model.device),
        "threads": torch.get_num_threads(),
        "batch": args.batch,
        "wall_seconds": round(wall, 3),
        "sentences": sentences,
        "throughput_per_s": round(sentences / wall, 1),
        "calls": len(latencies),
        "p50_ms": round(latencies[len(latencies) // 2], 2),
        "p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))], 2),
        "p99_ms": round(latencies[int(0.99 * (len(latencies) - 1))], 2),
        "mean_ms": round(sum(latencies) / len(latencies), 2),
        # Whether the timed region actually accounts for the wall clock; a low
        # fraction would mean the measurement, not the model, is being reported.
        "measured_fraction": round(sum(latencies) / 1000 / wall, 3),
        "rss_after_load_bytes": rss_after_load,
        "peak_rss_bytes": peak_rss_bytes(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
