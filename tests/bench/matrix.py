"""Runs both benchmark arms over models and thread counts, and prints the
tables the README carries.

The reference arm needs torch, which the library itself does not, so it runs
under a separate interpreter: point --reference-python at an environment with
tests/parity/requirements.txt installed. Without it only this library is
measured, which is still useful for tracking a change against itself.
"""
import argparse
import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS = ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "BAAI/bge-large-en-v1.5"]


def cpu_name():
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True)
        return out.stdout.strip() or platform.processor()
    for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def run(argv):
    out = subprocess.run(argv, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"{argv[0]} failed:\n{out.stderr[-2000:]}")
    return json.loads(out.stdout)


def measure(args):
    """One row per arm per configuration. The requested thread count is recorded
    separately because the two arms name their own thread field differently."""
    rows = []
    for model in args.models:
        for threads in args.threads:
            common = ["--model", model, "--corpus", str(args.corpus),
                      "--batch", str(args.batch), "--threads", str(threads),
                      "--seconds", str(args.seconds)]

            result = run([str(args.bench)] + common)
            rows.append({**result, "arm": "cyborgdb-embed", "req_threads": threads})

            if args.reference_python:
                result = run([str(args.reference_python),
                              str(ROOT / "tests/bench/reference.py")]
                             + common + ["--device", "cpu"])
                rows.append({**result, "arm": "sentence-transformers",
                             "req_threads": threads})

            print(f"measured {model} threads={threads}", file=sys.stderr)
    return rows


def tables(rows, args):
    lines = [f"Measured on {cpu_name()}, {platform.system()}, CPU only. "
             f"Batch {args.batch}, {args.seconds:g}s per configuration, "
             f"{args.corpus.name}.", ""]
    width = max(len(m.split("/")[-1]) + 2 for m in args.models)
    for threads in args.threads:
        lines += [f"### {threads} thread" + ("s" if threads != 1 else ""), "",
                  f"| {'Model':<{width}} | {'Library':<21} | Throughput |    p95 | Peak RSS |",
                  f"| {'-' * width} | {'-' * 21} | ---------: | -----: | -------: |"]
        # Value widths match the headers above so the table stays aligned.
        for model in args.models:
            shown = f"`{model.split('/')[1]}`"
            for arm in ("cyborgdb-embed", "sentence-transformers"):
                r = next((x for x in rows if x["model"] == model
                          and x["req_threads"] == threads and x["arm"] == arm), None)
                if not r:
                    continue
                lines.append(
                    f"| {shown:<{width}} | {arm:<21} | "
                    f"{r['throughput_per_s']:>8.0f}/s | "
                    f"{r['p95_ms']:>4.0f}ms | "
                    f"{r['peak_rss_bytes'] / 1e6:>5.0f} MB |")
                shown = ""  # one model label per pair, so the pairing reads
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", type=pathlib.Path, default=ROOT / "build/bench")
    p.add_argument("--reference-python", type=pathlib.Path,
                   help="interpreter with torch and sentence-transformers")
    p.add_argument("--corpus", type=pathlib.Path, default=ROOT / "tests/data/corpus.bin")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--threads", nargs="+", type=int, default=[1, 12])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--json", type=pathlib.Path, help="also write the raw measurements here")
    p.add_argument("--from-json", type=pathlib.Path,
                   help="re-render tables from a previous --json run, measuring nothing")
    args = p.parse_args()

    if args.from_json:
        rows = json.loads(args.from_json.read_text())
        print(tables(rows, args))
        return 0

    if not args.bench.exists():
        sys.exit(f"no benchmark binary at {args.bench}; build it first")

    rows = measure(args)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"raw measurements in {args.json}", file=sys.stderr)
    print(tables(rows, args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
