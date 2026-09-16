# cyborgdb-embed

Text embedding in C++, numerically equivalent to `sentence-transformers`.

A static library built on ONNX Runtime. Load a supported embedding model, get the same vectors `sentence-transformers` would produce, with no Python at runtime and nothing to install alongside your binary.

## Why

Running embedding models outside Python usually means reimplementing the parts of `sentence-transformers` that are easy to get subtly wrong — pooling mode, truncation length, normalization, query/document prefixes. Getting any of them wrong produces plausible vectors that quietly degrade retrieval rather than failing.

This library pins those choices per model, ships ONNX graphs with pooling already baked in, and verifies every supported model against real `sentence-transformers` output on every release.

## Integrate

```cmake
include(FetchContent)
FetchContent_Declare(cyborgdb_embed
  GIT_REPOSITORY https://github.com/cyborg/cyborgdb-embed.git
  GIT_TAG        v0.1.0)
FetchContent_MakeAvailable(cyborgdb_embed)
target_link_libraries(your_target PRIVATE cyborgdb::embed)
```

ONNX Runtime is built from pinned source and linked statically. There is no shared library to ship or install.

## Use

```cpp
#include <cyborgdb/embed.hpp>

auto model = cyborgdb::embed::open("BAAI/bge-base-en-v1.5", {.intra_op_threads = 1});

cyborgdb::embed::embed_documents(*model, docs, out);
cyborgdb::embed::embed_queries(*model, queries, out);
```

Documents and queries are separate calls because several models expect asymmetric prefixes — e5 wants `query: ` / `passage: `, bge wants an instruction prefix on queries. Applying the wrong one degrades retrieval without erroring, so the API does not let you pass a flag and get it backwards.

`open` returns a refcounted handle onto a cached session. `Model` is safe to use from many threads; each caller writes into its own output buffer.

## Supported models

Defined in `registry.yaml`, which pins the upstream revision, ONNX artifact, dimension, pooling mode, normalization, `max_seq_length`, and prefixes for every model.

<!-- TODO: generate from registry.yaml -->

| Model | Dim | Pooling | Max tokens |
| --- | --- | --- | --- |
| _TBD_ | | | |

Unlisted models can be used via an override, which must supply the same metadata explicitly rather than just a model ID — guessing it is how silent drift happens. Unsupported and unverified.

## Correctness

Every supported model is compared against real `sentence-transformers` output on a frozen corpus. Fast tests check committed golden vectors; a nightly job regenerates them from `sentence-transformers` and fails on drift.

| Verdict | Criterion |
| --- | --- |
| `numerically-equivalent` | `cosine_min ≥ 1-1e-6`, `max_abs_diff < 1e-4` |
| `retrieval-equivalent` | `cosine_min ≥ 0.9999`, `recall@10 ≥ 0.99` |

Bit-identical output is not achievable across different BLAS implementations, so equivalence is measured rather than assumed. Verdicts are reported separately for real prose and for degenerate inputs (empty, whitespace-only, emoji, right-to-left), because a handful of pathological strings should not be able to disguise whether ordinary text matches.

<!-- TODO: current parity results -->

| Model | CPU | CoreML |
| --- | --- | --- |
| _TBD_ | | |

## Memory

Sessions are cached and shared: opening the same model many times holds **one** copy of the weights.

The cache is keyed by `(model, registry version, execution provider)`, bounded by resident bytes with an idle timeout, and evicts only models nothing currently holds open. Resident memory exceeds the weights because ONNX Runtime's allocation arena grows to the largest batch and sequence it has seen and does not shrink — the arena is configured for predictable rather than minimal footprint, with maximum batch and sequence length capped.

<!-- TODO: fill from release benchmark -->

| Model | Weights | Resident (1 session) | Resident (16 concurrent) |
| --- | --- | --- | --- |
| _TBD_ | | | |

## Performance

<!-- TODO: fill from release benchmark -->

| Model | Dim | p50 batch 8 | p50 batch 32 |
| --- | --- | --- | --- |
| _TBD_ | | | |

Intra-op threading defaults to 1. Parallelism is expected to come from concurrent callers rather than from splitting a single request across cores, which reaches the same throughput at a fraction of the memory.

## Size

| | Raw | Compressed |
| --- | ---: | ---: |
| Library, linked | ~8.5 MB | ~3.2 MB |
| + CoreML | +1.1 MB | +0.4 MB |

Measured on darwin/arm64 against ONNX Runtime v1.30.0.

## Execution providers

CPU is the default and is statically linked. CoreML is available on darwin/arm64.

**CUDA is not supported.** ONNX Runtime builds its CUDA provider as a loadable module that cannot be statically linked, and GPU acceleration only pays off for embedding at bulk-ingestion batch sizes rather than on a per-request path.

Providers are not numerically neutral — CoreML may run fp16 — so each has its own parity results and golden vectors. If you store vectors, record which provider produced them; switching providers changes the output.

## Development

```bash
make build          # static library
make test           # golden-vector tests, no Python required
make parity         # full comparison vs sentence-transformers (needs torch + model weights)
make export MODEL=  # re-export a pooled ONNX graph
make bench          # latency, concurrency, memory
```

See [SPEC.md](SPEC.md) for design rationale, the session cache design, required ONNX Runtime build flags, and known pitfalls.

## License

<!-- TODO -->
