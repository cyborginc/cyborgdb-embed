# cyborgdb-embed

Text embedding in C++, numerically equivalent to `sentence-transformers`.

A static library built on ONNX Runtime. Load a supported embedding model, get the same vectors `sentence-transformers` would produce, with no Python at runtime and nothing to install alongside your binary.

## Why

Running embedding models outside Python usually means reimplementing the parts of `sentence-transformers` that are easy to get subtly wrong — pooling mode, truncation length, normalization, query/document prefixes. Getting any of them wrong produces plausible vectors that quietly degrade retrieval rather than failing.

This library pins those choices per model in a registry, resolves the model from its upstream repository at a fixed revision, and verifies every supported model against real `sentence-transformers` output on every release.

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
#include <cyborgdb_embed/embed.hpp>

namespace embed = cyborgdb::embed;

embed::Embedder model;
if (auto s = embed::open(embed::ModelId::BgeBaseEnV15, embed::Options{}, model); !s) {
  return s;  // unknown model, download failed, digest mismatch, load failed
}

std::vector<std::string_view> docs = {"first passage", "second passage"};
std::vector<float> out(docs.size() * model.dimension());

model.embed_documents(docs.data(), docs.size(), out.data(), out.size());
```

Models are named by an enum rather than a string, so a typo is a compile error and the supported set is visible to every binding.

Documents and queries are separate calls because several models expect asymmetric prefixes — e5 wants `query: ` / `passage: `, bge wants an instruction prefix on queries. Applying the wrong one degrades retrieval without erroring, so the API does not let you pass a flag and get it backwards.

Every entry point returns a status. A tokenizer failure, a failed download, a digest that no longer matches, and an undersized output buffer are all reachable, and none of them should surface as a plausible-looking vector.

The session cache is process-wide, so opening one model from many indexes holds a single copy of the weights. An `Embedder` is a handle onto that session: copies share it, and it is safe to use from many threads because each call writes only into its own output buffer.

## Supported models

Defined in `registry.yaml`, which pins the upstream revision, ONNX artifact, dimension, pooling mode, normalization, `max_seq_length`, and prefixes for every model.

<!-- TODO: generate from registry.yaml -->

| Model | Dim | Pooling | Max tokens |
| --- | --- | --- | --- |
| _TBD_ | | | |

Unlisted models can be used via an override, which must supply the same metadata explicitly rather than just a model ID — guessing it is how silent drift happens. Unsupported and unverified.

## Downloading and caching

Models are downloaded on first use and cached on disk. Weights are fp32, matching what `sentence-transformers` loads by default — the equivalence guarantee is against that reference, so the default precision is not configurable.

Each supported model pins an upstream revision and a file digest, both verified before the graph is loaded. A repository that changes underneath you will fail loudly rather than silently returning different vectors.

### Offline and air-gapped use

The cache directory is set with `CYBORGDB_EMBED_CACHE`, defaulting to a per-user location. Nothing outside that directory is written, and nothing is fetched for a model already present in it.

To run without network access, populate the cache on a connected machine and copy it to the target:

```bash
# On a machine with network access
cyborgdb-embed-fetch --model bge-base-en-v1.5 --cache ./embed-cache

# On the air-gapped machine
export CYBORGDB_EMBED_CACHE=/opt/cyborgdb/embed-cache
```

Set `CYBORGDB_EMBED_OFFLINE=1` to fail immediately on a cache miss instead of attempting a download. Use this in production: it turns a missing model into a startup error rather than an unexpected network call on a request path.

`CYBORGDB_EMBED_ENDPOINT` overrides the download host for an internal mirror.

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

Adding a model means editing `registry.yaml` and regenerating, which needs PyYAML (`pip install -r scripts/requirements.txt`). Building the library does not: `src/registry_generated.hpp` is committed, so no Python is required to compile.

See [SPEC.md](SPEC.md) for design rationale, the session cache design, required ONNX Runtime build flags, and known pitfalls.

## License

MIT. See [LICENSE](LICENSE).
