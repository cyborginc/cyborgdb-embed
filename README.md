# cyborgdb-embed

Text embedding in C++, numerically equivalent to `sentence-transformers`.

A static library built on ONNX Runtime. Load a supported embedding model from HF, get the same vectors `sentence-transformers` would produce, with no Python at runtime and nothing to install alongside your binary.

## Why

Running embedding models outside Python usually means reimplementing the parts of `sentence-transformers` that are easy to get subtly wrong (pooling mode, truncation length, normalization, query/document prefixes...). Getting any of them wrong produces plausible vectors that quietly degrade retrieval rather than failing.

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

Everything links statically. ONNX Runtime and the HuggingFace tokenizer are committed as prebuilt archives under `onnxruntime/prebuilt/` and `tokenizer/prebuilt/`. There is no shared library to ship or install, and nothing is downloaded at configure time.

Linux builds need `libcurl` and OpenSSL development headers (`libcurl4-openssl-dev`, `libssl-dev` on Debian and Ubuntu).

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

Documents and queries are separate calls because several models expect asymmetric prefixes (e.g., `query: ` / `passage: `).

The session cache is process-wide, so opening one model from many threads holds a single copy of the weights. An `Embedder` is a handle onto that session: copies share it, and it is safe to use from many threads because each call writes only into its own output buffer.

## Supported models

Defined in `registry.yaml`, which pins the upstream revision, the exact ONNX file and its digest, dimension, pooling mode, normalization, `max_seq_length`, and prefixes for every model.

| Model                                   | Dim | Max tokens | Pooling | Prefixes | Parity |
| --------------------------------------- | ---: | ---: | --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | mean | — | numerically-equivalent |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | 128 | mean | — | numerically-equivalent |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 384 | mean | — | numerically-equivalent |
| `BAAI/bge-small-en-v1.5`                | 384 | 512 | cls | yes | numerically-equivalent |
| `BAAI/bge-base-en-v1.5`                 | 768 | 512 | cls | yes | numerically-equivalent |
| `BAAI/bge-large-en-v1.5`                | 1024 | 512 | cls | yes | numerically-equivalent |
| `intfloat/e5-small-v2`                  | 384 | 512 | mean | yes | numerically-equivalent |
| `intfloat/e5-base-v2`                   | 768 | 512 | mean | yes | numerically-equivalent |
| `intfloat/e5-large-v2`                  | 1024 | 512 | mean | yes | numerically-equivalent |
| `intfloat/multilingual-e5-small`        | 384 | 512 | mean | yes | numerically-equivalent |

Parity is measured against `sentence-transformers`

## Downloading and caching

Models are downloaded on first use and cached on disk. Weights are fp32, matching what `sentence-transformers` loads by default.

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

## Memory

Sessions are shared: opening the same model from many indexes holds one copy of the weights, and the last handle released frees them. There is no retention beyond use, so resident memory is a function of what is open rather than of an eviction policy.

Most memory is not weights. It is the activations of in-flight work, which scale with the number of sentences being embedded at once times the padded sequence length — so a batch mixing one long document with short ones costs as much per row as the longest. Batches are therefore formed against a token budget over length-sorted input rather than a fixed row count.

Measured on darwin/arm64, batch 8, embedding a corpus with sequences up to 512 tokens:

| Model | Dim | Peak RSS, 1 caller | Peak RSS, 8 callers |
| --- | ---: | ---: | ---: |
| `bge-small-en-v1.5` | 384 | 835 MB | 3281 MB |
| `bge-base-en-v1.5` | 768 | 1146 MB | 3366 MB |
| `bge-large-en-v1.5` | 1024 | 2436 MB | 3106 MB |

## Performance

Same machine and batch size. Throughput is sentences per second; latency is per call.

| Model | 1 caller | 8 callers | p50 (1 caller) | p50 (8 callers) |
| --- | ---: | ---: | ---: | ---: |
| `bge-small-en-v1.5` | 394/s | 2026/s | 7.9 ms | 11.4 ms |
| `bge-base-en-v1.5` | 212/s | 555/s | 19.8 ms | 38.0 ms |
| `bge-large-en-v1.5` | 32/s | 198/s | 65.6 ms | 131.0 ms |


## Size

A statically linked executable using this library, stripped and dead-stripped, is 16.9 MB on darwin/arm64.

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
