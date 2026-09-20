# cyborgdb-embed

Text embedding in C++, numerically equivalent to `sentence-transformers`.

`cyborgdb-embed` is a static C++ library built on ONNX Runtime, with no Python runtime or shared-library dependencies.

“Numerically equivalent” means cosine similarity ≥ `1 - 1e-6` and max absolute difference < `1e-4`. Numerical equivalence does not imply bit-identical output across platforms or runtimes.

## Quick start

```cmake
include(FetchContent)

FetchContent_Declare(
  cyborgdb_embed
  GIT_REPOSITORY https://github.com/cyborg/cyborgdb-embed.git
  GIT_TAG v0.1.0
)

FetchContent_MakeAvailable(cyborgdb_embed)

target_link_libraries(your_target PRIVATE cyborgdb::embed)
```

Dependencies are linked statically. Linux builds require `libcurl` and OpenSSL development headers (`libcurl4-openssl-dev` and `libssl-dev` on Debian/Ubuntu).

```cpp
#include <cyborgdb_embed/embed.hpp>

namespace embed = cyborgdb::embed;

embed::Embedder model;

if (auto s = embed::open(
        embed::ModelId::BgeBaseEnV15,
        embed::Options{},
        model);
    !s) {
  return s;
}

std::vector<std::string_view> docs = {
    "first passage",
    "second passage",
};

std::vector<float> out(
    docs.size() * model.dimension());

model.embed_documents(
    docs.data(),
    docs.size(),
    out.data(),
    out.size());
```

Documents and queries use separate calls because some models require asymmetric prefixes such as `query:` and `passage:`.

`Embedder` is cheap to copy and thread-safe. Copies of the same model share one in-process session and one copy of the weights.

## Supported models

All supported models are verified against `sentence-transformers`.

| Model                                     |  Dim | Max tokens | Pooling | Prefixes |
| ----------------------------------------- | ---: | ---------: | ------- | -------- |
| `sentence-transformers/all-MiniLM-L6-v2`  |  384 |        256 | mean    | —        |
| `sentence-transformers/all-MiniLM-L12-v2` |  384 |        128 | mean    | —        |
| `sentence-transformers/all-mpnet-base-v2` |  768 |        384 | mean    | —        |
| `BAAI/bge-small-en-v1.5`                  |  384 |        512 | cls     | yes      |
| `BAAI/bge-base-en-v1.5`                   |  768 |        512 | cls     | yes      |
| `BAAI/bge-large-en-v1.5`                  | 1024 |        512 | cls     | yes      |
| `intfloat/e5-small-v2`                    |  384 |        512 | mean    | yes      |
| `intfloat/e5-base-v2`                     |  768 |        512 | mean    | yes      |
| `intfloat/e5-large-v2`                    | 1024 |        512 | mean    | yes      |
| `intfloat/multilingual-e5-small`          |  384 |        512 | mean    | yes      |

Model configuration and pinned upstream revisions are defined in `registry.yaml`.

## Model downloads and offline use

Models are downloaded on first use and cached on disk.

For air-gapped deployment:

```bash
# On a connected machine
cyborgdb-embed-fetch \
  --model bge-base-en-v1.5 \
  --cache ./embed-cache

# On the target machine
export CYBORGDB_EMBED_CACHE=/opt/cyborgdb/embed-cache
export CYBORGDB_EMBED_OFFLINE=1
```

Set `CYBORGDB_EMBED_OFFLINE=1` to fail on a cache miss without attempting network access. `CYBORGDB_EMBED_ENDPOINT` overrides the download host.

## Platform support

CPU execution is supported today. CoreML and CUDA are planned for future releases.

## Development

```bash
make build          # static library
make test           # golden-vector tests; no Python required
make parity         # compare against sentence-transformers
make export MODEL=  # re-export a pooled ONNX graph
make bench          # latency, concurrency, memory
```

Python is only required for model regeneration and parity testing; building and using the library does not require it.

## License

MIT. See [LICENSE](LICENSE).
