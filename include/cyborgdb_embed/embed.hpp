// Text embedding numerically equivalent to sentence-transformers.

#ifndef CYBORGDB_EMBED_HPP
#define CYBORGDB_EMBED_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cyborgdb::embed {

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

// Generated from registry.yaml. A model reaches this enum only after a parity
// run has produced a verdict for it.
enum class ModelId {
  AllMiniLmL6V2,
  AllMiniLmL12V2,
  AllMpnetBaseV2,
  BgeSmallEnV15,
  BgeBaseEnV15,
  BgeLargeEnV15,
  E5SmallV2,
  E5BaseV2,
  E5LargeV2,
  MultilingualE5Small,
};

struct ModelInfo {
  ModelId id;
  std::string_view name;      // upstream repository, e.g. "BAAI/bge-base-en-v1.5"
  std::string_view revision;
  std::size_t dimension;

  // Longest input the model sees; anything beyond it is truncated.
  std::size_t max_seq_length;

  // Normalized vectors make cosine distance and inner product equivalent,
  // which decides how an index should be configured.
  bool normalize;
};

// Backed by a static table; no allocation, and usable before anything is loaded.
const ModelInfo* supported_models(std::size_t& count) noexcept;
const ModelInfo& info(ModelId) noexcept;

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

enum class StatusCode {
  Ok = 0,
  InvalidArgument,
  DownloadFailed,
  DigestMismatch,     // upstream file no longer matches the pinned digest
  NotCached,          // offline mode, and the model is absent from the cache
  ModelLoadFailed,
  ProviderUnavailable,
  TokenizerFailed,
  InferenceFailed,
  OutputTooSmall,
  PrecisionUnavailable,
};

struct Status {
  StatusCode code = StatusCode::Ok;
  std::string message;

  bool ok() const noexcept { return code == StatusCode::Ok; }
  explicit operator bool() const noexcept { return ok(); }
};

// Resolves a name the way sentence-transformers does: the upstream repository
// ("sentence-transformers/all-MiniLM-L6-v2"), or a bare name, which it qualifies
// with "sentence-transformers/". Case-insensitive. An unsupported name returns
// InvalidArgument listing the supported models.
Status find_model(std::string_view name, ModelId& out);

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

// CoreML and CUDA mayb be added later
enum class Provider {
  CPU,
};

bool provider_available(Provider) noexcept;

// Weight precision. A quantised graph is a different graph: it produces
// different vectors, so precision carries the same obligations a provider does
// — its own parity verdict, its own golden vectors, and a place in the identity
// an index records.
//
// Only Fp32 is supported today; the others are rejected rather than approximated.
enum class Precision {
  Fp32,
  Fp16,
  Int8,
};

bool precision_available(ModelId, Precision) noexcept;

// ---------------------------------------------------------------------------
// Embedder
// ---------------------------------------------------------------------------

struct Options {
  Provider provider = Provider::CPU;
  Precision precision = Precision::Fp32;
};

// What an index must record to reject vectors it cannot compare against.
// Dimension alone is insufficient: many models share one.
struct Identity {
  std::string_view model;
  std::string_view revision;
  std::string_view registry_version;
  Provider provider;
  Precision precision;
};

// A handle onto a shared session. Copies share it, and the session outlives
// cache eviction until the last handle is gone.
class Embedder {
 public:
  Embedder() noexcept;
  ~Embedder();

  Embedder(const Embedder&) noexcept;
  Embedder& operator=(const Embedder&) noexcept;
  Embedder(Embedder&&) noexcept;
  Embedder& operator=(Embedder&&) noexcept;

  bool valid() const noexcept;
  std::size_t dimension() const noexcept;
  Identity identity() const noexcept;

  // Write n * dimension() floats into out, one row per input.
  //
  // Documents and queries are separate calls because models apply asymmetric
  // prefixes, and a direction flag is easy to pass backwards.
  //
  // Safe to call concurrently. Each call writes only to its own out.
  Status embed_documents(const std::string_view* texts, std::size_t n,
                         float* out, std::size_t out_capacity) const;

  Status embed_queries(const std::string_view* texts, std::size_t n,
                       float* out, std::size_t out_capacity) const;

 private:
  friend Status open(ModelId, const Options&, Embedder&);

  struct Impl;
  std::shared_ptr<Impl> impl_;
};

// Downloads and verifies the model if it is not already cached, then loads it.
// Returns a handle onto the existing session when one matches.
Status open(ModelId, const Options&, Embedder& out);

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

// Process-wide, like the sessions: every model runs on one thread pool, so
// opening another model adds its weights but no threads.
struct RuntimeConfig {
  // Threads within a single call. Parallelism is expected to come from
  // concurrent callers, which reaches the same throughput on less memory.
  int threads = 1;
};

// Call before the first open. The first open creates the pool and it lives for
// the rest of the process, so from then on only the value already in force is
// accepted; any other returns InvalidArgument.
Status configure_runtime(const RuntimeConfig&);

// ---------------------------------------------------------------------------
// Download cache
// ---------------------------------------------------------------------------

// Where model files live on disk, and whether they may be fetched.
struct CacheConfig {
  std::string cache_dir;      // empty = per-user default
  std::string endpoint;       // empty = upstream default

  // Turn a cache miss into an error instead of a download. Production
  // deployments want failures at startup, not on a request path.
  bool offline = false;
};

// Call before the first open. Defaults come from the environment.
Status configure_cache(const CacheConfig&);

// ---------------------------------------------------------------------------
// Loaded models
// ---------------------------------------------------------------------------

// A session lives exactly as long as some Embedder holds it. Opening the same
// model twice shares one session; releasing the last handle frees the weights.
// There is no retention beyond use, so memory is a function of what is open
// rather than of an eviction policy.
struct LoadedModel {
  ModelId model;
  Provider provider;
  Precision precision;
  // Distinct successful opens still outstanding. Copying an Embedder does not
  // add one: copies share a handle.
  int handles;
};

struct LoadStats {
  std::size_t shared;         // opens that reused a live session
  std::size_t loaded;         // opens that had to build one
};

std::vector<LoadedModel> loaded_models();
LoadStats load_stats() noexcept;

// The ONNX Runtime build behind this library. Vectors are only comparable
// across hosts running the same one, so a health endpoint should report it.
std::string runtime_version();

}  // namespace cyborgdb::embed

#endif  // CYBORGDB_EMBED_HPP
