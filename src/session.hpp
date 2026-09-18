#ifndef CYBORGDB_EMBED_SESSION_HPP
#define CYBORGDB_EMBED_SESSION_HPP

#include <memory>
#include <string>

#include "cyborgdb_embed/embed.hpp"
#include "registry_generated.hpp"

namespace cyborgdb::embed::detail {

// A loaded model: the inference session, its tokenizer, and the metadata needed
// to tag vectors it produces.
class Session {
 public:
  virtual ~Session() = default;

  virtual Status encode(const std::string_view* texts, std::size_t n,
                        std::string_view prefix, float* out,
                        std::size_t out_capacity) const = 0;

  const RegistryEntry* entry = nullptr;
  Provider provider = Provider::CPU;

  // Weights plus the allocation arena, which grows to the largest batch and
  // sequence seen and does not shrink.
  std::size_t resident_bytes = 0;
};

// Resolves the model, verifies it, and loads it. Blocking: downloads on a cold
// cache. The cache passes its own configuration rather than being consulted from
// here, which would be circular.
Status load_session(const RegistryEntry&, Provider, int threads,
                    const CacheConfig&, std::shared_ptr<Session>& out);

std::string_view registry_version() noexcept;

// Defined in session_ort.cpp, which is the only place ONNX Runtime types appear.
Status make_ort_session(const RegistryEntry&, Provider, int threads,
                        const std::string& graph, const std::string& tokenizer_json,
                        std::shared_ptr<Session>& out);

}  // namespace cyborgdb::embed::detail

#endif  // CYBORGDB_EMBED_SESSION_HPP
