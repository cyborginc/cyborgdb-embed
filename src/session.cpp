#include "session.hpp"

#include "resolve.hpp"
#include "tokenizer.hpp"

namespace cyborgdb::embed::detail {

std::string_view registry_version() noexcept { return kRegistryVersion; }

Status load_session(const RegistryEntry& entry, Provider provider, int threads,
                    const CacheConfig& config, std::shared_ptr<Session>& out) {
  (void)threads;
  (void)out;

  if (!provider_available(provider)) {
    return {StatusCode::ProviderUnavailable,
            "provider is not available in this build"};
  }

  // An unpinned entry cannot be verified, and an unverified graph can change
  // underneath a built index without anything failing.
  if (entry.info.revision.empty() || entry.onnx_sha256.empty()) {
    return {StatusCode::ModelLoadFailed,
            std::string(entry.info.name) +
                " has no pinned revision or digest; run the registry pin job"};
  }

  std::string graph;
  if (Status status = ensure_graph(entry, config, graph); !status) {
    return status;
  }

  std::string tokenizer_json;
  if (Status status = ensure_tokenizer(entry, config, tokenizer_json); !status) {
    return status;
  }

  std::unique_ptr<Tokenizer> tokenizer;
  if (Status status = Tokenizer::load(tokenizer_json, tokenizer); !status) {
    return status;
  }

  return {StatusCode::ModelLoadFailed,
          "graph and tokenizer are cached and loadable; inference is next"};
}

}  // namespace cyborgdb::embed::detail
