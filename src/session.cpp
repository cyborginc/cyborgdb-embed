#include "session.hpp"

#include "resolve.hpp"
#include "ort_api.hpp"
#include "tokenizer.hpp"

namespace cyborgdb::embed::detail {

std::string_view registry_version() noexcept { return kRegistryVersion; }

Status load_session(const RegistryEntry& entry, Provider provider,
                    Precision precision, int threads, const CacheConfig& config,
                    std::shared_ptr<Session>& out) {
  (void)threads;
  (void)out;

  // Rejected rather than approximated: a quantised graph is a different graph,
  // and silently serving fp32 for it would be the drift this library exists to
  // prevent.
  if (!precision_available(entry.info.id, precision)) {
    return {StatusCode::PrecisionUnavailable,
            "only fp32 is supported for this model"};
  }
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

  // Must precede any ONNX Runtime type: constructing one before the API table
  // is acquired dereferences a null pointer.
  if (Status status = init_ort(); !status) {
    return status;
  }
  return make_ort_session(entry, provider, precision, threads, graph,
                          tokenizer_json, out);
}

}  // namespace cyborgdb::embed::detail
