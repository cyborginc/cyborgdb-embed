#include "session.hpp"

namespace cyborgdb::embed::detail {

std::string_view registry_version() noexcept { return kRegistryVersion; }

Status load_session(const RegistryEntry& entry, Provider provider, int threads,
                    std::shared_ptr<Session>& out) {
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

  return {StatusCode::ModelLoadFailed,
          "model loading is not implemented: needs the tokenizer and the "
          "download path"};
}

}  // namespace cyborgdb::embed::detail
