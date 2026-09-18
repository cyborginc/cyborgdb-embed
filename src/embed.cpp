#include "cyborgdb_embed/embed.hpp"

#include <algorithm>

#include "cache.hpp"
#include "ort_api.hpp"
#include "registry_generated.hpp"
#include "session.hpp"

namespace cyborgdb::embed {
namespace {

const detail::RegistryEntry* find(ModelId id) noexcept {
  for (const auto& entry : detail::kRegistry) {
    if (entry.info.id == id) return &entry;
  }
  return nullptr;
}

}  // namespace

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

const ModelInfo* supported_models(std::size_t& count) noexcept {
  count = detail::kModelCount;
  return detail::kModels;
}

const ModelInfo& info(ModelId id) noexcept {
  const detail::RegistryEntry* entry = find(id);
  // Unreachable for any value of the generated enum.
  return entry != nullptr ? entry->info : detail::kModels[0];
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

bool provider_available(Provider provider) noexcept {
  switch (provider) {
    case Provider::CPU:
      return true;
    case Provider::CoreML:
#if defined(__APPLE__) && defined(CYBORGDB_EMBED_WITH_COREML)
      return true;
#else
      return false;
#endif
  }
  return false;
}

std::string runtime_version() { return detail::ort_version(); }

bool precision_available(ModelId, Precision precision) noexcept {
  // The registry pins one graph per model today. Adding a quantised variant is
  // a registry change plus its own parity run, not an API change.
  return precision == Precision::Fp32;
}

// ---------------------------------------------------------------------------
// Embedder
// ---------------------------------------------------------------------------

struct Embedder::Impl {
  std::shared_ptr<detail::Session> session;
};

Embedder::Embedder() noexcept = default;
Embedder::~Embedder() = default;
Embedder::Embedder(const Embedder&) noexcept = default;
Embedder& Embedder::operator=(const Embedder&) noexcept = default;
Embedder::Embedder(Embedder&&) noexcept = default;
Embedder& Embedder::operator=(Embedder&&) noexcept = default;

bool Embedder::valid() const noexcept {
  return impl_ != nullptr && impl_->session != nullptr;
}

std::size_t Embedder::dimension() const noexcept {
  return valid() ? impl_->session->entry->info.dimension : 0;
}

Identity Embedder::identity() const noexcept {
  if (!valid()) return {};
  const detail::Session& session = *impl_->session;
  return {session.entry->info.name, session.entry->info.revision,
          detail::registry_version(), session.provider, session.precision};
}

Status Embedder::embed_documents(const std::string_view* texts, std::size_t n,
                                 float* out, std::size_t out_capacity) const {
  if (!valid()) return {StatusCode::InvalidArgument, "embedder is not open"};
  return impl_->session->encode(texts, n, impl_->session->entry->doc_prefix, out,
                                out_capacity);
}

Status Embedder::embed_queries(const std::string_view* texts, std::size_t n,
                               float* out, std::size_t out_capacity) const {
  if (!valid()) return {StatusCode::InvalidArgument, "embedder is not open"};
  return impl_->session->encode(texts, n, impl_->session->entry->query_prefix,
                                out, out_capacity);
}

Status open(ModelId id, const Options& options, Embedder& out) {
  const detail::RegistryEntry* entry = find(id);
  if (entry == nullptr) {
    return {StatusCode::InvalidArgument, "unknown model"};
  }
  if (options.threads < 1) {
    return {StatusCode::InvalidArgument, "threads must be at least 1"};
  }

  std::shared_ptr<detail::Session> session;
  if (Status status = detail::SessionCache::instance().acquire(
          *entry, options.provider, options.precision, options.threads, session);
      !status) {
    return status;
  }

  auto impl = std::make_shared<Embedder::Impl>();
  impl->session = std::move(session);
  out.impl_ = std::move(impl);
  return {};
}

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

Status configure_cache(const CacheConfig& config) {
  return detail::SessionCache::instance().configure(config);
}

std::vector<LoadedModel> loaded_models() {
  return detail::SessionCache::instance().loaded();
}

CacheStats cache_stats() noexcept {
  return detail::SessionCache::instance().stats();
}

Status unload(ModelId id, Provider provider, Precision precision) {
  return detail::SessionCache::instance().drop(id, provider, precision);
}

}  // namespace cyborgdb::embed
