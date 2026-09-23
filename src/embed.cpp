#include "cyborgdb_embed/embed.hpp"

#include <algorithm>
#include <cctype>

#include "session_store.hpp"
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

bool equals_ignoring_case(std::string_view a, std::string_view b) noexcept {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
           return std::tolower(static_cast<unsigned char>(x)) ==
                  std::tolower(static_cast<unsigned char>(y));
         });
}

// OpenAI's embedding models: named often enough in configurations carried over
// from a hosted setup that they deserve a clearer answer than "unknown".
bool is_hosted_model(std::string_view name) noexcept {
  const std::string_view bare = name.substr(name.rfind('/') + 1);
  constexpr std::string_view kThirdGeneration = "text-embedding-3-";
  return equals_ignoring_case(bare, "text-embedding-ada-002") ||
         (bare.size() > kThirdGeneration.size() &&
          equals_ignoring_case(bare.substr(0, kThirdGeneration.size()),
                               kThirdGeneration));
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

Status find_model(std::string_view name, ModelId& out) {
  std::string qualified;
  if (name.find('/') == std::string_view::npos) {
    qualified = "sentence-transformers/" + std::string(name);
  }
  const std::string_view wanted = qualified.empty() ? name : qualified;
  for (const auto& entry : detail::kRegistry) {
    if (equals_ignoring_case(entry.info.name, wanted)) {
      out = entry.info.id;
      return {};
    }
  }

  std::string supported;
  for (const auto& entry : detail::kRegistry) {
    if (!supported.empty()) supported += ", ";
    supported += entry.info.name;
  }
  const std::string reason = is_hosted_model(name)
                                 ? " is a hosted API model, not supported"
                                 : " is not a supported model";
  return {StatusCode::InvalidArgument,
          std::string(name) + reason + "; supported: " + supported};
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

bool provider_available(Provider provider) noexcept {
  return provider == Provider::CPU;
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

  std::shared_ptr<detail::Session> session;
  if (Status status = detail::SessionStore::instance().acquire(
          *entry, options.provider, options.precision, session);
      !status) {
    return status;
  }

  auto impl = std::make_shared<Embedder::Impl>();
  impl->session = std::move(session);
  out.impl_ = std::move(impl);
  return {};
}

// ---------------------------------------------------------------------------
// Runtime and cache
// ---------------------------------------------------------------------------

Status configure_runtime(const RuntimeConfig& config) {
  return detail::configure_runtime(config);
}

Status configure_cache(const CacheConfig& config) {
  return detail::SessionStore::instance().configure(config);
}

std::vector<LoadedModel> loaded_models() {
  return detail::SessionStore::instance().loaded();
}

LoadStats load_stats() noexcept { return detail::SessionStore::instance().stats(); }

}  // namespace cyborgdb::embed
