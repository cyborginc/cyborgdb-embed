#include "ort_api.hpp"

#include <mutex>

namespace cyborgdb::embed::detail {
namespace {

std::once_flag g_once;
const OrtApi* g_api = nullptr;

}  // namespace

Status init_ort() noexcept {
  std::call_once(g_once, [] {
    const OrtApiBase* base = OrtGetApiBase();
    if (base != nullptr) {
      g_api = base->GetApi(ORT_API_VERSION);
    }
  });
  if (g_api == nullptr) {
    return {StatusCode::ModelLoadFailed,
            "ONNX Runtime does not provide API version " +
                std::to_string(ORT_API_VERSION)};
  }
  return {};
}

const OrtApi* ort_api() noexcept { return g_api; }

std::string ort_version() {
  const OrtApiBase* base = OrtGetApiBase();
  return base != nullptr ? base->GetVersionString() : std::string{};
}

}  // namespace cyborgdb::embed::detail
