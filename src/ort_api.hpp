// Sole point of contact with ONNX Runtime. Nothing else in this library may call
// a linked ORT symbol directly: routing through one table is what allows running
// against a dlopen'd runtime later without touching any call site.

#ifndef CYBORGDB_EMBED_ORT_API_HPP
#define CYBORGDB_EMBED_ORT_API_HPP

#include <string>

// Manual init keeps the C++ wrapper's global pointing at the table acquired
// here, so there is still exactly one place this library reaches ONNX Runtime.
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include "cyborgdb_embed/embed.hpp"

namespace cyborgdb::embed::detail {

// Idempotent. Safe to call from any thread.
Status init_ort() noexcept;

// Null until init_ort has returned Ok.
const OrtApi* ort_api() noexcept;

std::string ort_version();

}  // namespace cyborgdb::embed::detail

#endif  // CYBORGDB_EMBED_ORT_API_HPP
