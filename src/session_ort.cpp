#include <algorithm>
#include <cmath>
#include <vector>

#include "ort_api.hpp"
#include "resolve.hpp"
#include "session.hpp"
#include "tokenizer.hpp"

namespace cyborgdb::embed::detail {
namespace {

// One environment per process: ONNX Runtime expects it, and sessions share it.
Ort::Env& environment() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cyborgdb-embed");
  return env;
}

constexpr std::size_t kBatch = 32;

void pool(const float* hidden, const std::int64_t* mask, std::size_t rows,
          std::size_t width, std::size_t dim, Pooling mode, float* out) {
  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = hidden + r * width * dim;
    float* dest = out + r * dim;

    if (mode == Pooling::ClsToken) {
      std::copy_n(row, dim, dest);
      continue;
    }
    if (mode == Pooling::MaxTokens) {
      std::fill_n(dest, dim, -std::numeric_limits<float>::infinity());
      for (std::size_t t = 0; t < width; ++t) {
        if (!mask[r * width + t]) continue;
        for (std::size_t d = 0; d < dim; ++d) {
          dest[d] = std::max(dest[d], row[t * dim + d]);
        }
      }
      continue;
    }

    std::fill_n(dest, dim, 0.0f);
    float count = 0.0f;
    for (std::size_t t = 0; t < width; ++t) {
      if (!mask[r * width + t]) continue;
      count += 1.0f;
      for (std::size_t d = 0; d < dim; ++d) {
        dest[d] += row[t * dim + d];
      }
    }
    // An all-padding row would divide by zero; empty input is a real case.
    const float divisor = count > 0.0f ? count : 1.0f;
    for (std::size_t d = 0; d < dim; ++d) dest[d] /= divisor;
  }
}

void l2_normalize(float* vectors, std::size_t rows, std::size_t dim) {
  for (std::size_t r = 0; r < rows; ++r) {
    float* row = vectors + r * dim;
    float sum = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) sum += row[d] * row[d];
    const float norm = std::sqrt(sum);
    if (norm <= 1e-12f) continue;
    for (std::size_t d = 0; d < dim; ++d) row[d] /= norm;
  }
}

class OrtSession final : public Session {
 public:
  OrtSession(const RegistryEntry& registry_entry, Provider used_provider,
             Precision used_precision, Ort::Session&& session,
             std::unique_ptr<Tokenizer> tokenizer, bool needs_token_type_ids)
      : session_(std::move(session)),
        tokenizer_(std::move(tokenizer)),
        needs_token_type_ids_(needs_token_type_ids) {
    entry = &registry_entry;
    provider = used_provider;
    precision = used_precision;
  }

  Status encode(const std::string_view* texts, std::size_t n,
                std::string_view prefix, float* out,
                std::size_t out_capacity) const override;

 private:
  mutable Ort::Session session_;
  std::unique_ptr<Tokenizer> tokenizer_;
  bool needs_token_type_ids_;
};

Status OrtSession::encode(const std::string_view* texts, std::size_t n,
                          std::string_view prefix, float* out,
                          std::size_t out_capacity) const {
  const std::size_t dim = entry->info.dimension;
  if (n != 0 && (texts == nullptr || out == nullptr)) {
    return {StatusCode::InvalidArgument, "texts and out must be non-null"};
  }
  if (out_capacity < n * dim) {
    return {StatusCode::OutputTooSmall, "need " + std::to_string(n * dim) +
                                            " floats, got " +
                                            std::to_string(out_capacity)};
  }

  const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<std::uint32_t> ids;

  for (std::size_t start = 0; start < n; start += kBatch) {
    const std::size_t rows = std::min(kBatch, n - start);

    std::vector<std::vector<std::uint32_t>> encoded(rows);
    std::size_t width = 1;
    for (std::size_t r = 0; r < rows; ++r) {
      if (Status status = tokenizer_->encode(texts[start + r], prefix, ids);
          !status) {
        return status;
      }
      encoded[r] = ids;
      width = std::max(width, ids.size());
    }

    // Pad to the batch, never to a fixed length: a fixed strategy costs an order
    // of magnitude on short inputs and changes nothing in the output.
    std::vector<std::int64_t> input_ids(rows * width, 0);
    std::vector<std::int64_t> mask(rows * width, 0);
    for (std::size_t r = 0; r < rows; ++r) {
      for (std::size_t t = 0; t < encoded[r].size(); ++t) {
        input_ids[r * width + t] = encoded[r][t];
        mask[r * width + t] = 1;
      }
    }
    std::vector<std::int64_t> token_types(rows * width, 0);

    const std::array<std::int64_t, 2> shape{static_cast<std::int64_t>(rows),
                                            static_cast<std::int64_t>(width)};
    std::vector<const char*> names{"input_ids", "attention_mask"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(
        memory, input_ids.data(), input_ids.size(), shape.data(), shape.size()));
    inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(
        memory, mask.data(), mask.size(), shape.data(), shape.size()));
    if (needs_token_type_ids_) {
      names.push_back("token_type_ids");
      inputs.push_back(Ort::Value::CreateTensor<std::int64_t>(
          memory, token_types.data(), token_types.size(), shape.data(), shape.size()));
    }

    const char* outputs[] = {"last_hidden_state"};
    std::vector<Ort::Value> result;
    try {
      result = session_.Run(Ort::RunOptions{nullptr}, names.data(), inputs.data(),
                            inputs.size(), outputs, 1);
    } catch (const Ort::Exception& error) {
      return {StatusCode::InferenceFailed, error.what()};
    }

    const float* hidden = result[0].GetTensorData<float>();
    float* destination = out + start * dim;
    pool(hidden, mask.data(), rows, width, dim, entry->pooling, destination);
    if (entry->info.normalize) {
      l2_normalize(destination, rows, dim);
    }
  }
  return {};
}

}  // namespace

Status make_ort_session(const RegistryEntry& entry, Provider provider,
                        Precision precision, int threads, const std::string& graph,
                        const std::string& tokenizer_json,
                        std::shared_ptr<Session>& out) {
  std::unique_ptr<Tokenizer> tokenizer;
  if (Status status = Tokenizer::load(tokenizer_json, entry.info.max_seq_length,
                                     tokenizer);
      !status) {
    return status;
  }

  try {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(threads);
    options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // The arena otherwise grows to the largest batch and sequence ever seen and
    // never shrinks, so one long document permanently raises the floor.
    options.AddConfigEntry("session.use_env_allocators", "0");

    Ort::Session session(environment(), graph.c_str(), options);

    bool needs_token_type_ids = false;
    Ort::AllocatorWithDefaultOptions allocator;
    for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
      if (std::string(session.GetInputNameAllocated(i, allocator).get()) ==
          "token_type_ids") {
        needs_token_type_ids = true;
      }
    }

    out = std::make_shared<OrtSession>(entry, provider, precision, std::move(session),
                                       std::move(tokenizer), needs_token_type_ids);
  } catch (const Ort::Exception& error) {
    return {StatusCode::ModelLoadFailed, error.what()};
  }
  return {};
}

}  // namespace cyborgdb::embed::detail
