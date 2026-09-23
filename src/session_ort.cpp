#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

#include "ort_api.hpp"
#include "pooling.hpp"
#include "resolve.hpp"
#include "session.hpp"
#include "tokenizer.hpp"

namespace cyborgdb::embed::detail {
namespace {

// One environment per process: ONNX Runtime expects it, and every session runs
// on its thread pool. The first session creates it, which fixes the pool size.
struct Runtime {
  std::mutex mutex;
  bool started = false;
  int threads = 1;
};

Runtime& runtime() {
  static Runtime instance;
  return instance;
}

Ort::Env& environment() {
  Runtime& state = runtime();
  std::lock_guard<std::mutex> guard(state.mutex);
  // A local static, not a member of Runtime: constructing it creates ONNX
  // Runtime's own statics, so it is destroyed before them. Held anywhere
  // constructed earlier, its destructor runs after them and aborts at exit.
  static Ort::Env env = [&] {
    Ort::ThreadingOptions threading;
    threading.SetGlobalIntraOpNumThreads(state.threads);
    threading.SetGlobalInterOpNumThreads(1);
    // Idle pool threads otherwise busy-wait after each call, holding cores in
    // a host that embeds only now and then.
    threading.SetGlobalSpinControl(0);
    return Ort::Env(threading, ORT_LOGGING_LEVEL_WARNING, "cyborgdb-embed");
  }();
  state.started = true;
  return env;
}

// Activation memory scales with rows x padded width, not rows, so batches are
// formed against a token budget rather than a fixed count. Without this a batch
// of long texts costs an order of magnitude more than the same count of short
// ones, and peak resident memory is set by whichever batch happened to be widest.
constexpr std::size_t kMaxBatchTokens = 8192;
constexpr std::size_t kMaxBatchRows = 64;

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

  // Tokenise everything first, then group similar lengths together. A batch is
  // padded to its longest member, so one long text in a batch of short ones
  // inflates every row and costs memory proportional to the longest. Grouping by
  // length is output-equivalent: padded positions are masked out of attention
  // and excluded from pooling.
  std::vector<std::vector<std::uint32_t>> tokens(n);
  std::vector<std::size_t> order(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (Status status = tokenizer_->encode(texts[i], prefix, tokens[i]); !status) {
      return status;
    }
    order[i] = i;
  }
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return tokens[a].size() < tokens[b].size();
  });

  for (std::size_t start = 0; start < n;) {
    // Sorted order means width grows slowly, so the budget admits many short
    // rows and few long ones.
    std::size_t rows = 0;
    std::size_t width = 1;
    while (start + rows < n && rows < kMaxBatchRows) {
      const std::size_t candidate =
          std::max(width, tokens[order[start + rows]].size());
      if (rows > 0 && (rows + 1) * candidate > kMaxBatchTokens) break;
      width = candidate;
      ++rows;
    }

    std::vector<std::vector<std::uint32_t>> encoded(rows);
    for (std::size_t r = 0; r < rows; ++r) {
      encoded[r] = tokens[order[start + r]];
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
    std::vector<float> pooled(rows * dim);
    pool(hidden, mask.data(), rows, width, dim, entry->pooling, pooled.data());
    if (entry->info.normalize) {
      l2_normalize(pooled.data(), rows, dim);
    }
    for (std::size_t r = 0; r < rows; ++r) {
      std::copy_n(pooled.data() + r * dim, dim, out + order[start + r] * dim);
    }
    start += rows;
  }
  return {};
}

}  // namespace

Status configure_runtime(const RuntimeConfig& config) {
  if (config.threads < 1) {
    return {StatusCode::InvalidArgument, "threads must be at least 1"};
  }
  Runtime& state = runtime();
  std::lock_guard<std::mutex> guard(state.mutex);
  if (state.started && config.threads != state.threads) {
    return {StatusCode::InvalidArgument,
            "the thread pool already started with " + std::to_string(state.threads) +
                " threads; configure the runtime before the first open"};
  }
  state.threads = config.threads;
  return {};
}

Status make_ort_session(const RegistryEntry& entry, Provider provider,
                        Precision precision, const std::string& graph,
                        const std::string& tokenizer_json,
                        std::shared_ptr<Session>& out) {
  std::unique_ptr<Tokenizer> tokenizer;
  if (Status status = Tokenizer::load(tokenizer_json, entry.info.max_seq_length,
                                     tokenizer);
      !status) {
    return status;
  }

  try {
    // Construct the environment first: it registers the default logger, and
    // appending a provider uses it.
    Ort::Env& env = environment();

    Ort::SessionOptions options;
    options.DisablePerSessionThreads();
    options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    Ort::Session session(env, graph.c_str(), options);

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
