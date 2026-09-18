// Throughput, latency and memory under concurrency.
//
// Two load models, because they answer different questions:
//
//   closed loop  N callers each embed as fast as they can. Answers "how many
//                cores do I need", which is what saturation throughput measures.
//   open loop    requests arrive at a fixed rate regardless of how fast they
//                complete. Answers "what is p99 at N requests per second",
//                which is what a query path actually experiences. A closed loop
//                cannot show queueing, because it never queues.

#include <cyborgdb_embed/embed.hpp>

#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace embed = cyborgdb::embed;
using Clock = std::chrono::steady_clock;

namespace {

struct Config {
  std::string model = "BAAI/bge-small-en-v1.5";
  std::string corpus = "tests/data/corpus.bin";
  int concurrency = 1;
  int threads = 1;
  int batch = 8;
  double seconds = 10.0;
  double rate = 0.0;  // requests per second; 0 selects the closed loop
};

std::vector<std::string> read_texts(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string blob = buffer.str();
  std::vector<std::string> texts;
  for (std::size_t start = 0; start < blob.size();) {
    const std::size_t end = blob.find('\0', start);
    if (end == std::string::npos) break;
    // Degenerate rows are for correctness, not timing.
    if (end > start) texts.emplace_back(blob.substr(start, end - start));
    start = end + 1;
  }
  return texts;
}

// Peak resident set. Linux reports kilobytes, Darwin bytes.
std::size_t peak_rss_bytes() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
  return static_cast<std::size_t>(usage.ru_maxrss);
#else
  return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
}

double percentile(std::vector<double>& samples, double q) {
  if (samples.empty()) return 0.0;
  const std::size_t index = static_cast<std::size_t>(q * (samples.size() - 1));
  std::nth_element(samples.begin(), samples.begin() + index, samples.end());
  return samples[index];
}

}  // namespace

int main(int argc, char** argv) {
  Config config;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string flag = argv[i], value = argv[i + 1];
    if (flag == "--model") config.model = value;
    else if (flag == "--corpus") config.corpus = value;
    else if (flag == "--concurrency") config.concurrency = std::stoi(value);
    else if (flag == "--threads") config.threads = std::stoi(value);
    else if (flag == "--batch") config.batch = std::stoi(value);
    else if (flag == "--seconds") config.seconds = std::stod(value);
    else if (flag == "--rate") config.rate = std::stod(value);
  }

  const auto texts = read_texts(config.corpus);
  if (texts.empty()) {
    std::fprintf(stderr, "no corpus at %s\n", config.corpus.c_str());
    return 2;
  }

  std::size_t count = 0;
  const embed::ModelInfo* all = embed::supported_models(count);
  const embed::ModelInfo* info = nullptr;
  for (std::size_t i = 0; i < count; ++i) {
    if (all[i].name == config.model) info = &all[i];
  }
  if (info == nullptr) {
    std::fprintf(stderr, "%s is not in the registry\n", config.model.c_str());
    return 2;
  }

  embed::Options options;
  options.threads = config.threads;
  embed::Embedder model;
  if (auto status = embed::open(info->id, options, model); !status) {
    std::fprintf(stderr, "open: %s\n", status.message.c_str());
    return 1;
  }

  const std::size_t dim = model.dimension();
  const std::size_t rss_after_load = peak_rss_bytes();

  // One warm pass: the first call pays for graph warmup and arena growth, and
  // charging that to the measurement would misreport steady state.
  {
    std::vector<std::string_view> warm(texts.begin(), texts.begin() + config.batch);
    std::vector<float> out(warm.size() * dim);
    model.embed_documents(warm.data(), warm.size(), out.data(), out.size());
  }

  std::mutex lock;
  std::vector<double> latencies;
  std::atomic<std::size_t> embedded{0};
  std::atomic<bool> stop{false};
  const auto started = Clock::now();
  const auto deadline = started + std::chrono::duration<double>(config.seconds);

  const auto worker = [&](int id) {
    std::vector<float> out(static_cast<std::size_t>(config.batch) * dim);
    std::vector<double> local;
    std::size_t cursor = static_cast<std::size_t>(id) * 97;

    // Open loop: each caller owns a slice of the arrival schedule, so a slow
    // response delays that request rather than the ones behind it.
    const double interval = config.rate > 0
        ? config.concurrency / config.rate
        : 0.0;
    auto next = started;

    while (!stop.load(std::memory_order_relaxed)) {
      if (Clock::now() >= deadline) break;
      if (interval > 0) {
        next += std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(interval));
        if (next > Clock::now()) std::this_thread::sleep_until(next);
      }

      std::vector<std::string_view> batch;
      batch.reserve(config.batch);
      for (int b = 0; b < config.batch; ++b) {
        batch.push_back(texts[(cursor++) % texts.size()]);
      }

      const auto call = Clock::now();
      const auto status =
          model.embed_documents(batch.data(), batch.size(), out.data(), out.size());
      const double elapsed =
          std::chrono::duration<double, std::milli>(Clock::now() - call).count();
      if (!status) {
        std::fprintf(stderr, "embed: %s\n", status.message.c_str());
        stop.store(true);
        break;
      }
      local.push_back(elapsed);
      embedded.fetch_add(batch.size(), std::memory_order_relaxed);
    }

    std::lock_guard<std::mutex> guard(lock);
    latencies.insert(latencies.end(), local.begin(), local.end());
  };

  std::vector<std::thread> workers;
  for (int i = 0; i < config.concurrency; ++i) workers.emplace_back(worker, i);
  for (auto& thread : workers) thread.join();

  const double wall = std::chrono::duration<double>(Clock::now() - started).count();
  const std::size_t rss = peak_rss_bytes();

  std::printf("{\n");
  std::printf("  \"model\": \"%s\",\n", config.model.c_str());
  std::printf("  \"load\": \"%s\",\n", config.rate > 0 ? "open" : "closed");
  std::printf("  \"concurrency\": %d,\n", config.concurrency);
  std::printf("  \"intra_op_threads\": %d,\n", config.threads);
  std::printf("  \"batch\": %d,\n", config.batch);
  if (config.rate > 0) std::printf("  \"target_rate\": %.1f,\n", config.rate);
  std::printf("  \"wall_seconds\": %.3f,\n", wall);
  std::printf("  \"sentences\": %zu,\n", embedded.load());
  std::printf("  \"throughput_per_s\": %.1f,\n", embedded.load() / wall);
  std::printf("  \"calls\": %zu,\n", latencies.size());
  std::printf("  \"p50_ms\": %.2f,\n", percentile(latencies, 0.50));
  std::printf("  \"p95_ms\": %.2f,\n", percentile(latencies, 0.95));
  std::printf("  \"p99_ms\": %.2f,\n", percentile(latencies, 0.99));
  std::printf("  \"rss_after_load_bytes\": %zu,\n", rss_after_load);
  std::printf("  \"peak_rss_bytes\": %zu\n", rss);
  std::printf("}\n");
  return 0;
}
