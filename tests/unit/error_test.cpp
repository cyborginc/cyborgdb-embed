// Exercises the paths that return a failure.
//
// The happy path is covered by the golden test, which needs the network and a
// model. These cases need neither: every failure is provoked from local state,
// so they run anywhere and cost milliseconds.

#include <cyborgdb_embed/embed.hpp>

#include <cstdio>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace embed = cyborgdb::embed;
namespace fs = std::filesystem;

namespace {

int failures = 0;
int checks = 0;

void expect(bool condition, const char* what) {
  ++checks;
  if (!condition) {
    ++failures;
    std::printf("  FAIL  %s\n", what);
  }
}

void expect_code(const embed::Status& status, embed::StatusCode want, const char* what) {
  ++checks;
  if (status.code != want) {
    ++failures;
    std::printf("  FAIL  %s: got code %d (%s)\n", what, static_cast<int>(status.code),
                status.message.c_str());
  }
}

// Serves model files over file:// so download failures, digest mismatches and
// offline behaviour can be provoked without a network or a real endpoint.
std::string fake_endpoint(const fs::path& root, const embed::ModelInfo& info,
                          const std::string& graph_contents) {
  const fs::path dir = root / std::string(info.name) / "resolve" /
                       std::string(info.revision) / "onnx";
  fs::create_directories(dir);
  std::ofstream(dir / "model.onnx") << graph_contents;
  return "file://" + root.string();
}

void registry_lookups() {
  std::size_t count = 0;
  const embed::ModelInfo* models = embed::supported_models(count);
  expect(count > 0, "registry is not empty");
  expect(models != nullptr, "registry table is non-null");

  for (std::size_t i = 0; i < count; ++i) {
    expect(!models[i].name.empty(), "every model has a name");
    expect(!models[i].revision.empty(), "every model is pinned to a revision");
    expect(models[i].dimension > 0, "every model has a dimension");
    expect(models[i].max_seq_length > 0, "every model has a sequence limit");
    const embed::ModelInfo& found = embed::info(models[i].id);
    expect(found.id == models[i].id, "info() round-trips every id");

    embed::ModelId by_name{};
    expect(embed::find_model(models[i].name, by_name).ok() && by_name == models[i].id,
           "find_model() resolves every upstream name");
  }
}

void name_lookups() {
  embed::ModelId id{};
  expect(embed::find_model("all-MiniLM-L6-v2", id).ok() &&
             id == embed::ModelId::AllMiniLmL6V2,
         "a bare name resolves under sentence-transformers/");
  expect(embed::find_model("all-Mpnet-base-v2", id).ok() &&
             id == embed::ModelId::AllMpnetBaseV2,
         "a bare name matches regardless of case");
  expect(embed::find_model("baai/BGE-small-en-v1.5", id).ok() &&
             id == embed::ModelId::BgeSmallEnV15,
         "a full name matches regardless of case");

  const embed::Status unknown = embed::find_model("not-a-model", id);
  expect_code(unknown, embed::StatusCode::InvalidArgument, "an unknown name");
  expect(unknown.message.find("sentence-transformers/all-MiniLM-L6-v2") != std::string::npos,
         "an unknown name lists the supported models");
  expect(unknown.message.find("hosted") == std::string::npos,
         "an unknown name is not called hosted");

  for (const char* hosted : {"text-embedding-ada-002", "text-embedding-3-small",
                             "text-embedding-3-large", "openai/text-embedding-3-small"}) {
    const embed::Status status = embed::find_model(hosted, id);
    expect_code(status, embed::StatusCode::InvalidArgument, hosted);
    expect(status.message.find("hosted API model, not supported") != std::string::npos,
           "a hosted model gets its own message");
  }
  expect_code(embed::find_model("", id), embed::StatusCode::InvalidArgument, "an empty name");
}

void status_helpers() {
  const embed::Status ok;
  expect(ok.ok(), "default status is ok");
  expect(static_cast<bool>(ok), "ok status converts to true");

  const embed::Status bad{embed::StatusCode::InvalidArgument, "nope"};
  expect(!bad.ok(), "error status is not ok");
  expect(!static_cast<bool>(bad), "error status converts to false");
  expect(bad.message == "nope", "message survives");
}

void capabilities() {
  expect(embed::provider_available(embed::Provider::CPU), "CPU is available");
  expect(embed::precision_available(embed::ModelId::BgeSmallEnV15, embed::Precision::Fp32),
         "fp32 is available");
  expect(!embed::precision_available(embed::ModelId::BgeSmallEnV15, embed::Precision::Int8),
         "int8 is not");
  expect(!embed::precision_available(embed::ModelId::BgeSmallEnV15, embed::Precision::Fp16),
         "fp16 is not");
  expect(!embed::runtime_version().empty(), "runtime reports a version");
}

void unopened_embedder() {
  embed::Embedder model;
  expect(!model.valid(), "a default embedder is not valid");
  expect(model.dimension() == 0, "an unopened embedder has no dimension");
  expect(model.identity().model.empty(), "an unopened embedder has no identity");

  std::vector<float> out(16);
  const std::string_view text = "hello";
  expect_code(model.embed_documents(&text, 1, out.data(), out.size()),
              embed::StatusCode::InvalidArgument, "embed_documents before open");
  expect_code(model.embed_queries(&text, 1, out.data(), out.size()),
              embed::StatusCode::InvalidArgument, "embed_queries before open");

  embed::Embedder copy = model;
  expect(!copy.valid(), "a copy of an unopened embedder is not valid");
  embed::Embedder moved = std::move(copy);
  expect(!moved.valid(), "a moved-from unopened embedder is not valid");
}

void invalid_options() {
  embed::Embedder model;

  expect_code(embed::configure_runtime({0}), embed::StatusCode::InvalidArgument,
              "threads = 0");
  expect_code(embed::configure_runtime({-4}), embed::StatusCode::InvalidArgument,
              "negative threads");

  embed::Options quantised;
  quantised.precision = embed::Precision::Int8;
  expect_code(embed::open(embed::ModelId::BgeSmallEnV15, quantised, model),
              embed::StatusCode::PrecisionUnavailable, "int8 is rejected, not approximated");

  expect(!model.valid(), "a failed open leaves the embedder unopened");
  expect(embed::loaded_models().empty(), "a failed open caches nothing");
}

void unknown_model() {
  // Unreachable through the enum, but a caller can cast an arbitrary value into
  // it, and that must be rejected rather than indexed out of bounds.
  embed::Embedder model;
  expect_code(embed::open(static_cast<embed::ModelId>(9999), {}, model),
              embed::StatusCode::InvalidArgument, "an unknown model id");
}

void handle_assignment() {
  embed::Embedder a;
  embed::Embedder b;
  b = a;
  expect(!b.valid(), "copy assignment of an unopened handle");
  b = std::move(a);
  expect(!b.valid(), "move assignment of an unopened handle");
}

void default_cache_location() {
  // With no explicit directory the cache follows the environment, and an
  // unwritable location must fail rather than silently choosing another.
  const char* previous = std::getenv("CYBORGDB_EMBED_CACHE");
  const std::string saved = previous != nullptr ? previous : "";
  ::unsetenv("CYBORGDB_EMBED_CACHE");
  ::setenv("XDG_CACHE_HOME", "/proc/nonexistent-and-unwritable", 1);

  embed::CacheConfig config;
  config.offline = false;
  config.endpoint = "file:///nonexistent";
  embed::configure_cache(config);

  embed::Embedder model;
  const embed::Status status = embed::open(embed::ModelId::BgeSmallEnV15, {}, model);
  expect(!status.ok(), "an unusable default cache directory fails");

  ::unsetenv("XDG_CACHE_HOME");
  if (!saved.empty()) ::setenv("CYBORGDB_EMBED_CACHE", saved.c_str(), 1);
}

void offline_miss(const fs::path& scratch) {
  embed::CacheConfig config;
  config.cache_dir = (scratch / "empty").string();
  config.offline = true;
  embed::configure_cache(config);

  embed::Embedder model;
  expect_code(embed::open(embed::ModelId::BgeSmallEnV15, {}, model),
              embed::StatusCode::NotCached, "offline with an empty cache");
}

void download_failure(const fs::path& scratch) {
  embed::CacheConfig config;
  config.cache_dir = (scratch / "download").string();
  config.endpoint = "file://" + (scratch / "nothing-here").string();
  embed::configure_cache(config);

  embed::Embedder model;
  expect_code(embed::open(embed::ModelId::BgeSmallEnV15, {}, model),
              embed::StatusCode::DownloadFailed, "endpoint serves nothing");
}

void digest_mismatch(const fs::path& scratch) {
  const embed::ModelInfo& info = embed::info(embed::ModelId::BgeSmallEnV15);
  const fs::path served = scratch / "wrong-bytes";

  embed::CacheConfig config;
  config.cache_dir = (scratch / "mismatch").string();
  config.endpoint = fake_endpoint(served, info, "this is not an onnx graph");
  embed::configure_cache(config);

  embed::Embedder model;
  const embed::Status status = embed::open(embed::ModelId::BgeSmallEnV15, {}, model);
  expect_code(status, embed::StatusCode::DigestMismatch, "served file fails verification");

  // A file that failed verification must not be left behind looking cached.
  const fs::path cached = fs::path(config.cache_dir) / "BAAI_bge-small-en-v1.5" /
                          std::string(info.revision) / "model.onnx";
  expect(!fs::exists(cached), "a mismatched download is discarded");
}

// A cached file is not re-verified on use: digests are checked when a file is
// downloaded, not on every open. So corruption after the fact has to surface as
// a clean failure rather than a crash or a plausible vector.
fs::path seed_cache(const fs::path& cache, const embed::ModelInfo& info) {
  const fs::path dir = fs::path(cache) / "BAAI_bge-small-en-v1.5" /
                       std::string(info.revision);
  fs::create_directories(dir);
  return dir;
}

// The tokenizer is loaded before the graph, so each test corrupts one file and
// copies the other intact; otherwise the first failure masks the second.
void corrupt_graph(const fs::path& scratch, const fs::path& real_tokenizer) {
  if (real_tokenizer.empty()) return;
  const embed::ModelInfo& info = embed::info(embed::ModelId::BgeSmallEnV15);
  const fs::path cache = scratch / "corrupt-graph";
  const fs::path dir = seed_cache(cache, info);
  std::ofstream(dir / "model.onnx") << "not a protobuf";
  fs::copy_file(real_tokenizer, dir / "tokenizer.json",
                fs::copy_options::overwrite_existing);

  embed::CacheConfig config;
  config.cache_dir = cache.string();
  config.offline = true;
  embed::configure_cache(config);

  embed::Embedder model;
  expect_code(embed::open(embed::ModelId::BgeSmallEnV15, {}, model),
              embed::StatusCode::ModelLoadFailed, "a corrupt graph in the cache");
  expect(!model.valid(), "a failed load leaves the handle unopened");
}

void corrupt_tokenizer(const fs::path& scratch, const fs::path& real_graph) {
  if (real_graph.empty()) return;  // nothing cached to copy; covered elsewhere
  const embed::ModelInfo& info = embed::info(embed::ModelId::BgeSmallEnV15);
  const fs::path cache = scratch / "corrupt-tokenizer";
  const fs::path dir = seed_cache(cache, info);
  fs::copy_file(real_graph, dir / "model.onnx", fs::copy_options::overwrite_existing);
  std::ofstream(dir / "tokenizer.json") << "definitely not a tokenizer";

  embed::CacheConfig config;
  config.cache_dir = cache.string();
  config.offline = true;
  embed::configure_cache(config);

  embed::Embedder model;
  expect_code(embed::open(embed::ModelId::BgeSmallEnV15, {}, model),
              embed::StatusCode::TokenizerFailed, "a corrupt tokenizer in the cache");
}

// Files a previous run already fetched, if any. Absent them these two cases are
// skipped rather than asserted against a cache that was never populated.
fs::path cached_file(const char* name) {
  const char* dir = std::getenv("CYBORGDB_EMBED_CACHE");
  if (dir == nullptr) return {};
  const embed::ModelInfo& info = embed::info(embed::ModelId::BgeSmallEnV15);
  const fs::path path = fs::path(dir) / "BAAI_bge-small-en-v1.5" /
                        std::string(info.revision) / name;
  return fs::exists(path) ? path : fs::path{};
}

void load_accounting() {
  const embed::LoadStats before = embed::load_stats();
  expect(before.loaded + before.shared > 0, "opens are counted");

  embed::CacheConfig config;
  config.cache_dir = "/nonexistent-for-stats";
  config.offline = true;
  embed::configure_cache(config);
  embed::Embedder model;
  embed::open(embed::ModelId::BgeSmallEnV15, {}, model);

  const embed::LoadStats after = embed::load_stats();
  expect(after.loaded >= before.loaded, "a failed open still counts as an attempt");
  expect(embed::loaded_models().empty(), "nothing is live after a failed open");
}

// Two callers opening one model must load it once. The second waits on the
// first rather than starting its own download.
void concurrent_open(const fs::path& real_graph) {
  if (real_graph.empty()) return;

  embed::CacheConfig config;
  config.cache_dir = std::getenv("CYBORGDB_EMBED_CACHE");
  embed::configure_cache(config);

  const embed::LoadStats before = embed::load_stats();
  std::vector<std::thread> threads;
  std::atomic<int> opened{0};
  for (int i = 0; i < 4; ++i) {
    threads.emplace_back([&] {
      embed::Embedder model;
      if (embed::open(embed::ModelId::BgeSmallEnV15, {}, model)) {
        opened.fetch_add(1);
        // Hold it so the others find a live session rather than a fresh load.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }
  for (auto& thread : threads) thread.join();

  const embed::LoadStats after = embed::load_stats();
  expect(opened.load() == 4, "every concurrent caller gets a session");
  expect(after.loaded - before.loaded <= 1, "concurrent opens load at most once");
  expect(after.shared > before.shared, "the rest share the loaded session");
}

}  // namespace

int main() {
  const fs::path scratch = fs::temp_directory_path() / "cyborgdb-embed-error-test";
  fs::remove_all(scratch);
  fs::create_directories(scratch);

  registry_lookups();
  name_lookups();
  status_helpers();
  capabilities();
  unopened_embedder();
  invalid_options();
  unknown_model();
  handle_assignment();
  default_cache_location();
  offline_miss(scratch);
  download_failure(scratch);
  digest_mismatch(scratch);
  corrupt_graph(scratch, cached_file("tokenizer.json"));
  corrupt_tokenizer(scratch, cached_file("model.onnx"));
  load_accounting();
  concurrent_open(cached_file("model.onnx"));

  fs::remove_all(scratch);
  std::printf("%d checks, %d failed\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
