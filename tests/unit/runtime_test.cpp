// The thread pool is process-wide: every model shares it, and its size is fixed
// once the first model loads. That state cannot be reset, so this is its own
// executable. Needs the network or a populated cache, like the golden test.

#include <cyborgdb_embed/embed.hpp>

#include <cstdio>
#include <string_view>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <mach/mach.h>
#else
#include <filesystem>
#endif

namespace embed = cyborgdb::embed;

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

int thread_count() {
#ifdef __APPLE__
  thread_act_array_t threads = nullptr;
  mach_msg_type_number_t count = 0;
  if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS) return -1;
  for (mach_msg_type_number_t i = 0; i < count; ++i) {
    mach_port_deallocate(mach_task_self(), threads[i]);
  }
  vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads),
                sizeof(*threads) * count);
  return static_cast<int>(count);
#else
  int count = 0;
  for ([[maybe_unused]] const auto& task :
       std::filesystem::directory_iterator("/proc/self/task")) {
    ++count;
  }
  return count;
#endif
}

int handles_of(embed::ModelId id) {
  for (const embed::LoadedModel& loaded : embed::loaded_models()) {
    if (loaded.model == id) return loaded.handles;
  }
  return 0;
}

bool embeds(const embed::Embedder& model) {
  const std::string_view text = "a sentence to embed";
  std::vector<float> out(model.dimension());
  return model.embed_documents(&text, 1, out.data(), out.size()).ok();
}

}  // namespace

int main() {
  // The caller runs work too, so a pool of N adds N - 1 threads.
  constexpr int kThreads = 4;

  expect(embed::configure_runtime({2}).ok(), "any value is accepted before the first open");
  expect(embed::configure_runtime({kThreads}).ok(), "and may be changed until then");

  const int before = thread_count();
  embed::Embedder bge;
  const embed::Status opened = embed::open(embed::ModelId::BgeSmallEnV15, {}, bge);
  if (!opened.ok()) {
    std::printf("  FAIL  cannot open a model: %s\n", opened.message.c_str());
    return 1;
  }
  const int after_first = thread_count();
  expect(after_first - before >= kThreads - 1, "the first open starts the pool");

  embed::Embedder e5;
  expect(embed::open(embed::ModelId::E5SmallV2, {}, e5).ok(), "a second model opens");
  const int after_second = thread_count();
  std::printf("threads: %d before, %d after one model, %d after two\n", before,
              after_first, after_second);
  expect(after_second - after_first < kThreads - 1, "a second model adds no pool");

  // Both models run at once on the one pool.
  bool bge_ok = false, e5_ok = false;
  std::thread a([&] { bge_ok = embeds(bge); });
  std::thread b([&] { e5_ok = embeds(e5); });
  a.join();
  b.join();
  expect(bge_ok && e5_ok, "two models embed concurrently on the shared pool");

  const embed::LoadStats stats_before = embed::load_stats();
  embed::Embedder bge_again;
  expect(embed::open(embed::ModelId::BgeSmallEnV15, {}, bge_again).ok(), "a model reopens");
  const embed::LoadStats stats_after = embed::load_stats();
  expect(stats_after.loaded == stats_before.loaded, "reopening loads no weights");
  expect(stats_after.shared == stats_before.shared + 1, "reopening shares the session");
  expect(handles_of(embed::ModelId::BgeSmallEnV15) == 2, "both handles hold one session");

  expect(embed::configure_runtime({kThreads}).ok(), "the value in force is accepted after open");
  expect(embed::configure_runtime({2}).code == embed::StatusCode::InvalidArgument,
         "a different value is rejected after open");

  bge = {};
  bge_again = {};
  e5 = {};
  expect(embed::loaded_models().empty(), "releasing every handle frees the models");
  expect(embed::configure_runtime({2}).code == embed::StatusCode::InvalidArgument,
         "the pool outlives the models, so the size stays fixed");

  std::printf("%d checks, %d failed\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
