#include <cyborgdb_embed/embed.hpp>

#include <cassert>
#include <cstdio>

namespace embed = cyborgdb::embed;

int main() {
  std::size_t count = 0;
  const embed::ModelInfo* models = embed::supported_models(count);
  std::printf("registry: %zu models\n", count);
  assert(count == 10);

  const embed::ModelInfo& bge = embed::info(embed::ModelId::BgeBaseEnV15);
  std::printf("  %s dim=%zu max_seq=%zu normalize=%d\n", bge.name.data(),
              bge.dimension, bge.max_seq_length, bge.normalize);
  assert(bge.dimension == 768);
  assert(bge.max_seq_length == 512);

  const embed::ModelInfo& mini = embed::info(embed::ModelId::AllMiniLmL6V2);
  assert(mini.max_seq_length == 256);  // not max_position_embeddings (512)
  std::printf("  %s max_seq=%zu\n", mini.name.data(), mini.max_seq_length);

  assert(embed::provider_available(embed::Provider::CPU));
  std::printf("runtime: %s\n", embed::runtime_version().c_str());
  std::printf("provider CPU available: %d\n", embed::provider_available(embed::Provider::CPU));
  std::printf("provider CoreML available: %d\n", embed::provider_available(embed::Provider::CoreML));

  embed::Embedder e;
  assert(!e.valid());
  assert(e.dimension() == 0);

  embed::Status s = e.embed_documents(nullptr, 0, nullptr, 0);
  std::printf("embed before open: %s\n", s.message.c_str());
  assert(s.code == embed::StatusCode::InvalidArgument);

  // Asserting the exact code, not just failure: every model in the registry is
  // still unpinned, and a test that accepts any error cannot tell that apart
  // from a build that cannot embed at all.
  s = embed::open(embed::ModelId::BgeBaseEnV15, embed::Options{}, e);
  std::printf("open: %s\n", s.message.c_str());
  assert(s.code == embed::StatusCode::ModelLoadFailed);
  assert(!e.valid());

  embed::Options bad;
  bad.threads = 0;
  s = embed::open(embed::ModelId::BgeBaseEnV15, bad, e);
  std::printf("open with threads=0: %s\n", s.message.c_str());
  assert(s.code == embed::StatusCode::InvalidArgument);

  embed::CacheStats stats = embed::cache_stats();
  std::printf("cache: %zu hits, %zu misses, %zu resident, %zu loaded\n",
              stats.hits, stats.misses, stats.resident_bytes,
              embed::loaded_models().size());
  assert(embed::loaded_models().empty());  // a failed load is not cached

  (void)models;
  std::printf("\nall assertions passed\n");
  return 0;
}
