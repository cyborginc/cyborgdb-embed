#ifndef CYBORGDB_EMBED_CACHE_HPP
#define CYBORGDB_EMBED_CACHE_HPP

#include <chrono>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "session.hpp"

namespace cyborgdb::embed::detail {

// Process-wide. Scoping it per client would hold one copy of the weights per
// client, which is the duplication this library exists to remove.
class SessionCache {
 public:
  static SessionCache& instance();

  Status configure(const CacheConfig&);
  CacheConfig config() const;

  // Concurrent misses on one key load once; the rest wait on that load.
  Status acquire(const RegistryEntry&, Provider, Precision, int threads,
                 std::shared_ptr<Session>& out);

  std::vector<LoadedModel> loaded() const;
  CacheStats stats() const noexcept;

  // Drops the cache's reference. A session a caller still holds stays alive.
  Status drop(ModelId, Provider, Precision);

 private:
  SessionCache() = default;

  struct Key {
    ModelId model;
    Provider provider;
    Precision precision;
    bool operator<(const Key& other) const {
      if (model != other.model) return model < other.model;
      if (provider != other.provider) return provider < other.provider;
      return precision < other.precision;
    }
  };

  struct Entry {
    std::shared_future<std::shared_ptr<Session>> pending;
    std::shared_ptr<Session> session;
    std::chrono::steady_clock::time_point last_used;
    std::string error;
  };

  void evict_locked();

  mutable std::mutex mutex_;
  std::map<Key, Entry> entries_;
  CacheConfig config_;
  std::size_t hits_ = 0;
  std::size_t misses_ = 0;
};

}  // namespace cyborgdb::embed::detail

#endif  // CYBORGDB_EMBED_CACHE_HPP
