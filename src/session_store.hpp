#ifndef CYBORGDB_EMBED_SESSION_STORE_HPP
#define CYBORGDB_EMBED_SESSION_STORE_HPP

#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "session.hpp"

namespace cyborgdb::embed::detail {

// Hands out shared sessions and holds none of its own.
//
// Entries are weak: a session lives exactly as long as some caller holds it, so
// opening one model from many indexes loads the weights once, and the last
// handle going away frees them. Nothing is retained beyond use, which makes
// resident memory a function of what is open rather than of an eviction policy
// that has to guess.
//
// Process-wide. Scoping it any narrower would hold one copy of the weights per
// scope, which is the duplication this library exists to remove.
class SessionStore {
 public:
  static SessionStore& instance();

  Status configure(const CacheConfig&);

  // Concurrent opens of one model load once; the rest wait on that load.
  Status acquire(const RegistryEntry&, Provider, Precision,
                 std::shared_ptr<Session>& out);

  std::vector<LoadedModel> loaded() const;
  LoadStats stats() const noexcept;

 private:
  SessionStore() = default;

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
    std::weak_ptr<Session> session;
    std::shared_future<std::shared_ptr<Session>> pending;
  };

  mutable std::mutex mutex_;
  std::map<Key, Entry> entries_;
  CacheConfig config_;
  std::size_t shared_ = 0;
  std::size_t loaded_ = 0;
};

}  // namespace cyborgdb::embed::detail

#endif  // CYBORGDB_EMBED_SESSION_STORE_HPP
