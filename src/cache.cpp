#include "cache.hpp"

#include <algorithm>

namespace cyborgdb::embed::detail {
namespace {

std::chrono::seconds idle_for(std::chrono::steady_clock::time_point last) {
  return std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now() - last);
}

}  // namespace

SessionCache& SessionCache::instance() {
  static SessionCache cache;
  return cache;
}

Status SessionCache::configure(const CacheConfig& config) {
  std::lock_guard<std::mutex> guard(mutex_);
  config_ = config;
  evict_locked();
  return {};
}

CacheConfig SessionCache::config() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return config_;
}

Status SessionCache::acquire(const RegistryEntry& entry, Provider provider,
                             Precision precision, int threads,
                             std::shared_ptr<Session>& out) {
  const Key key{entry.info.id, provider, precision};
  std::shared_future<std::shared_ptr<Session>> pending;
  std::shared_ptr<std::promise<std::shared_ptr<Session>>> promise;

  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      it->second.last_used = std::chrono::steady_clock::now();
      ++hits_;
      pending = it->second.pending;
    } else {
      ++misses_;
      promise = std::make_shared<std::promise<std::shared_ptr<Session>>>();
      Entry fresh;
      fresh.pending = promise->get_future().share();
      fresh.last_used = std::chrono::steady_clock::now();
      pending = fresh.pending;
      entries_.emplace(key, std::move(fresh));
    }
  }

  // Loading happens outside the lock: it downloads and can take seconds.
  if (promise) {
    std::shared_ptr<Session> session;
    CacheConfig config;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      config = config_;
    }
    const Status status =
        load_session(entry, provider, precision, threads, config, session);
    promise->set_value(status.ok() ? session : nullptr);

    std::lock_guard<std::mutex> guard(mutex_);
    auto it = entries_.find(key);
    if (it == entries_.end()) {
      return status;
    }
    // A failed load is not cached; the next caller retries rather than
    // inheriting a stale error.
    if (!status.ok()) {
      entries_.erase(it);
      return status;
    }
    it->second.session = session;
    evict_locked();
    out = session;
    return {};
  }

  out = pending.get();
  if (!out) {
    return {StatusCode::ModelLoadFailed, "a concurrent load of this model failed"};
  }
  return {};
}

// Evicts by resident bytes rather than entry count: models differ by an order of
// magnitude. Only entries nothing else holds are candidates, so a model backing
// an open index is pinned and eviction cannot thrash.
void SessionCache::evict_locked() {
  const auto expired = [&](const Entry& entry) {
    return config_.idle_ttl.count() > 0 && entry.session &&
           idle_for(entry.last_used) > config_.idle_ttl;
  };

  for (auto it = entries_.begin(); it != entries_.end();) {
    it = (expired(it->second) && it->second.session.use_count() == 1)
             ? entries_.erase(it)
             : std::next(it);
  }

  if (config_.max_resident_bytes == 0) {
    return;
  }

  const auto resident = [&] {
    std::size_t total = 0;
    for (const auto& [key, entry] : entries_) {
      if (entry.session) total += entry.session->resident_bytes;
    }
    return total;
  };

  while (resident() > config_.max_resident_bytes) {
    auto oldest = entries_.end();
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (!it->second.session || it->second.session.use_count() != 1) continue;
      if (oldest == entries_.end() ||
          it->second.last_used < oldest->second.last_used) {
        oldest = it;
      }
    }
    // Every remaining session is in use. Exceeding the cap beats failing a call
    // that has already been admitted; the cap is a target, not a hard limit.
    if (oldest == entries_.end()) break;
    entries_.erase(oldest);
  }
}

std::vector<LoadedModel> SessionCache::loaded() const {
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<LoadedModel> result;
  result.reserve(entries_.size());
  for (const auto& [key, entry] : entries_) {
    if (!entry.session) continue;
    result.push_back({key.model, key.provider, key.precision,
                      entry.session->resident_bytes,
                      idle_for(entry.last_used),
                      static_cast<int>(entry.session.use_count()) - 1});
  }
  return result;
}

CacheStats SessionCache::stats() const noexcept {
  std::lock_guard<std::mutex> guard(mutex_);
  std::size_t resident = 0;
  for (const auto& [key, entry] : entries_) {
    if (entry.session) resident += entry.session->resident_bytes;
  }
  return {hits_, misses_, resident};
}

Status SessionCache::drop(ModelId model, Provider provider, Precision precision) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = entries_.find({model, provider, precision});
  if (it == entries_.end()) {
    return {StatusCode::InvalidArgument, "model is not loaded"};
  }
  entries_.erase(it);
  return {};
}

}  // namespace cyborgdb::embed::detail
