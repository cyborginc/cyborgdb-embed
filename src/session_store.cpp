#include "session_store.hpp"

namespace cyborgdb::embed::detail {

SessionStore& SessionStore::instance() {
  static SessionStore store;
  return store;
}

Status SessionStore::configure(const CacheConfig& config) {
  std::lock_guard<std::mutex> guard(mutex_);
  config_ = config;
  return {};
}

Status SessionStore::acquire(const RegistryEntry& entry, Provider provider,
                             Precision precision, int threads,
                             std::shared_ptr<Session>& out) {
  const Key key{entry.info.id, provider, precision};
  std::shared_future<std::shared_ptr<Session>> pending;
  std::shared_ptr<std::promise<std::shared_ptr<Session>>> promise;
  CacheConfig config;

  {
    std::lock_guard<std::mutex> guard(mutex_);
    config = config_;
    auto it = entries_.find(key);

    if (it != entries_.end()) {
      if (auto live = it->second.session.lock()) {
        ++shared_;
        out = std::move(live);
        return {};
      }
      // A load already in flight is still worth waiting on, even though the
      // weak entry has expired: the loader will publish into it.
      if (it->second.pending.valid()) {
        pending = it->second.pending;
      }
    }

    if (!pending.valid()) {
      ++loaded_;
      promise = std::make_shared<std::promise<std::shared_ptr<Session>>>();
      Entry fresh;
      fresh.pending = promise->get_future().share();
      pending = fresh.pending;
      entries_[key] = std::move(fresh);
    }
  }

  // Loading happens outside the lock: it downloads and can take seconds.
  if (promise) {
    std::shared_ptr<Session> session;
    const Status status =
        load_session(entry, provider, precision, threads, config, session);
    promise->set_value(status.ok() ? session : nullptr);

    std::lock_guard<std::mutex> guard(mutex_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      // A failure leaves nothing behind, so the next caller retries rather than
      // inheriting a stale error.
      if (!status.ok()) {
        entries_.erase(it);
      } else {
        it->second.session = session;
        it->second.pending = {};
      }
    }
    if (!status.ok()) return status;
    out = std::move(session);
    return {};
  }

  out = pending.get();
  if (!out) {
    return {StatusCode::ModelLoadFailed, "a concurrent load of this model failed"};
  }
  return {};
}

std::vector<LoadedModel> SessionStore::loaded() const {
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<LoadedModel> result;
  for (const auto& [key, entry] : entries_) {
    if (auto live = entry.session.lock()) {
      // One reference is the local lock above; the rest are callers.
      result.push_back({key.model, key.provider, key.precision,
                        static_cast<int>(live.use_count()) - 1});
    }
  }
  return result;
}

LoadStats SessionStore::stats() const noexcept {
  std::lock_guard<std::mutex> guard(mutex_);
  return {shared_, loaded_};
}

}  // namespace cyborgdb::embed::detail
