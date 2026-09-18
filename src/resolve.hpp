#ifndef CYBORGDB_EMBED_RESOLVE_HPP
#define CYBORGDB_EMBED_RESOLVE_HPP

#include <string>

#include "cyborgdb_embed/embed.hpp"
#include "registry_generated.hpp"

namespace cyborgdb::embed::detail {

// Environment defaults for anything the caller left unset.
CacheConfig resolved_config(const CacheConfig&);

// Where a model's graph lives once cached. Keyed by revision, so two revisions
// of one model coexist and nothing needs invalidating.
std::string cached_graph_path(const RegistryEntry&, const CacheConfig&);

// Returns the local path, downloading and verifying first if it is absent.
// Verification is not optional: an unverified graph can differ from the one the
// parity run measured, and nothing downstream would notice.
Status ensure_graph(const RegistryEntry&, const CacheConfig&, std::string& path);

Status sha256_file(const std::string& path, std::string& digest);

}  // namespace cyborgdb::embed::detail

#endif  // CYBORGDB_EMBED_RESOLVE_HPP
