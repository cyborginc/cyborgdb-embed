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

// Return local paths, downloading and verifying first if absent. Verification is
// not optional: an unverified file can differ from the one the parity run
// measured, and nothing downstream would notice. That applies to the tokenizer
// as much as the graph — a changed tokenizer moves every vector.
Status ensure_graph(const RegistryEntry&, const CacheConfig&, std::string& path);
Status ensure_tokenizer(const RegistryEntry&, const CacheConfig&, std::string& path);

Status sha256_file(const std::string& path, std::string& digest);

}  // namespace cyborgdb::embed::detail

#endif  // CYBORGDB_EMBED_RESOLVE_HPP
