#ifndef CYBORGDB_EMBED_TOKENIZER_HPP
#define CYBORGDB_EMBED_TOKENIZER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cyborgdb_embed/embed.hpp"

namespace cyborgdb::embed::detail {

// Wraps the HuggingFace tokenizers library. It is the implementation
// sentence-transformers uses, and no C++ equivalent reproduces it: the closest
// one turns every accented character into [UNK].
class Tokenizer {
 public:
  ~Tokenizer();

  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  static Status load(const std::string& tokenizer_json_path,
                     std::unique_ptr<Tokenizer>& out);

  // Ids for one string, truncated to max_tokens. The caller's prefix is applied
  // before tokenisation, not after: a prefix tokenises differently joined to the
  // text than on its own.
  Status encode(std::string_view text, std::string_view prefix,
                std::size_t max_tokens, std::vector<std::uint32_t>& ids) const;

 private:
  explicit Tokenizer(void* handle) noexcept : handle_(handle) {}

  void* handle_;
};

}  // namespace cyborgdb::embed::detail

#endif  // CYBORGDB_EMBED_TOKENIZER_HPP
