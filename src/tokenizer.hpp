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

  // Truncation is fixed here, and applied by the tokenizer so the trailing
  // separator token survives. Trimming ids afterwards would drop it.
  static Status load(const std::string& tokenizer_json_path,
                     std::size_t max_tokens, std::unique_ptr<Tokenizer>& out);

  // The caller's prefix is applied before tokenisation, not after: a prefix
  // tokenises differently joined to the text than on its own.
  Status encode(std::string_view text, std::string_view prefix,
                std::vector<std::uint32_t>& ids) const;

 private:
  Tokenizer(void* handle, std::size_t max_tokens) noexcept
      : handle_(handle), max_tokens_(max_tokens) {}

  void* handle_;
  std::size_t max_tokens_;
};

}  // namespace cyborgdb::embed::detail

#endif  // CYBORGDB_EMBED_TOKENIZER_HPP
