#include "tokenizer.hpp"

#include <fstream>
#include <sstream>

namespace {

struct TokenizerHandle;

extern "C" {
TokenizerHandle* cyborgdb_tokenizer_new(const unsigned char* json, std::size_t len,
                                        std::size_t max_length);
void cyborgdb_tokenizer_free(TokenizerHandle* handle);
int cyborgdb_tokenizer_encode(const TokenizerHandle* handle, const char* text,
                              std::uint32_t* ids, std::size_t capacity,
                              std::size_t* written);
}

}  // namespace

namespace cyborgdb::embed::detail {

Tokenizer::~Tokenizer() {
  cyborgdb_tokenizer_free(static_cast<TokenizerHandle*>(handle_));
}

Status Tokenizer::load(const std::string& path, std::size_t max_tokens,
                       std::unique_ptr<Tokenizer>& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return {StatusCode::TokenizerFailed, "cannot read " + path};
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  const std::string json = buffer.str();

  TokenizerHandle* handle = cyborgdb_tokenizer_new(
      reinterpret_cast<const unsigned char*>(json.data()), json.size(), max_tokens);
  if (handle == nullptr) {
    return {StatusCode::TokenizerFailed, path + " is not a valid tokenizer"};
  }
  out.reset(new Tokenizer(handle, max_tokens));
  return {};
}

Status Tokenizer::encode(std::string_view text, std::string_view prefix,
                         std::vector<std::uint32_t>& ids) const {
  std::string input;
  input.reserve(prefix.size() + text.size());
  input.append(prefix).append(text);

  std::size_t written = 0;
  ids.resize(max_tokens_);
  const int rc = cyborgdb_tokenizer_encode(
      static_cast<const TokenizerHandle*>(handle_), input.c_str(), ids.data(),
      ids.size(), &written);
  if (rc != 0) {
    return {StatusCode::TokenizerFailed, "encoding failed"};
  }
  ids.resize(written);
  return {};
}

}  // namespace cyborgdb::embed::detail
