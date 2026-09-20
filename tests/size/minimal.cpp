// The smallest complete use of the library, linked to measure what a consumer
// actually pays for it.
//
// It includes only the public header, so a private header that leaked into the
// public API fails to compile here.
#include <cyborgdb_embed/embed.hpp>

#include <cstdio>
#include <string_view>
#include <vector>

int main() {
  namespace embed = cyborgdb::embed;

  embed::Embedder embedder;
  const embed::Status opened =
      embed::open(embed::ModelId::BgeSmallEnV15, {}, embedder);
  if (!opened.ok()) {
    std::puts(opened.message.c_str());
    return 1;
  }

  const std::string_view text = "hello world";
  std::vector<float> vector(embedder.dimension());
  const embed::Status embedded =
      embedder.embed_documents(&text, 1, vector.data(), vector.size());
  if (!embedded.ok()) {
    std::puts(embedded.message.c_str());
    return 1;
  }

  std::printf("%zu dimensions\n", embedder.dimension());
  return 0;
}
