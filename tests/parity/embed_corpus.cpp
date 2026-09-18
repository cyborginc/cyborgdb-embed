// Embeds the frozen corpus with the library and writes vectors in the same
// format as the Python arms, so the comparison never needs to know which arm
// produced a file.
//
// Texts arrive NUL-separated rather than as JSON: the corpus deliberately
// contains newlines, empty strings and control characters.

#include <cyborgdb_embed/embed.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace embed = cyborgdb::embed;

namespace {

std::vector<std::string> read_texts(const char* path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string blob = buffer.str();

  std::vector<std::string> texts;
  std::size_t start = 0;
  while (start <= blob.size()) {
    const std::size_t end = blob.find('\0', start);
    if (end == std::string::npos) break;
    texts.emplace_back(blob.substr(start, end - start));
    start = end + 1;
  }
  return texts;
}

int write(const std::string& path, const std::vector<float>& vectors,
          std::size_t rows, std::size_t dim, const char* kind) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(vectors.data()),
            static_cast<std::streamsize>(vectors.size() * sizeof(float)));
  if (!out) return 1;

  std::ofstream meta(path.substr(0, path.size() - 4) + ".json");
  meta << "{\n  \"arm\": \"embed\",\n  \"kind\": \"" << kind
       << "\",\n  \"rows\": " << rows << ",\n  \"dim\": " << dim << "\n}\n";
  return meta ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: embed_corpus <corpus.bin> <out-prefix> <model>\n");
    return 2;
  }

  const std::vector<std::string> texts = read_texts(argv[1]);
  if (texts.empty()) {
    std::fprintf(stderr, "no texts in %s\n", argv[1]);
    return 1;
  }

  std::size_t count = 0;
  const embed::ModelInfo* all = embed::supported_models(count);
  const embed::ModelInfo* chosen = nullptr;
  for (std::size_t i = 0; i < count; ++i) {
    if (all[i].name == argv[3]) chosen = &all[i];
  }
  if (chosen == nullptr) {
    std::fprintf(stderr, "%s is not in the registry\n", argv[3]);
    return 1;
  }

  embed::Embedder model;
  if (auto status = embed::open(chosen->id, embed::Options{}, model); !status) {
    std::fprintf(stderr, "open: %s\n", status.message.c_str());
    return 1;
  }

  std::vector<std::string_view> views(texts.begin(), texts.end());
  const std::size_t dim = model.dimension();
  std::vector<float> vectors(views.size() * dim);

  struct { const char* kind; bool query; } passes[] = {
      {"document", false}, {"query", true}};

  for (const auto& pass : passes) {
    const auto status =
        pass.query ? model.embed_queries(views.data(), views.size(), vectors.data(), vectors.size())
                   : model.embed_documents(views.data(), views.size(), vectors.data(), vectors.size());
    if (!status) {
      std::fprintf(stderr, "embed %s: %s\n", pass.kind, status.message.c_str());
      return 1;
    }
    const std::string path = std::string(argv[2]) + "-" + pass.kind + ".f32";
    if (write(path, vectors, views.size(), dim, pass.kind) != 0) {
      std::fprintf(stderr, "cannot write %s\n", path.c_str());
      return 1;
    }
    std::printf("  %s  (%zu, %zu)\n", path.c_str(), views.size(), dim);
  }
  return 0;
}
