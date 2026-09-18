// Scores the library against committed reference vectors.
//
// This is the fast gate: no Python, no model download beyond the cache, and it
// runs on every change. It answers "did the output move", which the nightly
// parity run cannot answer quickly and which matters on every commit.

#include <cyborgdb_embed/embed.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace embed = cyborgdb::embed;

namespace {

// Real prose carries the verdict. Degenerate inputs are scored beside it: a
// handful of pathological strings can drag a whole-corpus minimum far below
// what every real sentence achieves, and conflating them hides which you have.
constexpr double kCosineFloor = 1.0 - 1e-6;
constexpr double kMaxAbsDiff = 1e-4;

std::vector<std::string> read_texts(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string blob = buffer.str();

  std::vector<std::string> texts;
  for (std::size_t start = 0; start < blob.size();) {
    const std::size_t end = blob.find('\0', start);
    if (end == std::string::npos) break;
    texts.emplace_back(blob.substr(start, end - start));
    start = end + 1;
  }
  return texts;
}

std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream in(path);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);) lines.push_back(line);
  return lines;
}

std::vector<float> read_vectors(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return {};
  const auto bytes = static_cast<std::size_t>(in.tellg());
  in.seekg(0);
  std::vector<float> data(bytes / sizeof(float));
  in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
  return data;
}

struct Score {
  double cosine_min = 1.0;
  double max_abs_diff = 0.0;
  bool passes() const { return cosine_min >= kCosineFloor && max_abs_diff < kMaxAbsDiff; }
};

Score score(const std::vector<float>& reference, const std::vector<float>& candidate,
            const std::vector<std::string>& groups, std::size_t dim,
            const char* want_group) {
  Score result;
  for (std::size_t row = 0; row < groups.size(); ++row) {
    if (std::strcmp(groups[row].c_str(), want_group) != 0) continue;
    double dot = 0, a2 = 0, b2 = 0;
    for (std::size_t d = 0; d < dim; ++d) {
      const double a = reference[row * dim + d], b = candidate[row * dim + d];
      dot += a * b;
      a2 += a * a;
      b2 += b * b;
      result.max_abs_diff = std::max(result.max_abs_diff, std::abs(a - b));
    }
    const double norm = std::sqrt(a2) * std::sqrt(b2);
    result.cosine_min = std::min(result.cosine_min, norm > 0 ? dot / norm : 1.0);
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: golden_test <testdata-dir> <model> [model...]\n");
    return 2;
  }
  const std::string testdata = argv[1];
  const auto texts = read_texts(testdata + "/corpus.bin");
  const auto groups = read_lines(testdata + "/corpus.groups");
  if (texts.empty() || texts.size() != groups.size()) {
    std::fprintf(stderr, "corpus fixtures missing or inconsistent\n");
    return 2;
  }

  std::vector<std::string_view> views(texts.begin(), texts.end());
  int failures = 0, checked = 0;

  for (int i = 2; i < argc; ++i) {
    const std::string name = argv[i];
    std::string slug = name;
    for (char& c : slug) {
      if (c == '/') c = '_';
    }

    std::size_t count = 0;
    const embed::ModelInfo* all = embed::supported_models(count);
    const embed::ModelInfo* info = nullptr;
    for (std::size_t m = 0; m < count; ++m) {
      if (all[m].name == name) info = &all[m];
    }
    if (info == nullptr) {
      std::printf("%-44s SKIP  not in registry\n", name.c_str());
      continue;
    }

    embed::Embedder model;
    if (auto status = embed::open(info->id, embed::Options{}, model); !status) {
      std::printf("%-44s FAIL  open: %s\n", name.c_str(), status.message.c_str());
      ++failures;
      continue;
    }

    const std::size_t dim = model.dimension();
    std::vector<float> vectors(views.size() * dim);

    for (const char* kind : {"document", "query"}) {
      const auto reference =
          read_vectors(testdata + "/golden/" + slug + "/reference-" + kind + ".f32");
      if (reference.size() != views.size() * dim) {
        std::printf("%-44s SKIP  no %s golden\n", name.c_str(), kind);
        continue;
      }

      const auto status =
          std::strcmp(kind, "query") == 0
              ? model.embed_queries(views.data(), views.size(), vectors.data(), vectors.size())
              : model.embed_documents(views.data(), views.size(), vectors.data(), vectors.size());
      if (!status) {
        std::printf("%-44s FAIL  embed %s: %s\n", name.c_str(), kind, status.message.c_str());
        ++failures;
        continue;
      }

      const Score real = score(reference, vectors, groups, dim, "real");
      const Score edge = score(reference, vectors, groups, dim, "edge");
      ++checked;
      const bool ok = real.passes();
      failures += ok ? 0 : 1;
      std::printf("%-44s %-8s %s  real cos=%.7f absdiff=%.2e | edge cos=%.7f\n",
                  name.c_str(), kind, ok ? "PASS" : "FAIL", real.cosine_min,
                  real.max_abs_diff, edge.cosine_min);
    }
  }

  std::printf("\n%d checked, %d failed\n", checked, failures);
  return failures == 0 ? 0 : 1;
}
