// Checks the per-input flag reporting that a text was cut at the model's limit.
// Needs the network or a populated cache, like the golden test.

#include <cyborgdb_embed/embed.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace embed = cyborgdb::embed;

namespace {

int failures = 0;
int checks = 0;

void expect(bool condition, const char* what) {
  ++checks;
  if (!condition) {
    ++failures;
    std::printf("  FAIL  %s\n", what);
  }
}

// "the" is a single token in every registry vocabulary, so n repetitions
// encode to n tokens plus the model's special tokens.
std::string words(std::size_t n) {
  std::string text;
  for (std::size_t i = 0; i < n; ++i) text += i == 0 ? "the" : " the";
  return text;
}

bool open(embed::ModelId id, embed::Embedder& out) {
  const embed::Status status = embed::open(id, {}, out);
  if (!status.ok()) std::printf("  FAIL  cannot open a model: %s\n", status.message.c_str());
  return status.ok();
}

void mixed_batch() {
  embed::Embedder model;
  if (!open(embed::ModelId::AllMiniLmL6V2, model)) {
    ++failures;
    return;
  }
  // The long input sits in the middle so the length sort reorders the batch.
  const std::string long_text = words(600);
  const std::vector<std::string_view> texts{"a short ticket", long_text, "another short one"};
  const std::size_t dim = model.dimension();

  std::vector<float> flagged(texts.size() * dim);
  std::vector<std::uint8_t> truncated(texts.size(), 0xff);
  expect(model.embed_documents(texts.data(), texts.size(), flagged.data(),
                               flagged.size(), truncated.data())
             .ok(),
         "embed_documents succeeds with flags");
  expect(truncated == std::vector<std::uint8_t>{0, 1, 0},
         "only the input past the limit is flagged, at its own position");

  std::vector<float> plain(texts.size() * dim);
  expect(model.embed_documents(texts.data(), texts.size(), plain.data(), plain.size())
             .ok(),
         "embed_documents succeeds without flags");
  expect(flagged == plain, "asking for flags does not change the vectors");

  std::vector<std::uint8_t> query_flags(texts.size(), 0xff);
  expect(model.embed_queries(texts.data(), texts.size(), plain.data(), plain.size(),
                             query_flags.data())
             .ok(),
         "embed_queries succeeds with flags");
  expect(query_flags == std::vector<std::uint8_t>{0, 1, 0}, "queries are flagged too");

  // Text after an embedded NUL still counts, so it can push an input over.
  const std::string with_nul = std::string("a short ticket") + '\0' + long_text;
  const std::string_view nul_view = with_nul;
  std::uint8_t nul_flag = 0xff;
  std::vector<float> nul_out(dim);
  expect(model.embed_documents(&nul_view, 1, nul_out.data(), nul_out.size(), &nul_flag)
                 .ok() &&
             nul_flag == 1,
         "text after an embedded NUL is embedded and counted");

  // Invalid UTF-8 fails in the tokenizer after the first input has been
  // encoded, so a flag written eagerly would already be in place.
  const std::vector<std::string_view> failing{"a short ticket", "\xff"};
  std::vector<std::uint8_t> untouched(failing.size(), 0xff);
  std::vector<float> discarded(failing.size() * dim);
  expect(model.embed_documents(failing.data(), failing.size(), discarded.data(),
                               discarded.size(), untouched.data())
                 .code == embed::StatusCode::TokenizerFailed,
         "invalid UTF-8 fails in the tokenizer");
  expect(untouched == std::vector<std::uint8_t>(failing.size(), 0xff),
         "flags are not written when the call fails");
}

// BGE prefixes queries but not documents, so a text that exactly fills the
// limit as a document is cut as a query.
void prefix_counts_against_the_limit() {
  embed::Embedder model;
  if (!open(embed::ModelId::BgeSmallEnV15, model)) {
    ++failures;
    return;
  }
  const std::size_t limit = embed::info(embed::ModelId::BgeSmallEnV15).max_seq_length;
  const std::string exact = words(limit - 2);
  const std::string_view text = exact;
  std::vector<float> out(model.dimension());

  std::uint8_t truncated = 0xff;
  expect(model.embed_documents(&text, 1, out.data(), out.size(), &truncated).ok() &&
             truncated == 0,
         "a document that exactly fills the limit is not flagged");

  truncated = 0xff;
  expect(model.embed_queries(&text, 1, out.data(), out.size(), &truncated).ok() &&
             truncated == 1,
         "the query prefix pushes the same text over");
}

}  // namespace

int main() {
  mixed_batch();
  prefix_counts_against_the_limit();
  std::printf("%d checks, %d failed\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
