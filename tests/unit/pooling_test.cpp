// Pooling and normalization, exercised directly.
//
// Only one pooling mode runs during a golden comparison, because a model uses
// one. These are the reductions every vector passes through, so each mode is
// checked against a hand-computed result rather than against whichever model
// happens to be in the registry.

#include "pooling.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace detail = cyborgdb::embed::detail;

namespace {

int failures = 0;
int checks = 0;

void close_to(float got, float want, const char* what) {
  ++checks;
  if (std::abs(got - want) > 1e-6f) {
    ++failures;
    std::printf("  FAIL  %s: got %g, want %g\n", what, got, want);
  }
}

// Two rows, three tokens each, two dimensions. The second token of row 1 is
// padding, so anything that includes it is wrong.
const std::vector<float> kHidden = {
    1.0f, 2.0f,   3.0f, 4.0f,   5.0f, 6.0f,
    7.0f, 8.0f,   9.0f, 10.0f,  11.0f, 12.0f,
};
const std::vector<std::int64_t> kMask = {1, 1, 1, 1, 0, 0};

void cls_takes_the_first_token() {
  std::vector<float> out(4);
  detail::pool(kHidden.data(), kMask.data(), 2, 3, 2, detail::Pooling::ClsToken, out.data());
  close_to(out[0], 1.0f, "cls row 0 dim 0");
  close_to(out[1], 2.0f, "cls row 0 dim 1");
  close_to(out[2], 7.0f, "cls row 1 dim 0");
  close_to(out[3], 8.0f, "cls row 1 dim 1");
}

void mean_ignores_padding() {
  std::vector<float> out(4);
  detail::pool(kHidden.data(), kMask.data(), 2, 3, 2, detail::Pooling::MeanTokens, out.data());
  close_to(out[0], 3.0f, "mean row 0 dim 0");   // (1 + 3 + 5) / 3
  close_to(out[1], 4.0f, "mean row 0 dim 1");   // (2 + 4 + 6) / 3
  close_to(out[2], 7.0f, "mean row 1 dim 0");   // only the first token is real
  close_to(out[3], 8.0f, "mean row 1 dim 1");
}

void max_ignores_padding() {
  std::vector<float> out(4);
  detail::pool(kHidden.data(), kMask.data(), 2, 3, 2, detail::Pooling::MaxTokens, out.data());
  close_to(out[0], 5.0f, "max row 0 dim 0");
  close_to(out[1], 6.0f, "max row 0 dim 1");
  close_to(out[2], 7.0f, "max row 1 dim 0");
  close_to(out[3], 8.0f, "max row 1 dim 1");
}

void all_padding_does_not_divide_by_zero() {
  const std::vector<std::int64_t> empty_mask = {0, 0, 0};
  std::vector<float> out(2, 1234.0f);
  detail::pool(kHidden.data(), empty_mask.data(), 1, 3, 2, detail::Pooling::MeanTokens,
               out.data());
  close_to(out[0], 0.0f, "empty row pools to zero, not NaN");
  close_to(out[1], 0.0f, "empty row pools to zero, not NaN");
  ++checks;
  if (std::isnan(out[0]) || std::isnan(out[1])) {
    ++failures;
    std::printf("  FAIL  empty row produced NaN\n");
  }
}

void normalize_scales_to_unit_length() {
  std::vector<float> vectors = {3.0f, 4.0f, 1.0f, 0.0f};
  detail::l2_normalize(vectors.data(), 2, 2);
  close_to(vectors[0], 0.6f, "3-4-5 triangle, first component");
  close_to(vectors[1], 0.8f, "3-4-5 triangle, second component");
  close_to(vectors[2], 1.0f, "already unit length");
  close_to(vectors[3], 0.0f, "already unit length");

  float norm = 0;
  for (std::size_t d = 0; d < 2; ++d) norm += vectors[d] * vectors[d];
  close_to(std::sqrt(norm), 1.0f, "row has unit norm");
}

void normalize_leaves_a_zero_row_alone() {
  std::vector<float> zero = {0.0f, 0.0f};
  detail::l2_normalize(zero.data(), 1, 2);
  close_to(zero[0], 0.0f, "zero row stays zero");
  close_to(zero[1], 0.0f, "zero row stays zero");
  ++checks;
  if (std::isnan(zero[0])) {
    ++failures;
    std::printf("  FAIL  zero row normalized to NaN\n");
  }
}

}  // namespace

int main() {
  cls_takes_the_first_token();
  mean_ignores_padding();
  max_ignores_padding();
  all_padding_does_not_divide_by_zero();
  normalize_scales_to_unit_length();
  normalize_leaves_a_zero_row_alone();
  std::printf("%d checks, %d failed\n", checks, failures);
  return failures == 0 ? 0 : 1;
}
