#ifndef CYBORGDB_EMBED_POOLING_HPP
#define CYBORGDB_EMBED_POOLING_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "registry_generated.hpp"

namespace cyborgdb::embed::detail {

// Reduces a model's per-token hidden states to one vector per row.
//
// `hidden` is rows x width x dim, `mask` is rows x width and marks real tokens.
// Padded positions are excluded, which is what makes batching by length
// output-equivalent to batching in any other order.
inline void pool(const float* hidden, const std::int64_t* mask, std::size_t rows,
                 std::size_t width, std::size_t dim, Pooling mode, float* out) {
  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = hidden + r * width * dim;
    float* dest = out + r * dim;

    if (mode == Pooling::ClsToken) {
      std::copy_n(row, dim, dest);
      continue;
    }

    if (mode == Pooling::MaxTokens) {
      std::fill_n(dest, dim, -std::numeric_limits<float>::infinity());
      for (std::size_t t = 0; t < width; ++t) {
        if (!mask[r * width + t]) continue;
        for (std::size_t d = 0; d < dim; ++d) {
          dest[d] = std::max(dest[d], row[t * dim + d]);
        }
      }
      continue;
    }

    std::fill_n(dest, dim, 0.0f);
    float count = 0.0f;
    for (std::size_t t = 0; t < width; ++t) {
      if (!mask[r * width + t]) continue;
      count += 1.0f;
      for (std::size_t d = 0; d < dim; ++d) {
        dest[d] += row[t * dim + d];
      }
    }
    // An all-padding row divides by zero otherwise, and empty input is a real
    // case rather than a caller error.
    const float divisor = count > 0.0f ? count : 1.0f;
    for (std::size_t d = 0; d < dim; ++d) dest[d] /= divisor;
  }
}

// Scales each row to unit length. A row that is already zero is left alone:
// there is no direction to preserve, and dividing would produce NaN.
inline void l2_normalize(float* vectors, std::size_t rows, std::size_t dim) {
  for (std::size_t r = 0; r < rows; ++r) {
    float* row = vectors + r * dim;
    float sum = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) sum += row[d] * row[d];
    const float norm = std::sqrt(sum);
    if (norm <= 1e-12f) continue;
    for (std::size_t d = 0; d < dim; ++d) row[d] /= norm;
  }
}

}  // namespace cyborgdb::embed::detail

#endif  // CYBORGDB_EMBED_POOLING_HPP
