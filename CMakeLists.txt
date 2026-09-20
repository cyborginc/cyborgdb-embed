cmake_minimum_required(VERSION 3.24)

project(cyborgdb_embed
  VERSION 0.1.0
  DESCRIPTION "Text embedding numerically equivalent to sentence-transformers"
  LANGUAGES CXX)

# Standard and RTTI must match the consuming build. Both feed abseil's ABI
# options, and ONNX Runtime vendors its own abseil under identical symbol names.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(cyborgdb_embed_top_level ON)
else()
  set(cyborgdb_embed_top_level OFF)
endif()

option(CYBORGDB_EMBED_BUILD_TESTS "Build tests" ${cyborgdb_embed_top_level})
option(CYBORGDB_EMBED_COVERAGE "Instrument for coverage" OFF)

# Rebuild the committed archives instead of failing when one is absent. Opt-in:
# ONNX Runtime takes 10 to 20 minutes and the tokenizer needs a Rust toolchain.
option(CYBORGDB_EMBED_BUILD_VENDORED "Build vendored archives from source" OFF)

if(CYBORGDB_EMBED_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate)
  else()
    add_compile_options(--coverage)
    add_link_options(--coverage)
  endif()
endif()

# Hand ONNX Runtime an existing source tree for a dependency it would otherwise
# vendor at its own pin. Empty means ORT uses its pin, which is what a standalone
# build wants. A parent sets these to trees it already built.
set(CYBORGDB_EMBED_ABSL_SOURCE_DIR        "" CACHE PATH "abseil source for ONNX Runtime")
set(CYBORGDB_EMBED_FLATBUFFERS_SOURCE_DIR "" CACHE PATH "flatbuffers source for ONNX Runtime")
set(CYBORGDB_EMBED_RE2_SOURCE_DIR         "" CACHE PATH "re2 source for ONNX Runtime")
set(CYBORGDB_EMBED_PROTOBUF_SOURCE_DIR    "" CACHE PATH "protobuf source for ONNX Runtime")

find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

add_library(cyborgdb_embed STATIC
  src/session_store.cpp
  src/resolve.cpp
  src/embed.cpp
  src/ort_api.cpp
  src/tokenizer.cpp
  src/session.cpp
  src/session_ort.cpp)
add_library(cyborgdb::embed ALIAS cyborgdb_embed)

target_include_directories(cyborgdb_embed
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_features(cyborgdb_embed PUBLIC cxx_std_17)

include(cmake/onnxruntime.cmake)
include(cmake/tokenizer.cmake)
target_link_libraries(cyborgdb_embed
  PRIVATE cyborgdb::onnxruntime cyborgdb::tokenizer CURL::libcurl OpenSSL::Crypto)

# Embeds the frozen corpus so the parity harness can score the library against
# sentence-transformers. Built with the tests, not part of the library.
if(CYBORGDB_EMBED_BUILD_TESTS)
  add_executable(embed_corpus tests/parity/embed_corpus.cpp)
  target_link_libraries(embed_corpus PRIVATE cyborgdb::embed)

  # Linked only to measure the footprint a consumer inherits, and to prove the
  # public header stands alone. scripts/check_size.sh holds the budget.
  add_executable(size_probe tests/size/minimal.cpp)
  target_link_libraries(size_probe PRIVATE cyborgdb::embed)

  add_executable(bench tests/bench/bench.cpp)
  target_link_libraries(bench PRIVATE cyborgdb::embed)
  find_package(Threads REQUIRED)
  target_link_libraries(bench PRIVATE Threads::Threads)

  add_executable(pooling_test tests/unit/pooling_test.cpp)
  target_link_libraries(pooling_test PRIVATE cyborgdb::embed)
  target_include_directories(pooling_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  add_test(NAME pooling COMMAND pooling_test)

  add_executable(error_test tests/unit/error_test.cpp)
  target_link_libraries(error_test PRIVATE cyborgdb::embed)
  add_test(NAME errors COMMAND error_test)

  add_executable(golden_test tests/golden_test.cpp)
  target_link_libraries(golden_test PRIVATE cyborgdb::embed)

  enable_testing()
  add_test(NAME golden
           COMMAND golden_test "${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
                   BAAI/bge-small-en-v1.5 intfloat/e5-small-v2)
endif()
