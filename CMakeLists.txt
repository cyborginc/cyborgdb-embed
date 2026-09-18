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
  src/cache.cpp
  src/resolve.cpp
  src/embed.cpp
  src/ort_api.cpp
  src/tokenizer.cpp
  src/session.cpp)
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
