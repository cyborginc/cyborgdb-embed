# Imports the merged ONNX Runtime archive produced by scripts/build_ort.sh.
# ORT drives its own build system, so it is built out of band rather than added
# as a subproject.

set(CYBORGDB_EMBED_ORT_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/.ort/install"
    CACHE PATH "Prefix holding the merged ONNX Runtime archive and headers")

find_library(CYBORGDB_EMBED_ORT_LIBRARY
  NAMES onnxruntime_merged
  PATHS "${CYBORGDB_EMBED_ORT_PREFIX}/lib"
  NO_DEFAULT_PATH)

find_path(CYBORGDB_EMBED_ORT_INCLUDE_DIR
  NAMES onnxruntime_c_api.h
  PATHS "${CYBORGDB_EMBED_ORT_PREFIX}/include"
  NO_DEFAULT_PATH)

if(NOT CYBORGDB_EMBED_ORT_LIBRARY OR NOT CYBORGDB_EMBED_ORT_INCLUDE_DIR)
  message(FATAL_ERROR
    "ONNX Runtime not found under ${CYBORGDB_EMBED_ORT_PREFIX}. "
    "Run scripts/build_ort.sh first, or set CYBORGDB_EMBED_ORT_PREFIX.")
endif()

add_library(cyborgdb_onnxruntime STATIC IMPORTED GLOBAL)
add_library(cyborgdb::onnxruntime ALIAS cyborgdb_onnxruntime)

set_target_properties(cyborgdb_onnxruntime PROPERTIES
  IMPORTED_LOCATION "${CYBORGDB_EMBED_ORT_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${CYBORGDB_EMBED_ORT_INCLUDE_DIR}")

if(APPLE)
  target_link_libraries(cyborgdb_onnxruntime INTERFACE
    "-framework Foundation" "-framework Accelerate")
endif()

find_package(Threads REQUIRED)
target_link_libraries(cyborgdb_onnxruntime INTERFACE Threads::Threads)
