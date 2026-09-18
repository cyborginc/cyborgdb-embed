# Imports the prebuilt ONNX Runtime archive for this platform.
#
# The archives are committed under onnxruntime/prebuilt and built by
# scripts/build_ort.sh, so consuming projects never build ONNX Runtime — it takes
# 10 to 20 minutes. Microsoft's own releases cannot be substituted: they are
# shared libraries, built without RTTI (which must match the consumer's ABI) and
# without contrib ops (whose absence breaks linux/arm64), and they are not
# symbol-isolated. Each archive here exports only the public C API, so ONNX
# Runtime's vendored abseil, protobuf, re2 and flatbuffers cannot collide with a
# consumer's own copies.

if(APPLE)
  set(_ort_platform "darwin-${CMAKE_SYSTEM_PROCESSOR}")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_ort_platform "linux-arm64")
  else()
    set(_ort_platform "linux-amd64")
  endif()
else()
  message(FATAL_ERROR "No prebuilt ONNX Runtime for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(CYBORGDB_EMBED_ORT_LIBRARY
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/prebuilt/${_ort_platform}/libonnxruntime.a"
    CACHE FILEPATH "Prebuilt ONNX Runtime archive")
set(CYBORGDB_EMBED_ORT_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include"
    CACHE PATH "ONNX Runtime headers")

if(NOT EXISTS "${CYBORGDB_EMBED_ORT_LIBRARY}")
  message(FATAL_ERROR
    "No ONNX Runtime archive at ${CYBORGDB_EMBED_ORT_LIBRARY}. "
    "Run scripts/build_ort.sh, or set CYBORGDB_EMBED_ORT_LIBRARY.")
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
