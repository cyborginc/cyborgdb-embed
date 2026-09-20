# Imports the prebuilt ONNX Runtime archive for this platform.
#
# The archives are published as release assets and fetched on demand, so a
# consumer downloads only the platform it runs on and never builds ONNX Runtime,
# which takes 10 to 20 minutes. Microsoft's own releases cannot be substituted: they are
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

include(${CMAKE_CURRENT_LIST_DIR}/artifacts.cmake)

# An explicit path wins over everything: it is how an air-gapped build, or one
# verifying the binary from source, supplies its own archive.
set(CYBORGDB_EMBED_ORT_LIBRARY "" CACHE FILEPATH "ONNX Runtime archive to link")
set(CYBORGDB_EMBED_ORT_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include"
    CACHE PATH "ONNX Runtime headers")

if(NOT CYBORGDB_EMBED_ORT_LIBRARY AND NOT CYBORGDB_EMBED_BUILD_VENDORED)
  cyborgdb_embed_fetch_artifact(onnxruntime "${_ort_platform}" "${CYBORGDB_EMBED_ORT_TAG}" _fetched)
  set(CYBORGDB_EMBED_ORT_LIBRARY "${_fetched}" CACHE FILEPATH "" FORCE)
endif()

if(CYBORGDB_EMBED_BUILD_VENDORED AND NOT CYBORGDB_EMBED_ORT_LIBRARY)
  set(CYBORGDB_EMBED_ORT_LIBRARY
      "${CMAKE_CURRENT_SOURCE_DIR}/.ort/install/lib/libonnxruntime_isolated.a"
      CACHE FILEPATH "" FORCE)
endif()

if(NOT EXISTS "${CYBORGDB_EMBED_ORT_LIBRARY}")
  if(CYBORGDB_EMBED_BUILD_VENDORED)
    # Deliberately opt-in: this takes 10 to 20 minutes, and a configure step that
    # silently does that is worse than one that fails in a second saying how.
    message(STATUS "Building ONNX Runtime from source; this takes 10-20 minutes")
    execute_process(
      COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_ort.sh" all
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE _ort_build)
    if(NOT _ort_build EQUAL 0)
      message(FATAL_ERROR "scripts/build_ort.sh failed")
    endif()
  else()
    message(FATAL_ERROR
      "No ONNX Runtime archive at ${CYBORGDB_EMBED_ORT_LIBRARY}.\n"
      "Run scripts/build_ort.sh, configure with -DCYBORGDB_EMBED_BUILD_VENDORED=ON "
      "to build it here, or set CYBORGDB_EMBED_ORT_LIBRARY to an existing archive.")
  endif()
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

# The archive leaves zlib, libm and the dynamic loader undefined for whoever
# links it. macOS resolves all three out of libSystem without being asked, so
# they only have to be named elsewhere.
find_package(ZLIB REQUIRED)
target_link_libraries(cyborgdb_onnxruntime INTERFACE ZLIB::ZLIB)
if(NOT APPLE)
  target_link_libraries(cyborgdb_onnxruntime INTERFACE dl m)
endif()
