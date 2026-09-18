# Imports the prebuilt tokenizer archive for this platform.
#
# The archives are committed under tokenizer/prebuilt and built by
# scripts/build_tokenizer.sh, so consuming projects never need a Rust toolchain.
# Each exports only the three C entry points; the Rust runtime is localised out.

if(APPLE)
  set(_tokenizer_platform "darwin-${CMAKE_SYSTEM_PROCESSOR}")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_tokenizer_platform "linux-arm64")
  else()
    set(_tokenizer_platform "linux-amd64")
  endif()
else()
  message(FATAL_ERROR "No prebuilt tokenizer for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(CYBORGDB_EMBED_TOKENIZER_LIBRARY
    "${CMAKE_CURRENT_SOURCE_DIR}/tokenizer/prebuilt/${_tokenizer_platform}/libcyborgdb_tokenizer.a"
    CACHE FILEPATH "Prebuilt tokenizer archive")

if(NOT EXISTS "${CYBORGDB_EMBED_TOKENIZER_LIBRARY}")
  if(CYBORGDB_EMBED_BUILD_VENDORED)
    message(STATUS "Building the tokenizer from source; this needs cargo")
    execute_process(
      COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_tokenizer.sh" all
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE _tokenizer_build)
    if(NOT _tokenizer_build EQUAL 0)
      message(FATAL_ERROR "scripts/build_tokenizer.sh failed; is cargo installed?")
    endif()
  else()
    message(FATAL_ERROR
      "No tokenizer archive at ${CYBORGDB_EMBED_TOKENIZER_LIBRARY}.\n"
      "Run scripts/build_tokenizer.sh, or configure with "
      "-DCYBORGDB_EMBED_BUILD_VENDORED=ON to build it here. Needs cargo.")
  endif()
endif()

add_library(cyborgdb_tokenizer STATIC IMPORTED GLOBAL)
add_library(cyborgdb::tokenizer ALIAS cyborgdb_tokenizer)
set_target_properties(cyborgdb_tokenizer PROPERTIES
  IMPORTED_LOCATION "${CYBORGDB_EMBED_TOKENIZER_LIBRARY}")

if(APPLE)
  target_link_libraries(cyborgdb_tokenizer INTERFACE
    "-framework CoreFoundation" "-framework Security")
else()
  target_link_libraries(cyborgdb_tokenizer INTERFACE dl m)
endif()
