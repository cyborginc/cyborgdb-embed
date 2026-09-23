# Which prebuilt archives this source tree expects, and their digests.
#
# Each component has its own tag series, named for the upstream version it
# contains plus a build number: ort-1.30.0-2 is the second build we published of
# ONNX Runtime 1.30.0. The build number matters because a build-flag fix
# reissues an archive without the upstream version moving, which happened
# repeatedly while the flags were being settled.
#
# Separate series mean an ONNX Runtime bump does not republish the tokenizer, or
# invalidate anyone's cached copy of it.
#
# They are also separate from library version tags, which avoids a circularity:
# a library tag cannot contain digests of artifacts built by tagging it.
#
# Versions live in versions.json, which is the only place either is written.
# To publish: push the component tag, let the release workflow build and upload,
# then paste the digests it prints into this file.

# Composed from versions.json rather than written here, so a version cannot be
# bumped in one place and forgotten in another.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/versions.json" _versions)
string(JSON _ort_version GET "${_versions}" onnxruntime version)
string(JSON _ort_build GET "${_versions}" onnxruntime build)
string(JSON _tok_version GET "${_versions}" tokenizers version)
string(JSON _tok_build GET "${_versions}" tokenizers build)

set(CYBORGDB_EMBED_ORT_TAG "ort-${_ort_version}-${_ort_build}"
    CACHE STRING "Release tag for the ONNX Runtime archives")
set(CYBORGDB_EMBED_TOKENIZER_TAG "tokenizers-${_tok_version}-${_tok_build}"
    CACHE STRING "Release tag for the tokenizer archives")
set(CYBORGDB_EMBED_ARTIFACT_REPO "https://github.com/cyborginc/cyborgdb-embed"
    CACHE STRING "Repository hosting the artifact releases")

# sha256 of each published archive. A download that does not match is discarded:
# these are binaries nobody can review, so the pin is what makes them trustable.
set(CYBORGDB_EMBED_ARTIFACT_DIGESTS
  "onnxruntime-darwin-arm64  8298f53dd9044ad1f1e15a87ae2e42cc0a4125de1fec00e25ed1b90cb159bfe2"
  "onnxruntime-linux-amd64   2c8794b2ab577f6381cf013882417fc5492a13f937d28ae5c771a0c24111b084"
  "onnxruntime-linux-arm64   c7f9cf02741add1b3b29dbc4fce5748ae149b1c48f066e79b08affb82bf75409"
  "tokenizer-darwin-arm64    PENDING"
  "tokenizer-linux-amd64     PENDING"
  "tokenizer-linux-arm64     PENDING"
)

option(CYBORGDB_EMBED_STRIP_ARTIFACTS
       "Discard local symbols from the prebuilt archives before linking" ON)

# Removes the ONNX Runtime archive's local symbols, which are two thirds of its
# size and none of its interface. They belong to ONNX Runtime rather than to
# whoever links it, so dropping them shrinks a consumer's binary by a third
# without touching the symbols they debug their own code with.
#
# Only that archive. The tokenizer's references its own Rust personality routine
# as a local symbol, so stripping leaves an archive that still links elsewhere
# but fails on that one undefined symbol -- and strip reports success, so the
# breakage would surface as a confusing link error rather than here.
#
# The verified download is left alone and the result written beside it, so the
# file the digest describes stays on disk exactly as it arrived.
function(cyborgdb_embed_strip_artifact name archive out_path)
  set(${out_path} "${archive}" PARENT_SCOPE)
  if(NOT CYBORGDB_EMBED_STRIP_ARTIFACTS OR NOT CMAKE_STRIP
     OR NOT name STREQUAL "onnxruntime")
    return()
  endif()

  string(REGEX REPLACE "\\.a$" "-local-stripped.a" stripped "${archive}")
  set(${out_path} "${stripped}" PARENT_SCOPE)
  if(EXISTS "${stripped}")
    return()
  endif()

  # -x on Mach-O, --discard-all elsewhere: both keep every global the linker
  # resolves against and drop the rest.
  if(APPLE)
    set(flag -x)
  else()
    set(flag --discard-all)
  endif()

  configure_file("${archive}" "${stripped}" COPYONLY)
  execute_process(COMMAND "${CMAKE_STRIP}" ${flag} "${stripped}"
                  RESULT_VARIABLE stripped_result ERROR_QUIET)
  if(NOT stripped_result EQUAL 0)
    # Nothing here is required; fall back to the archive as downloaded.
    file(REMOVE "${stripped}")
    set(${out_path} "${archive}" PARENT_SCOPE)
  endif()
endfunction()

# Fetches one archive into a tag-keyed cache shared by every build directory, so
# configuring twice or switching build types does not download again.
function(cyborgdb_embed_fetch_artifact name platform tag out_path)
  set(local "${CMAKE_CURRENT_SOURCE_DIR}/.artifacts/${tag}/lib${name}-${platform}.a")
  if(EXISTS "${local}")
    cyborgdb_embed_strip_artifact("${name}" "${local}" usable)
    set(${out_path} "${usable}" PARENT_SCOPE)
    return()
  endif()
  set(${out_path} "${local}" PARENT_SCOPE)

  set(digest "")
  foreach(entry IN LISTS CYBORGDB_EMBED_ARTIFACT_DIGESTS)
    string(REGEX MATCH "^${name}-${platform} +([0-9a-fA-F]+|PENDING)$" matched "${entry}")
    if(matched)
      set(digest "${CMAKE_MATCH_1}")
    endif()
  endforeach()

  if(digest STREQUAL "" OR digest STREQUAL "PENDING")
    message(FATAL_ERROR
      "No digest recorded for ${name}-${platform} at ${tag}.\n"
      "Push the ${tag} tag to publish it, or configure with "
      "-DCYBORGDB_EMBED_BUILD_VENDORED=ON to build from source.")
  endif()

  set(url "${CYBORGDB_EMBED_ARTIFACT_REPO}/releases/download/${tag}/lib${name}-${platform}.a")
  message(STATUS "Fetching ${name} for ${platform} from ${tag}")
  file(DOWNLOAD "${url}" "${local}.partial"
       EXPECTED_HASH "SHA256=${digest}"
       STATUS download_status
       SHOW_PROGRESS)
  list(GET download_status 0 code)
  if(NOT code EQUAL 0)
    list(GET download_status 1 reason)
    file(REMOVE "${local}.partial")
    message(FATAL_ERROR "Could not fetch ${url}: ${reason}")
  endif()
  # Renamed only after the hash matched, so an interrupted fetch never looks
  # like a complete one.
  file(RENAME "${local}.partial" "${local}")

  cyborgdb_embed_strip_artifact("${name}" "${local}" usable)
  set(${out_path} "${usable}" PARENT_SCOPE)
endfunction()
