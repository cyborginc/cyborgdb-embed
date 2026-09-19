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
  "onnxruntime-darwin-arm64  PENDING"
  "onnxruntime-linux-amd64   PENDING"
  "onnxruntime-linux-arm64   PENDING"
  "tokenizer-darwin-arm64    PENDING"
  "tokenizer-linux-amd64     PENDING"
  "tokenizer-linux-arm64     PENDING"
)

# Fetches one archive into a tag-keyed cache shared by every build directory, so
# configuring twice or switching build types does not download again.
function(cyborgdb_embed_fetch_artifact name platform tag out_path)
  set(local "${CMAKE_CURRENT_SOURCE_DIR}/.artifacts/${tag}/lib${name}-${platform}.a")
  set(${out_path} "${local}" PARENT_SCOPE)
  if(EXISTS "${local}")
    return()
  endif()

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
endfunction()
