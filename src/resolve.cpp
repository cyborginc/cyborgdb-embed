#include "resolve.hpp"

#include <curl/curl.h>
#include <openssl/evp.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace cyborgdb::embed::detail {
namespace {

namespace fs = std::filesystem;

std::string env(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr ? std::string(value) : std::string();
}

std::string default_cache_dir() {
  if (std::string dir = env("CYBORGDB_EMBED_CACHE"); !dir.empty()) return dir;
  if (std::string dir = env("XDG_CACHE_HOME"); !dir.empty()) {
    return dir + "/cyborgdb-embed";
  }
  return env("HOME") + "/.cache/cyborgdb-embed";
}

// "BAAI/bge-base-en-v1.5" -> "BAAI__bge-base-en-v1.5", so one directory level
// holds every model regardless of how many slashes an id contains.
std::string flatten(std::string_view id) {
  std::string flat(id);
  for (char& c : flat) {
    if (c == '/') c = '_';
  }
  return flat;
}

std::size_t write_to_file(void* data, std::size_t size, std::size_t count,
                          void* sink) {
  auto* out = static_cast<std::ofstream*>(sink);
  out->write(static_cast<const char*>(data), static_cast<std::streamsize>(size * count));
  return out->good() ? size * count : 0;
}

Status download(const std::string& url, const fs::path& destination) {
  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return {StatusCode::DownloadFailed, "could not initialise curl"};
  }

  std::ofstream out(destination, std::ios::binary);
  if (!out) {
    curl_easy_cleanup(curl);
    return {StatusCode::DownloadFailed, "cannot write to " + destination.string()};
  }

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_file);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);

  std::array<char, CURL_ERROR_SIZE> error{};
  curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error.data());

  const CURLcode code = curl_easy_perform(curl);
  curl_easy_cleanup(curl);
  out.close();

  if (code != CURLE_OK) {
    std::error_code ignored;
    fs::remove(destination, ignored);
    const std::string detail = error[0] != '\0' ? error.data() : curl_easy_strerror(code);
    return {StatusCode::DownloadFailed, url + ": " + detail};
  }
  return {};
}

}  // namespace

CacheConfig resolved_config(const CacheConfig& given) {
  CacheConfig config = given;
  if (config.cache_dir.empty()) config.cache_dir = default_cache_dir();
  if (config.endpoint.empty()) {
    std::string endpoint = env("CYBORGDB_EMBED_ENDPOINT");
    config.endpoint = endpoint.empty() ? "https://huggingface.co" : endpoint;
  }
  if (!config.offline) config.offline = !env("CYBORGDB_EMBED_OFFLINE").empty();
  return config;
}

std::string cached_graph_path(const RegistryEntry& entry,
                              const CacheConfig& config) {
  const std::string_view path = entry.onnx_path;
  const std::size_t slash = path.rfind('/');
  const std::string_view filename =
      slash == std::string_view::npos ? path : path.substr(slash + 1);

  return (fs::path(resolved_config(config).cache_dir) / flatten(entry.info.name) /
          std::string(entry.info.revision) / std::string(filename))
      .string();
}

Status sha256_file(const std::string& path, std::string& digest) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {StatusCode::DownloadFailed, "cannot read " + path};

  EVP_MD_CTX* context = EVP_MD_CTX_new();
  EVP_DigestInit_ex(context, EVP_sha256(), nullptr);

  std::array<char, 1 << 16> buffer{};
  while (in.read(buffer.data(), buffer.size()) || in.gcount() > 0) {
    EVP_DigestUpdate(context, buffer.data(), static_cast<std::size_t>(in.gcount()));
  }

  std::array<unsigned char, EVP_MAX_MD_SIZE> raw{};
  unsigned int length = 0;
  EVP_DigestFinal_ex(context, raw.data(), &length);
  EVP_MD_CTX_free(context);

  static constexpr char kHex[] = "0123456789abcdef";
  digest.clear();
  digest.reserve(length * 2);
  for (unsigned int i = 0; i < length; ++i) {
    digest.push_back(kHex[raw[i] >> 4]);
    digest.push_back(kHex[raw[i] & 0x0f]);
  }
  return {};
}

namespace {

Status ensure_file(std::string_view model, std::string_view revision,
                   std::string_view repo_path, std::string_view expected,
                   const CacheConfig& config, const std::string& path) {
  if (fs::exists(path)) return {};

  if (config.offline) {
    return {StatusCode::NotCached,
            std::string(model) + " is not cached and offline is set"};
  }

  std::error_code code;
  fs::create_directories(fs::path(path).parent_path(), code);
  if (code) {
    return {StatusCode::DownloadFailed, "cannot create cache directory: " + code.message()};
  }

  const std::string url = config.endpoint + "/" + std::string(model) + "/resolve/" +
                          std::string(revision) + "/" + std::string(repo_path);

  // Download beside the target and rename, so an interrupted transfer never
  // leaves a partial file that later looks cached.
  const fs::path partial = path + ".partial";
  if (Status status = download(url, partial); !status) return status;

  std::string digest;
  if (Status status = sha256_file(partial.string(), digest); !status) return status;

  if (digest != expected) {
    fs::remove(partial, code);
    return {StatusCode::DigestMismatch, std::string(model) + "/" +
                std::string(repo_path) + ": expected " + std::string(expected) +
                ", got " + digest};
  }

  fs::rename(partial, path, code);
  if (code) {
    return {StatusCode::DownloadFailed, "cannot finalise " + path + ": " + code.message()};
  }
  return {};
}

}  // namespace

Status ensure_graph(const RegistryEntry& entry, const CacheConfig& given,
                    std::string& path) {
  const CacheConfig config = resolved_config(given);
  path = cached_graph_path(entry, config);
  return ensure_file(entry.info.name, entry.info.revision, entry.onnx_path,
                     entry.onnx_sha256, config, path);
}

Status ensure_tokenizer(const RegistryEntry& entry, const CacheConfig& given,
                        std::string& path) {
  const CacheConfig config = resolved_config(given);
  path = (fs::path(cached_graph_path(entry, config)).parent_path() / "tokenizer.json")
             .string();
  return ensure_file(entry.info.name, entry.info.revision, "tokenizer.json",
                     entry.tokenizer_sha256, config, path);
}

}  // namespace cyborgdb::embed::detail
