#include "resolve.hpp"

#include <curl/curl.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>

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

// libcurl's default CA bundle is a path fixed at build time, which does not
// exist on every distro, so find one at run time. Empty keeps the default.
std::string ca_bundle() {
  for (const char* var : {"CURL_CA_BUNDLE", "SSL_CERT_FILE"}) {
    if (std::string path = env(var); !path.empty()) return path;
  }
  for (const char* path : {"/etc/ssl/certs/ca-certificates.crt",  // Debian/Ubuntu
                           "/etc/pki/tls/certs/ca-bundle.crt",    // RHEL/Fedora
                           "/etc/ssl/ca-bundle.pem",              // openSUSE
                           "/etc/ssl/cert.pem"}) {                // Alpine/macOS/BSD
    std::error_code ignored;
    if (fs::is_regular_file(path, ignored)) return path;
  }
  return {};
}

// Hugging Face rate-limits anonymous downloads per IP, which CI runners share; HF_TOKEN lifts that. It is sent only to
// huggingface.co (curl drops it on redirects to other hosts), never to a CYBORGDB_EMBED_ENDPOINT mirror.
std::string auth_header(const std::string& url) {
  if (url.rfind("https://huggingface.co/", 0) != 0) return {};
  std::string token = env("HF_TOKEN");
  if (token.empty()) token = env("HUGGING_FACE_HUB_TOKEN");
  return token.empty() ? std::string() : "Authorization: Bearer " + token;
}

// Added in curl 7.66.0; the manylinux_2_28 image has 7.61.1 headers. The value is ABI, so declaring it is safe.
#if LIBCURL_VERSION_NUM < 0x074200
constexpr CURLINFO CURLINFO_RETRY_AFTER = static_cast<CURLINFO>(CURLINFO_OFF_T + 57);
#endif

struct Attempt {
  Status status;
  bool transient = false;  // worth retrying: rate limit, server error, network hiccup
  long retry_after = 0;    // seconds the server asked for, if any
};

Attempt download_once(const std::string& url, const fs::path& destination) {
  CURL* curl = curl_easy_init();
  if (curl == nullptr) {
    return {{StatusCode::DownloadFailed, "could not initialise curl"}};
  }

  std::ofstream out(destination, std::ios::binary | std::ios::trunc);
  if (!out) {
    curl_easy_cleanup(curl);
    return {{StatusCode::DownloadFailed, "cannot write to " + destination.string()}};
  }

  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_file);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
  // No total timeout, since model files are large; abort a stalled transfer so it can be retried.
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
  curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
  curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 30L);
  const std::string ca = ca_bundle();
  if (!ca.empty()) curl_easy_setopt(curl, CURLOPT_CAINFO, ca.c_str());
  curl_slist* headers = nullptr;
  if (const std::string auth = auth_header(url); !auth.empty()) {
    headers = curl_slist_append(headers, auth.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  }

  std::array<char, CURL_ERROR_SIZE> error{};
  curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error.data());

  const CURLcode code = curl_easy_perform(curl);
  long http = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http);
  curl_off_t retry_after = 0;
  curl_easy_getinfo(curl, CURLINFO_RETRY_AFTER, &retry_after);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  out.close();

  if (code == CURLE_OK) return {};
  std::error_code ignored;
  fs::remove(destination, ignored);
  const std::string detail = error[0] != '\0' ? error.data() : curl_easy_strerror(code);
  Attempt failed{{StatusCode::DownloadFailed, url + ": " + detail}};
  failed.transient = http == 429 || http >= 500 || code == CURLE_OPERATION_TIMEDOUT ||
                     code == CURLE_COULDNT_CONNECT || code == CURLE_COULDNT_RESOLVE_HOST ||
                     code == CURLE_SEND_ERROR || code == CURLE_RECV_ERROR ||
                     code == CURLE_GOT_NOTHING || code == CURLE_PARTIAL_FILE;
  failed.retry_after = static_cast<long>(retry_after);
  return failed;
}

// Retries transient failures with exponential backoff (1, 2, 4, 8 s, or the server's Retry-After up to 60 s);
// anything else, like a 404, fails at once.
Status download(const std::string& url, const fs::path& destination) {
  constexpr int kAttempts = 5;
  for (int attempt = 1;; ++attempt) {
    Attempt result = download_once(url, destination);
    if (result.status.ok() || !result.transient || attempt == kAttempts) return result.status;
    const long wait = std::min(60L, std::max(1L << (attempt - 1), result.retry_after));
    std::this_thread::sleep_for(std::chrono::seconds(wait));
  }
}

// A name no other process or thread will pick for the same file.
std::string unique_suffix() {
  std::random_device random;
  const unsigned long long value = (static_cast<unsigned long long>(random()) << 32) ^ random();
  char hex[17];
  std::snprintf(hex, sizeof hex, "%016llx", value);
  return hex;
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

  // Download beside the target and rename, so an interrupted transfer never looks cached.
  // The name is unique per call because processes sharing a cache may fetch the same file at once.
  // The partial is removed on every exit; after a successful rename that is a no-op.
  struct PartialFile {
    fs::path path;
    ~PartialFile() {
      std::error_code ignored;
      fs::remove(path, ignored);
    }
  } partial{path + ".partial-" + unique_suffix()};
  if (Status status = download(url, partial.path); !status) return status;

  std::string digest;
  if (Status status = sha256_file(partial.path.string(), digest); !status) return status;

  if (digest != expected) {
    return {StatusCode::DigestMismatch, std::string(model) + "/" +
                std::string(repo_path) + ": expected " + std::string(expected) +
                ", got " + digest};
  }

  fs::rename(partial.path, path, code);
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
