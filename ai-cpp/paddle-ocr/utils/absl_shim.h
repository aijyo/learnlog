#pragma once
// Minimal Abseil Status / StatusOr / StrCat shim.
// This header is intended to remove the external absl dependency for this project.
// It only implements the small subset used by Utility and YamlConfig.

#include <cerrno>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace absl {

enum StatusCode {
    kOk = 0,
    kCancelled = 1,
    kUnknown = 2,
    kInvalidArgument = 3,
    kDeadlineExceeded = 4,
    kNotFound = 5,
    kAlreadyExists = 6,
    kPermissionDenied = 7,
    kResourceExhausted = 8,
    kFailedPrecondition = 9,
    kAborted = 10,
    kOutOfRange = 11,
    kUnimplemented = 12,
    kInternal = 13,
    kUnavailable = 14,
    kDataLoss = 15,
    kUnauthenticated = 16
};

class Status {
 public:
  Status() : code_(StatusCode::kOk) {}
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

  std::string ToString() const {
    if (ok()) return "OK";
    std::ostringstream oss;
    oss << CodeToString(code_);
    if (!message_.empty()) oss << ": " << message_;
    return oss.str();
  }

 private:
  static const char* CodeToString(StatusCode code) {
    switch (code) {
      case StatusCode::kOk: return "OK";
      case StatusCode::kCancelled: return "CANCELLED";
      case StatusCode::kUnknown: return "UNKNOWN";
      case StatusCode::kInvalidArgument: return "INVALID_ARGUMENT";
      case StatusCode::kNotFound: return "NOT_FOUND";
      case StatusCode::kInternal: return "INTERNAL";
      case StatusCode::kUnavailable: return "UNAVAILABLE";
      default: return "UNKNOWN";
    }
  }

  StatusCode code_;
  std::string message_;
};

inline Status OkStatus() { return Status(); }

inline Status NotFoundError(std::string message) {
  return Status(StatusCode::kNotFound, std::move(message));
}

inline Status InvalidArgumentError(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

inline Status InternalError(std::string message) {
  return Status(StatusCode::kInternal, std::move(message));
}

inline Status ErrnoToStatus(int err, const std::string& context) {
  std::ostringstream oss;
  oss << context;
  if (err != 0) {
    oss << " (errno=" << err << ")";
  }
  return Status(StatusCode::kInternal, oss.str());
}

// ---- StrCat (subset) ----
namespace internal {
inline void AppendToOss(std::ostringstream&) {}

template <typename T, typename... Rest>
inline void AppendToOss(std::ostringstream& oss, T&& v, Rest&&... rest) {
  oss << std::forward<T>(v);
  AppendToOss(oss, std::forward<Rest>(rest)...);
}
}  // namespace internal

template <typename... Args>
inline std::string StrCat(Args&&... args) {
  std::ostringstream oss;
  internal::AppendToOss(oss, std::forward<Args>(args)...);
  return oss.str();
}

// ---- StatusOr (minimal) ----
template <typename T>
class StatusOr {
 public:
  StatusOr(const T& value) : status_(OkStatus()), has_value_(true), value_(value) {}
  StatusOr(T&& value) : status_(OkStatus()), has_value_(true), value_(std::move(value)) {}
  StatusOr(const Status& status) : status_(status), has_value_(false) {}
  StatusOr(Status&& status) : status_(std::move(status)), has_value_(false) {}

  bool ok() const { return status_.ok(); }
  const Status& status() const { return status_; }

  const T& value() const { return value_; }
  T& value() { return value_; }

  const T& ValueOrDie() const { return value_; }
  T& ValueOrDie() { return value_; }

  const T& operator*() const { return value_; }
  T& operator*() { return value_; }
  const T* operator->() const { return &value_; }
  T* operator->() { return &value_; }

 private:
  Status status_;
  bool has_value_ = false;
  T value_{};
};


inline Status OutOfRangeError(const std::string& msg) {
    return Status(kOutOfRange, msg);
}

}  // namespace absl
