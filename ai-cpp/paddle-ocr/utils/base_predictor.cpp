#include "base_predictor.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "./utility.h"  // Utility::IsMkldnnAvailable()

namespace {

    // Minimal logger shim to remove ilogger.h dependency.
    inline void LogPrintf(const char* level, const char* fmt, ...) {
        std::fprintf(stderr, "[%s] ", (level ? level : "I"));
        va_list args;
        va_start(args, fmt);
        std::vfprintf(stderr, fmt, args);
        va_end(args);
        std::fprintf(stderr, "\n");
    }

}  // namespace

#ifndef INFO
#define INFO(...)  LogPrintf("I", __VA_ARGS__)
#endif
#ifndef INFOW
#define INFOW(...) LogPrintf("W", __VA_ARGS__)
#endif
#ifndef INFOE
#define INFOE(...) LogPrintf("E", __VA_ARGS__)
#endif

BasePredictor::BasePredictor(const InitParams& p) {
    auto st = InitFromParams(p);
    if (!st.ok()) {
        INFOE("BasePredictor init failed: %s", st.ToString().c_str());
        std::exit(-1);
    }
}

absl::Status BasePredictor::InitFromParams(const InitParams& p) {
    model_dir_ = p.model_dir;
    if (!model_dir_.has_value() || model_dir_.value().empty()) {
        return absl::InvalidArgumentError("Model dir is empty.");
    }

    config_ = YamlConfig(model_dir_.value());

    // Read model name from config and validate if user passed model_name.
    auto model_name_config = config_.GetString("Global.model_name");
    if (!model_name_config.ok()) {
        return absl::InvalidArgumentError(model_name_config.status().ToString());
    }

    model_name_ = model_name_config.value();

    if (p.model_name.has_value() && !p.model_name.value().empty()) {
        if (model_name_ != p.model_name.value()) {
            return absl::InvalidArgumentError(
                "Model name mismatch: model_dir=" + model_dir_.value() +
                ", expected=" + p.model_name.value() +
                ", got=" + model_name_);
        }
        model_name_ = p.model_name.value();
    }

    // Build pp options
    pp_option_ptr_.reset(new PaddlePredictorOption());

    // Parse device like "gpu:0" or "cpu"
    std::string device_str = p.device.value_or(DEVICE);

    size_t pos = device_str.find(':');
    std::string device_type;
    int device_id = 0;
    if (pos != std::string::npos) {
        device_type = device_str.substr(0, pos);
        device_id = std::stoi(device_str.substr(pos + 1));
    }
    else {
        device_type = device_str;
        device_id = 0;
    }

    auto st_dev_type = pp_option_ptr_->SetDeviceType(device_type);
    if (!st_dev_type.ok()) {
        return absl::InvalidArgumentError("Failed to set device_type: " +
            st_dev_type.ToString());
    }

    auto st_dev_id = pp_option_ptr_->SetDeviceId(device_id);
    if (!st_dev_id.ok()) {
        return absl::InvalidArgumentError("Failed to set device_id: " +
            st_dev_id.ToString());
    }

    // Run mode selection (keep old behavior as much as possible).
    if (p.enable_mkldnn && device_type == "cpu") {
        if (p.precision == "fp16") {
            INFOW("MKLDNN enabled: FP16 is not supported, use FP32 instead.");
        }
        if (Utility::IsMkldnnAvailable()) {
            auto st_rm = pp_option_ptr_->SetRunMode("mkldnn");
            if (!st_rm.ok()) {
                return absl::InvalidArgumentError("Failed to set run_mode mkldnn: " +
                    st_rm.ToString());
            }
        }
        else {
            INFOW("MKLDNN is not available, fallback to paddle.");
            auto st_rm = pp_option_ptr_->SetRunMode("paddle");
            if (!st_rm.ok()) {
                return absl::InvalidArgumentError("Failed to set run_mode paddle: " +
                    st_rm.ToString());
            }
        }
    }
    else if (p.precision == "fp16") {
        auto st_rm = pp_option_ptr_->SetRunMode("paddle_fp16");
        if (!st_rm.ok()) {
            return absl::InvalidArgumentError("Failed to set run_mode paddle_fp16: " +
                st_rm.ToString());
        }
    }
    else {
        auto st_rm = pp_option_ptr_->SetRunMode("paddle");
        if (!st_rm.ok()) {
            return absl::InvalidArgumentError("Failed to set run_mode paddle: " +
                st_rm.ToString());
        }
    }

    auto st_cache = pp_option_ptr_->SetMkldnnCacheCapacity(p.mkldnn_cache_capacity);
    if (!st_cache.ok()) {
        return absl::InvalidArgumentError("Set mkldnn_cache_capacity failed: " +
            st_cache.ToString());
    }

    auto st_threads = pp_option_ptr_->SetCpuThreads(p.cpu_threads);
    if (!st_threads.ok()) {
        return absl::InvalidArgumentError("Set cpu_threads failed: " +
            st_threads.ToString());
    }

    INFO("Create model: %s", model_name_.c_str());
    return absl::OkStatus();
}

std::unique_ptr<PaddleInfer> BasePredictor::CreateStaticInfer() {
    // NOTE: model_dir_ must exist (validated in InitFromParams).
    return std::unique_ptr<PaddleInfer>(new PaddleInfer(
        model_name_, model_dir_.value(), MODEL_FILE_PREFIX, PPOption()));
}
