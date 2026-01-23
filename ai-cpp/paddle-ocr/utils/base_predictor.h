#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "absl_shim.h"

#include "./../static_infer.h"
#include "./pp_option.h"
#include "./yaml_config.h"
//#include "processors/processors.h"  // BaseProcessor + derived ops (adjust include path to your tree)

class BasePredictor {
public:
    struct InitParams {
        std::optional<std::string> model_dir = std::nullopt;
        std::optional<std::string> model_name = std::nullopt;  // optional check
        std::optional<std::string> device = std::nullopt;      // "cpu" / "gpu:0"
        std::string precision = "fp32";                        // "fp32" / "fp16"
        bool enable_mkldnn = true;
        int mkldnn_cache_capacity = 10;
        int cpu_threads = 8;
    };

public:
    explicit BasePredictor(const InitParams& p);
    virtual ~BasePredictor() = default;

    // Create paddle static infer wrapper (PaddleInfer).
    std::unique_ptr<PaddleInfer> CreateStaticInfer();

    // Accessors.
    const PaddlePredictorOption& PPOption() const { return *pp_option_ptr_; }
    const std::string& ModelName() const { return model_name_; }
    const YamlConfig& Config() const { return config_; }
    std::string ConfigPath() const { return config_.ConfigYamlPath(); }

    // Register preprocess operators.
    template <typename T, typename... Args>
    void Register(const std::string& key, Args&&... args) {
        auto instance = std::unique_ptr<T>(new T(std::forward<Args>(args)...));
        pre_op_[key] = std::move(instance);
    }

protected:
    // These are used by derived predictors (Det/Rec/Table/etc).
    std::optional<std::string> model_dir_;
    YamlConfig config_;
    std::string model_name_;

    std::unique_ptr<PaddlePredictorOption> pp_option_ptr_;
    //std::unordered_map<std::string, std::unique_ptr<BaseProcessor>> pre_op_;

    static constexpr const char* MODEL_FILE_PREFIX = "inference";

private:
    absl::Status InitFromParams(const InitParams& p);
};
