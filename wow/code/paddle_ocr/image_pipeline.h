#pragma once
// image_pipeline.h
// ImagePipeline: run OCR pipeline on cv::Mat inputs.
// Comments are in English as requested.

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "src/pipelines/ocr/pipeline.h"
#include "src/base/base_pipeline.h"
#include "src/base/base_cv_result.h"


// Forward declarations are enough here; implementations include real headers.
class CropByPolys;

class _ImagePipeline : public BasePipeline {
public:
    explicit _ImagePipeline(const OCRPipelineParams& params);
    virtual ~_ImagePipeline() = default;
    _ImagePipeline() = delete;

    // NOTE: BasePipeline requires Predict(vector<string>) override; we keep it but
    // for ImagePipeline you should call Predict(const vector<cv::Mat>&).
    std::vector<std::unique_ptr<BaseCVResult>>
        Predict(const std::vector<std::string>& input) override;

    // New API: Mat inputs (single or batch).
    std::vector<std::unique_ptr<BaseCVResult>>
        Predict(const std::vector<cv::Mat>& input_images);

    std::vector<OCRPipelineResult> PipelineResult() const {
        return pipeline_result_vec_;
    }

    static absl::StatusOr<std::vector<cv::Mat>>
        RotateImage(const std::vector<cv::Mat>& image_array_list,
            const std::vector<int>& rotate_angle_list);

    std::unordered_map<std::string, bool> GetModelSettings() const;
    TextDetParams GetTextDetParams() const { return text_det_params_; }

    void OverrideConfig();

private:
    // Internal implementation shared by Mat inputs.
    std::vector<std::unique_ptr<BaseCVResult>>
        PredictImpl_(const std::vector<cv::Mat>& images, const std::vector<std::string>* input_paths);

private:
    OCRPipelineParams params_;
    YamlConfig config_;
    std::vector<OCRPipelineResult> pipeline_result_vec_;

    bool use_doc_preprocessor_ = false;
    bool use_doc_orientation_classify_ = false;
    bool use_doc_unwarping_ = false;
    std::unique_ptr<BasePipeline> doc_preprocessors_pipeline_;

    bool use_textline_orientation_ = false;
    std::unique_ptr<BasePredictor> textline_orientation_model_;
    std::unique_ptr<BasePredictor> text_det_model_;
    std::unique_ptr<BasePredictor> text_rec_model_;

    std::unique_ptr<CropByPolys> crop_by_polys_;
    std::function<std::vector<std::vector<cv::Point2f>>(
        const std::vector<std::vector<cv::Point2f>>&)>
        sort_boxes_;

    float text_rec_score_thresh_ = 0.0f;
    std::string text_type_;
    TextDetParams text_det_params_;
};

class ImagePipeline
    : public AutoParallelSimpleInferencePipeline<
    _ImagePipeline, OCRPipelineParams, std::vector<cv::Mat>,
    std::vector<std::unique_ptr<BaseCVResult>>> {
public:
    explicit ImagePipeline(const OCRPipelineParams& params)
        : AutoParallelSimpleInferencePipeline(params),
        thread_num_(params.thread_num) {
        if (thread_num_ == 1) {
            infer_ = std::unique_ptr<BasePipeline>(new _ImagePipeline(params));
        }
    }

    virtual std::vector<std::unique_ptr<BaseCVResult>>
        Predict(const std::vector<std::string>& input) {
        INFOE("_ImagePipeline does not support path input. Use Predict(vector<cv::Mat>).");
        return {};
    }
    std::vector<std::unique_ptr<BaseCVResult>>
        Predict(const std::vector<cv::Mat>& input);

private:
    int thread_num_;
    std::unique_ptr<BasePipeline> infer_;
};
