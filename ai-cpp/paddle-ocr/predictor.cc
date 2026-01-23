#include "predictor.h"

#include <cstdarg>
#include <cstdio>
#include <stdexcept>

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

TextRecPredictor::TextRecPredictor(const TextRecPredictorParams& params)
    : BasePredictor(InitParams{
        params.model_dir, params.model_name, params.device,
        params.precision, params.enable_mkldnn,
        params.mkldnn_cache_capacity, params.cpu_threads }),
        params_(params) {
    // If you want strict validation, keep this.
    auto st = CheckRecModelParams();
    if (!st.ok()) {
        INFOE("CheckRecModelParams failed: %s", st.ToString().c_str());
        // Keep consistent with your existing style: exit on build-time errors.
        // You can also "throw" if you prefer.
        exit(-1);
    }

    auto st_build = Build();
    if (!st_build.ok()) {
        INFOE("Build failed: %s", st_build.ToString().c_str());
        exit(-1);
    }
}

absl::Status TextRecPredictor::Build() {
    if (initialized_)
        return absl::OkStatus();
    // PreProcess ops
    // NOTE: We keep ReadImage registration for PredictPath usage.
    // For Predict(cv::Mat), we will bypass Read op and feed BGR directly.
    //Register<ReadImage>("Read", "BGR");
    //Register<OCRReisizeNormImg>("ReisizeNorm", params_.input_shape);
    //Register<ToBatchUniform>("ToBatch");

    // === Keep your required line ===
    infer_ptr_ = CreateStaticInfer();

    // PostProcess ops
    const auto& post_params = config_.PostProcessOpInfo();
    // === Keep your required snippet ===
    post_op_["CTCLabelDecode"] = std::unique_ptr<CTCLabelDecode>(
        new CTCLabelDecode(YamlConfig::SmartParseVector(
            post_params.at("PostProcess.character_dict"))
            .vec_string));

    initialized_ = true;
    return absl::OkStatus();
}

absl::StatusOr<TextRecPredictorResult> TextRecPredictor::Predict(const cv::Mat& bgr) {
    if (bgr.empty()) {
        return absl::InvalidArgumentError("Predict: input image is empty");
    }
    if (!infer_ptr_) {
        return absl::FailedPreconditionError("Predict: infer_ptr_ is null (Build not called?)");
    }
    if (post_op_.find("CTCLabelDecode") == post_op_.end()) {
        return absl::FailedPreconditionError("Predict: CTCLabelDecode is not initialized");
    }

    TextRecPredictorResult result;
    result.input_image = bgr.clone();
    result.vis_font = params_.vis_font_dir.value_or("");

    // NOTE: We keep internal vector<Mat> because existing processors work on batch containers.
    // This is NOT exposed as batch API to user, so "batch logic" is removed from interface.
    //std::vector<cv::Mat> batch_data;
    //batch_data.reserve(1);
    //batch_data.push_back(bgr);

    // 1) Resize + Normalize
    //auto st_resize = pre_op_.at("ReisizeNorm")->Apply(batch_data);
    //if (!st_resize.ok()) {
    //    return st_resize.status();
    //}

    //// 2) ToBatch (NCHW float blob)
    //auto st_tobatch = pre_op_.at("ToBatch")->Apply(st_resize.value());
    //if (!st_tobatch.ok()) {
    //    return st_tobatch.status();
    //}

    // 3) Infer
    auto st_infer = infer_ptr_->Apply(bgr);
    if (!st_infer.ok()) {
        return st_infer.status();
    }
    if (st_infer.value().empty()) {
        return absl::InternalError("Predict: infer output is empty");
    }

    // 4) CTC Decode
    auto st_ctc = post_op_.at("CTCLabelDecode")->Apply(st_infer.value()[0]);
    if (!st_ctc.ok()) {
        return st_ctc.status();
    }
    if (st_ctc.value().empty()) {
        return absl::InternalError("Predict: CTC decode output is empty");
    }

    // Single sample => take [0]
    result.rec_text = st_ctc.value()[0].first;
    result.rec_score = st_ctc.value()[0].second;
    return result;
}

absl::StatusOr<TextRecPredictorResult> TextRecPredictor::PredictPath(const std::string& image_path) {
    if (image_path.empty()) {
        return absl::InvalidArgumentError("PredictPath: image_path is empty");
    }

    // Prepare a batch container holding "path"
    // NOTE: ReadImage in your pipeline expects vector<cv::Mat> but internally may read based on input_path_ from BasePredictor.
    // In your old Process(), BasePredictor likely held input_path_ vector. Here we call Read op directly if it supports path string.
    // If your ReadImage only reads from BasePredictor::input_path_, then you should set input_path_ before calling Apply.
    //
    // Safest approach: use OpenCV imread here, bypass ReadImage operator.
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        return absl::NotFoundError("PredictPath: failed to read image: " + image_path);
    }

    auto r = Predict(bgr);
    if (!r.ok()) return r.status();
    auto out = r.value();
    out.input_path = image_path;
    return out;
}

absl::Status TextRecPredictor::CheckRecModelParams() {
    // Keep the original behavior as much as possible, but migrate absl::optional to std::optional.
    auto models_check = Utility::GetOcrModelInfo(
        params_.lang.value_or(""),
        params_.ocr_version.value_or(""));

    if (!models_check.ok()) {
        return absl::InvalidArgumentError(
            "lang and ocr_version is invalid : " + models_check.status().ToString());
    }

    auto model_name_st = ModelName();
    if (model_name_st.empty()) {
        return absl::InternalError(
            "Get model name fail : " + model_name_st);
    }

    const std::string& model_name = model_name_st;
    size_t pos_model_name = model_name.find('_');
    size_t pos_model_check = std::get<1>(models_check.value()).find('_');

    std::string prefix_model_name =
        (pos_model_name == std::string::npos) ? model_name
        : model_name.substr(0, pos_model_name);

    std::string prefix_model_check =
        (pos_model_check == std::string::npos) ? std::get<1>(models_check.value())
        : std::get<1>(models_check.value()).substr(0, pos_model_check);

    auto match_st = Utility::GetOcrModelInfo(params_.lang.value_or(""), prefix_model_name);
    if (!match_st.ok()) {
        return absl::InternalError(
            "Model and lang do not match : " + match_st.status().ToString());
    }

    if (params_.ocr_version.has_value()) {
        if (prefix_model_name != params_.ocr_version.value()) {
            INFOW("Rec model ocr_version and ocr_version params do not match");
        }
    }

#ifdef USE_FREETYPE
    if (!params_.vis_font_dir.has_value()) {
        return absl::InvalidArgumentError(
            "Visualization font path is empty, please provide " +
            std::get<2>(models_check.value()) + " path.");
    }
    else {
        size_t pos = params_.vis_font_dir.value().find_last_of("/\\");
        std::string filename = params_.vis_font_dir.value().substr(pos + 1);
        if (filename != std::get<2>(models_check.value())) {
            return absl::NotFoundError(
                "Expected visualization font is " + std::get<2>(models_check.value()) +
                ", but get is " + filename);
        }
    }
#endif

    (void)prefix_model_check; // avoid unused warnings if you later remove checks
    return absl::OkStatus();
}


// ============================
// TextDetPredictor (DB) - simplified single-image API
// ============================

TextDetPredictor::TextDetPredictor(const TextDetPredictorParams& params)
    : BasePredictor(InitParams{
        params.model_dir, params.model_name, params.device,
        params.precision, params.enable_mkldnn,
        params.mkldnn_cache_capacity, params.cpu_threads }),
        params_(params) {

    auto st_build = Build();
    if (!st_build.ok()) {
        INFOE("TextDetPredictor Build failed: %s", st_build.ToString().c_str());
        exit(-1);
    }
}

absl::Status TextDetPredictor::Build() {
    if(initialized_)
        return absl::OkStatus();
    // PreProcess ops (single-image API, but processors are vector-based internally)
    // NOTE: We do NOT register ReadImage here; PredictPath uses cv::imread directly.
    const auto& pre_tfs = config_.PreProcessOpInfo();

    //DetResizeForTestParam resize_param;
    //resize_param.input_shape = params_.input_shape;
    //resize_param.max_side_limit = params_.max_side_limit;
    //resize_param.limit_side_len = params_.limit_side_len;
    //resize_param.limit_type = params_.limit_type;

    //// Some yaml exports still contain this field; keep compatible.
    //auto it = pre_tfs.find("DetResizeForTest.resize_long");
    //if (it != pre_tfs.end()) {
    //    resize_param.resize_long = std::stoi(it->second);
    //}

    //Register<DetResizeForTest>("Resize", resize_param);
    //Register<NormalizeImage>("Normalize");
    //Register<ToCHWImage>("ToCHW");
    //Register<ToBatch>("ToBatch");

    // Infer
    infer_ptr_ = CreateStaticInfer();

    // PostProcess ops (DB)
    const auto& post_params = config_.PostProcessOpInfo();
    DBPostProcessParams db_param;

    auto GetFloat = [&](const char* key, std::optional<float> override_val) -> absl::StatusOr<float> {
        if (override_val.has_value()) return override_val.value();
        auto it2 = post_params.find(key);
        if (it2 == post_params.end()) {
            return absl::NotFoundError(std::string("Missing postprocess key: ") + key);
        }
        try {
            return std::stof(it2->second);
        }
        catch (const std::exception& e) {
            return absl::InvalidArgumentError(std::string("Invalid float for ") + key + ": " + e.what());
        }
        };

    auto st_thresh = GetFloat("PostProcess.thresh", params_.thresh);
    if (!st_thresh.ok()) return st_thresh.status();
    auto st_box_thresh = GetFloat("PostProcess.box_thresh", params_.box_thresh);
    if (!st_box_thresh.ok()) return st_box_thresh.status();
    auto st_unclip = GetFloat("PostProcess.unclip_ratio", params_.unclip_ratio);
    if (!st_unclip.ok()) return st_unclip.status();

    db_param.thresh = st_thresh.value();
    db_param.box_thresh = st_box_thresh.value();
    db_param.unclip_ratio = st_unclip.value();

    auto it_mc = post_params.find("PostProcess.max_candidates");
    if (it_mc != post_params.end()) {
        db_param.max_candidates = std::stoi(it_mc->second);
    }

    post_op_["DBPostProcess"] = std::unique_ptr<DBPostProcess>(new DBPostProcess(db_param));
    initialized_ = true;
    return absl::OkStatus();
}
absl::StatusOr<TextDetPredictorResult>
TextDetPredictor::Predict(const cv::Mat& bgr) {
    if (bgr.empty()) {
        return absl::InvalidArgumentError("TextDetPredictor::Predict: input image is empty");
    }
    if (!infer_ptr_) {
        return absl::FailedPreconditionError("TextDetPredictor::Predict: infer_ptr_ is null (Build not called?)");
    }
    //if (pre_op_.find("Resize") == pre_op_.end() ||
    //    pre_op_.find("Normalize") == pre_op_.end() ||
    //    pre_op_.find("ToCHW") == pre_op_.end() ||
    //    pre_op_.find("ToBatch") == pre_op_.end()) {
    //    return absl::FailedPreconditionError("TextDetPredictor::Predict: preprocess ops are not initialized");
    //}

    TextDetPredictorResult out;
    out.input_image = bgr.clone();

    // Keep origin shape
    // NOTE: This is the original image size BEFORE det resize.
    std::vector<int> origin_shape = { bgr.rows, bgr.cols };

    // Prepare single image container for existing processor implementations.
    std::vector<cv::Mat> imgs;
    imgs.reserve(1);
    imgs.push_back(bgr);

    // Optional "Read" stage (some pipelines keep it to unify decode/convert).
    // If you don't have it, just skip.
    //if (pre_op_.find("Read") != pre_op_.end()) {
    //    auto st_read = pre_op_.at("Read")->Apply(imgs);
    //    if (!st_read.ok()) return st_read.status();
    //    if (st_read->empty()) {
    //        return absl::InternalError("TextDetPredictor::Predict: Read output is empty");
    //    }
    //    imgs = std::move(st_read.value());
    //    origin_shape = { imgs[0].rows, imgs[0].cols };
    //}

    //// Resize stage MUST preserve DetResizeForTestParam.
    //// If your DetResizeForTest::Apply supports param_ptr, pass it here.
    //DetResizeForTestParam resize_param;  // Make sure this struct exists in your processors.h
    //{
    //    // Preferred: Resize Apply supports param_ptr -> fill resize_param
    //    // If your current signature doesn't take param_ptr, keep the old one and compute origin_shape from imgs[0].
    //    auto st_resize = pre_op_.at("Resize")->Apply(imgs, &resize_param);
    //    if (!st_resize.ok()) return st_resize.status();
    //    if (st_resize->empty()) {
    //        return absl::InternalError("TextDetPredictor::Predict: Resize output is empty");
    //    }
    //    imgs = std::move(st_resize.value());

    //    // origin_shape should represent ORIGINAL size for postprocess mapping.
    //    // If resize_param contains orig sizes, prefer it.
    //    if (resize_param.orig_h > 0 && resize_param.orig_w > 0) {
    //        origin_shape = { resize_param.orig_h, resize_param.orig_w };
    //    }
    //}

    //auto st_norm = pre_op_.at("Normalize")->Apply(imgs);
    //if (!st_norm.ok()) return st_norm.status();
    //if (st_norm->empty()) {
    //    return absl::InternalError("TextDetPredictor::Predict: Normalize output is empty");
    //}

    //auto st_chw = pre_op_.at("ToCHW")->Apply(st_norm.value());
    //if (!st_chw.ok()) return st_chw.status();
    //if (st_chw->empty()) {
    //    return absl::InternalError("TextDetPredictor::Predict: ToCHW output is empty");
    //}

    //auto st_batch = pre_op_.at("ToBatch")->Apply(st_chw.value());
    //if (!st_batch.ok()) return st_batch.status();
    //if (st_batch->empty()) {
    //    return absl::InternalError("TextDetPredictor::Predict: ToBatch output is empty");
    //}

    // Inference: infer expects the batched tensor (same as old Process()).
    auto st_infer = infer_ptr_->Apply(imgs);
    if (!st_infer.ok()) return st_infer.status();
    if (st_infer->empty()) {
        return absl::InternalError("TextDetPredictor::Predict: infer output is empty");
    }

    // Old code uses infer_result.value()[0] for DB postprocess input.
    // Keep this behavior.
    const cv::Mat& pred = st_infer.value()[0];
    if (pred.empty()) {
        return absl::InternalError("TextDetPredictor::Predict: pred map is empty");
    }

    // -----------------------------
    // Postprocess (choose ONE form)
    // -----------------------------

    // ===== Form A: DBPostProcess is stored as a processor-like object (has Apply)
    // Requirement: post_op_["DBPostProcess"] exists and its Apply takes (pred, origin_shape) or (pred, pack)
    //
    // If your current simplified DBPostProcess::Apply signature is:
    //   StatusOr<std::vector<std::pair<...>>> Apply(const cv::Mat& pred, const std::vector<int>& origin_shape)
    //
    // then:
    //
    // auto st_db = post_op_.at("DBPostProcess")->Apply(pred, origin_shape);
    // if (!st_db.ok()) return st_db.status();
    // if (st_db->empty()) return absl::InternalError("TextDetPredictor::Predict: DB output is empty");
    // out.dt_polys = st_db.value()[0].first;
    // out.dt_scores = st_db.value()[0].second;
    // return out;

    // ===== Form B (recommended): DBPostProcess is an independent class (no inheritance)
    // You keep a member like: std::unique_ptr<DBPostProcess> db_post_;
    // Then:
    if (!post_op_["DBPostProcess"]) {
        return absl::FailedPreconditionError("TextDetPredictor::Predict: db_post_ is null");
    }
    auto st_db1 = (*post_op_["DBPostProcess"])(pred, origin_shape); // StatusOr<pair<polys,scores>>
    if (!st_db1.ok()) return st_db1.status();

    out.dt_polys = std::move(st_db1.value().first);
    out.dt_scores = std::move(st_db1.value().second);
    return out;
}


absl::StatusOr<TextDetPredictorResult> TextDetPredictor::PredictPath(const std::string& image_path) {
    if (image_path.empty()) {
        return absl::InvalidArgumentError("TextDetPredictor::PredictPath: image_path is empty");
    }

    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        return absl::NotFoundError("TextDetPredictor::PredictPath: failed to read image: " + image_path);
    }

    auto r = Predict(bgr);
    if (!r.ok()) return r.status();
    auto out = r.value();
    out.input_path = image_path;
    return out;
}
