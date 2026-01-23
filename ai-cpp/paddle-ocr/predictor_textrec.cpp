//#include "predictor_textrec.h"
//
//#include <cstdarg>
//#include <cstdio>
//#include <stdexcept>
//
//namespace {
//
//    // Minimal logger shim to remove ilogger.h dependency.
//    inline void LogPrintf(const char* level, const char* fmt, ...) {
//        std::fprintf(stderr, "[%s] ", (level ? level : "I"));
//        va_list args;
//        va_start(args, fmt);
//        std::vfprintf(stderr, fmt, args);
//        va_end(args);
//        std::fprintf(stderr, "\n");
//    }
//
//}  // namespace
//
//#ifndef INFO
//#define INFO(...)  LogPrintf("I", __VA_ARGS__)
//#endif
//#ifndef INFOW
//#define INFOW(...) LogPrintf("W", __VA_ARGS__)
//#endif
//#ifndef INFOE
//#define INFOE(...) LogPrintf("E", __VA_ARGS__)
//#endif
//
//TextRecPredictor::TextRecPredictor(const TextRecPredictorParams& params)
//    : BasePredictor(InitParams{
//        params.model_dir, params.model_name, params.device,
//        params.precision, params.enable_mkldnn,
//        params.mkldnn_cache_capacity, params.cpu_threads }),
//    params_(params) {
//    // If you want strict validation, keep this.
//    auto st = CheckRecModelParams();
//    if (!st.ok()) {
//        INFOE("CheckRecModelParams failed: %s", st.ToString().c_str());
//        // Keep consistent with your existing style: exit on build-time errors.
//        // You can also "throw" if you prefer.
//        exit(-1);
//    }
//
//    auto st_build = Build();
//    if (!st_build.ok()) {
//        INFOE("Build failed: %s", st_build.ToString().c_str());
//        exit(-1);
//    }
//}
//
//absl::Status TextRecPredictor::Build() {
//    // PreProcess ops
//    // NOTE: We keep ReadImage registration for PredictPath usage.
//    // For Predict(cv::Mat), we will bypass Read op and feed BGR directly.
//    //Register<ReadImage>("Read", "BGR");
//    //Register<OCRReisizeNormImg>("ReisizeNorm", params_.input_shape);
//    //Register<ToBatchUniform>("ToBatch");
//
//    // === Keep your required line ===
//    infer_ptr_ = CreateStaticInfer();
//
//    // PostProcess ops
//    const auto& post_params = config_.PostProcessOpInfo();
//    // === Keep your required snippet ===
//    post_op_["CTCLabelDecode"] = std::unique_ptr<CTCLabelDecode>(
//        new CTCLabelDecode(YamlConfig::SmartParseVector(
//            post_params.at("PostProcess.character_dict"))
//            .vec_string));
//
//    return absl::OkStatus();
//}
//
//absl::StatusOr<TextRecPredictorResult> TextRecPredictor::Predict(const cv::Mat& bgr) {
//    if (bgr.empty()) {
//        return absl::InvalidArgumentError("Predict: input image is empty");
//    }
//    if (!infer_ptr_) {
//        return absl::FailedPreconditionError("Predict: infer_ptr_ is null (Build not called?)");
//    }
//    if (post_op_.find("CTCLabelDecode") == post_op_.end()) {
//        return absl::FailedPreconditionError("Predict: CTCLabelDecode is not initialized");
//    }
//
//    TextRecPredictorResult result;
//    result.input_image = bgr.clone();
//    result.vis_font = params_.vis_font_dir.value_or("");
//
//    // NOTE: We keep internal vector<Mat> because existing processors work on batch containers.
//    // This is NOT exposed as batch API to user, so "batch logic" is removed from interface.
//    //std::vector<cv::Mat> batch_data;
//    //batch_data.reserve(1);
//    //batch_data.push_back(bgr);
//
//    // 1) Resize + Normalize
//    //auto st_resize = pre_op_.at("ReisizeNorm")->Apply(batch_data);
//    //if (!st_resize.ok()) {
//    //    return st_resize.status();
//    //}
//
//    //// 2) ToBatch (NCHW float blob)
//    //auto st_tobatch = pre_op_.at("ToBatch")->Apply(st_resize.value());
//    //if (!st_tobatch.ok()) {
//    //    return st_tobatch.status();
//    //}
//
//    // 3) Infer
//    auto st_infer = infer_ptr_->Apply(bgr);
//    if (!st_infer.ok()) {
//        return st_infer.status();
//    }
//    if (st_infer.value().empty()) {
//        return absl::InternalError("Predict: infer output is empty");
//    }
//
//    // 4) CTC Decode
//    auto st_ctc = post_op_.at("CTCLabelDecode")->Apply(st_infer.value()[0]);
//    if (!st_ctc.ok()) {
//        return st_ctc.status();
//    }
//    if (st_ctc.value().empty()) {
//        return absl::InternalError("Predict: CTC decode output is empty");
//    }
//
//    // Single sample => take [0]
//    result.rec_text = st_ctc.value()[0].first;
//    result.rec_score = st_ctc.value()[0].second;
//    return result;
//}
//
//absl::StatusOr<TextRecPredictorResult> TextRecPredictor::PredictPath(const std::string& image_path) {
//    if (image_path.empty()) {
//        return absl::InvalidArgumentError("PredictPath: image_path is empty");
//    }
//
//    // Prepare a batch container holding "path"
//    // NOTE: ReadImage in your pipeline expects vector<cv::Mat> but internally may read based on input_path_ from BasePredictor.
//    // In your old Process(), BasePredictor likely held input_path_ vector. Here we call Read op directly if it supports path string.
//    // If your ReadImage only reads from BasePredictor::input_path_, then you should set input_path_ before calling Apply.
//    //
//    // Safest approach: use OpenCV imread here, bypass ReadImage operator.
//    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
//    if (bgr.empty()) {
//        return absl::NotFoundError("PredictPath: failed to read image: " + image_path);
//    }
//
//    auto r = Predict(bgr);
//    if (!r.ok()) return r.status();
//    auto out = r.value();
//    out.input_path = image_path;
//    return out;
//}
//
//absl::Status TextRecPredictor::CheckRecModelParams() {
//    // Keep the original behavior as much as possible, but migrate absl::optional to std::optional.
//    auto models_check = Utility::GetOcrModelInfo(
//        params_.lang.value_or(""),
//        params_.ocr_version.value_or(""));
//
//    if (!models_check.ok()) {
//        return absl::InvalidArgumentError(
//            "lang and ocr_version is invalid : " + models_check.status().ToString());
//    }
//
//    auto model_name_st = ModelName();
//    if (!model_name_st.empty()) {
//        return absl::InternalError(
//            "Get model name fail : " + model_name_st);
//    }
//
//    const std::string& model_name = model_name_st;
//    size_t pos_model_name = model_name.find('_');
//    size_t pos_model_check = std::get<1>(models_check.value()).find('_');
//
//    std::string prefix_model_name =
//        (pos_model_name == std::string::npos) ? model_name
//        : model_name.substr(0, pos_model_name);
//
//    std::string prefix_model_check =
//        (pos_model_check == std::string::npos) ? std::get<1>(models_check.value())
//        : std::get<1>(models_check.value()).substr(0, pos_model_check);
//
//    auto match_st = Utility::GetOcrModelInfo(params_.lang.value_or(""), prefix_model_name);
//    if (!match_st.ok()) {
//        return absl::InternalError(
//            "Model and lang do not match : " + match_st.status().ToString());
//    }
//
//    if (params_.ocr_version.has_value()) {
//        if (prefix_model_name != params_.ocr_version.value()) {
//            INFOW("Rec model ocr_version and ocr_version params do not match");
//        }
//    }
//
//#ifdef USE_FREETYPE
//    if (!params_.vis_font_dir.has_value()) {
//        return absl::InvalidArgumentError(
//            "Visualization font path is empty, please provide " +
//            std::get<2>(models_check.value()) + " path.");
//    }
//    else {
//        size_t pos = params_.vis_font_dir.value().find_last_of("/\\");
//        std::string filename = params_.vis_font_dir.value().substr(pos + 1);
//        if (filename != std::get<2>(models_check.value())) {
//            return absl::NotFoundError(
//                "Expected visualization font is " + std::get<2>(models_check.value()) +
//                ", but get is " + filename);
//        }
//    }
//#endif
//
//    (void)prefix_model_check; // avoid unused warnings if you later remove checks
//    return absl::OkStatus();
//}
