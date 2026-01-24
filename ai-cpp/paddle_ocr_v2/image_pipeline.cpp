// image_pipeline.cc
// ImagePipeline: run OCR pipeline on cv::Mat inputs.
// Comments are in English as requested.

#include "image_pipeline.h"

#include <algorithm>
#include <array>
#include <utility>

#include "src/pipelines/ocr/result.h"
#include "src/utils/args.h"
#include "src/utils/ilogger.h"
#include "src/utils/utility.h"

#include "src/common/processors.h"
#include "src/modules/image_classification/predictor.h"
#include "src/modules/text_detection/predictor.h"
#include "src/modules/text_recognition/predictor.h"
#include "src/pipelines/doc_preprocessor/pipeline.h"

_ImagePipeline::_ImagePipeline(const OCRPipelineParams& params)
    : BasePipeline(), params_(params) {
    // Same config loading as _OCRPipeline
    if (params.paddlex_config.has_value()) {
        if (params.paddlex_config.value().IsStr()) {
            config_ = YamlConfig(params.paddlex_config.value().GetStr());
        }
        else {
            config_ = YamlConfig(params.paddlex_config.value().GetMap());
        }
    }
    else {
        auto config_path = Utility::GetDefaultConfig("OCR");
        if (!config_path.ok()) {
            INFOE("Could not find OCR pipeline config file: %s",
                config_path.status().ToString().c_str());
            exit(-1);
        }
        config_ = YamlConfig(config_path.value());
    }

    OverrideConfig();

    // Decide doc preprocessor usage (same logic)
    auto result_use_doc_orientation_classify =
        config_.GetBool("use_doc_orientation_classify", true);
    if (!result_use_doc_orientation_classify.ok()) {
        INFOE("use_doc_orientation_classify config error : %s",
            result_use_doc_orientation_classify.status().ToString().c_str());
        exit(-1);
    }
    auto result_use_use_doc_unwarping =
        config_.GetBool("use_doc_unwarping", true);
    if (!result_use_use_doc_unwarping.ok()) {
        INFOE("use_doc_unwarping config error : %s",
            result_use_use_doc_unwarping.status().ToString().c_str());
        exit(-1);
    }

    if (result_use_doc_orientation_classify.value() ||
        result_use_use_doc_unwarping.value()) {
        use_doc_preprocessor_ = true;
    }
    else {
        use_doc_preprocessor_ = false;
    }

    if (use_doc_preprocessor_) {
        auto result_doc_preprocessor_config = config_.GetSubModule("SubPipelines");
        if (!result_doc_preprocessor_config.ok()) {
            INFOE("Get doc preprocessors subpipelines config fail : %s",
                result_doc_preprocessor_config.status().ToString().c_str());
            exit(-1);
        }
        DocPreprocessorPipelineParams p;
        p.device = params_.device;
        p.precision = params_.precision;
        p.enable_mkldnn = params_.enable_mkldnn;
        p.mkldnn_cache_capacity = params_.mkldnn_cache_capacity;
        p.cpu_threads = params_.cpu_threads;
        p.paddlex_config = result_doc_preprocessor_config.value();
        doc_preprocessors_pipeline_ = CreatePipeline<_DocPreprocessorPipeline>(p);

        use_doc_orientation_classify_ =
            config_.GetBool("DocPreprocessor.use_doc_orientation_classify", true)
            .value();
        use_doc_unwarping_ =
            config_.GetBool("DocPreprocessor.use_doc_unwarping", true).value();
    }

    // Textline orientation
    auto result_use_textline_orientation =
        config_.GetBool("use_textline_orientation", true);
    if (!result_use_textline_orientation.ok()) {
        INFOE("use_textline_orientation config error : %s",
            result_use_textline_orientation.status().ToString().c_str());
        exit(-1);
    }
    use_textline_orientation_ = result_use_textline_orientation.value();
    if (use_textline_orientation_) {
        ClasPredictorParams p;
        p.device = params_.device;
        p.precision = params_.precision;
        p.enable_mkldnn = params_.enable_mkldnn;
        p.mkldnn_cache_capacity = params_.mkldnn_cache_capacity;
        p.cpu_threads = params_.cpu_threads;

        auto result_batch_size =
            config_.GetInt("TextLineOrientation.batch_size", 1);
        if (!result_batch_size.ok()) {
            INFOE("Get TextLineOrientation batch size fail: %s",
                result_batch_size.status().ToString().c_str());
            exit(-1);
        }
        p.batch_size = result_batch_size.value();

        auto result_model_name =
            config_.GetString("TextLineOrientation.model_name");
        if (!result_model_name.ok()) {
            INFOE("Could not find TextLineOrientation model name : %s",
                result_model_name.status().ToString().c_str());
            exit(-1);
        }
        p.model_name = result_model_name.value();

        auto result_model_dir = config_.GetString("TextLineOrientation.model_dir");
        if (!result_model_dir.ok()) {
            INFOE("Could not find TextLineOrientation model dir : %s",
                result_model_dir.status().ToString().c_str());
            exit(-1);
        }
        p.model_dir = result_model_dir.value();

        textline_orientation_model_ = CreateModule<ClasPredictor>(p);
    }

    // Text type
    auto text_type = config_.GetString("text_type");
    if (!text_type.ok()) {
        INFOE("Get text type fail : %s", text_type.status().ToString().c_str());
        exit(-1);
    }
    text_type_ = text_type.value();

    // Detection params
    TextDetPredictorParams det;
    auto result_text_det_model_name =
        config_.GetString("TextDetection.model_name");
    if (!result_text_det_model_name.ok()) {
        INFOE("Could not find TextDetection model name : %s",
            result_text_det_model_name.status().ToString().c_str());
        exit(-1);
    }
    det.model_name = result_text_det_model_name.value();

    auto result_text_det_model_dir = config_.GetString("TextDetection.model_dir");
    if (!result_text_det_model_dir.ok()) {
        INFOE("Could not find TextDetection model dir : %s",
            result_text_det_model_dir.status().ToString().c_str());
        exit(-1);
    }
    det.model_dir = result_text_det_model_dir.value();

    auto result_det_input_shape = config_.GetString("TextDetection.input_shape");
    if (!result_det_input_shape.value().empty()) {
        det.input_shape =
            config_.SmartParseVector(result_det_input_shape.value()).vec_int;
    }

    det.device = params_.device;
    det.precision = params_.precision;
    det.enable_mkldnn = params_.enable_mkldnn;
    det.mkldnn_cache_capacity = params_.mkldnn_cache_capacity;
    det.cpu_threads = params_.cpu_threads;
    det.batch_size = config_.GetInt("TextDetection.batch_size", 1).value();

    if (text_type_ == "general") {
        det.limit_side_len =
            config_.GetInt("TextDetection.limit_side_len", 960).value();
        det.limit_type =
            config_.GetString("TextDetection.limit_type", "max").value();
        det.max_side_limit =
            config_.GetInt("TextDetection.max_side_limit", 4000).value();
        det.thresh = config_.GetFloat("TextDetection.thresh", 0.3).value();
        det.box_thresh = config_.GetFloat("TextDetection.box_thresh", 0.6).value();
        det.unclip_ratio =
            config_.GetFloat("TextDetection.unclip_ratio", 2.0).value();
        sort_boxes_ = ComponentsProcessor::SortQuadBoxes;
        crop_by_polys_ = std::unique_ptr<CropByPolys>(new CropByPolys("quad"));
    }
    else if (text_type_ == "seal") {
        det.limit_side_len =
            config_.GetInt("TextDetection.limit_side_len", 736).value();
        det.limit_type =
            config_.GetString("TextDetection.limit_type", "min").value();
        det.max_side_limit =
            config_.GetInt("TextDetection.max_side_limit", 4000).value();
        det.thresh = config_.GetFloat("TextDetection.thresh", 0.2).value();
        det.box_thresh = config_.GetFloat("TextDetection.box_thresh", 0.6).value();
        det.unclip_ratio =
            config_.GetFloat("TextDetection.unclip_ratio", 0.5).value();
        sort_boxes_ = ComponentsProcessor::SortPolyBoxes;
        crop_by_polys_ = std::unique_ptr<CropByPolys>(new CropByPolys("poly"));
    }
    else {
        INFOE("Unsupported text type: %s", text_type_.c_str());
        exit(-1);
    }

    text_det_model_ = CreateModule<TextDetPredictor>(det);

    text_det_params_.text_det_limit_side_len = det.limit_side_len.value();
    text_det_params_.text_det_limit_type = det.limit_type.value();
    text_det_params_.text_det_max_side_limit = det.max_side_limit.value();
    text_det_params_.text_det_thresh = det.thresh.value();
    text_det_params_.text_det_box_thresh = det.box_thresh.value();
    text_det_params_.text_det_unclip_ratio = det.unclip_ratio.value();

    // Recognition params
    TextRecPredictorParams rec;
    auto result_text_rec_model_name =
        config_.GetString("TextRecognition.model_name");
    if (!result_text_rec_model_name.ok()) {
        INFOE("Could not find TextRecognition model name : %s",
            result_text_rec_model_name.status().ToString().c_str());
        exit(-1);
    }
    rec.model_name = result_text_rec_model_name.value();

    auto result_text_rec_model_dir =
        config_.GetString("TextRecognition.model_dir");
    if (!result_text_rec_model_dir.ok()) {
        INFOE("Could not find TextRecognition model dir : %s",
            result_text_rec_model_dir.status().ToString().c_str());
        exit(-1);
    }

    auto result_rec_input_shape =
        config_.GetString("TextRecognition.input_shape");
    if (!result_rec_input_shape.value().empty()) {
        rec.input_shape =
            config_.SmartParseVector(result_rec_input_shape.value()).vec_int;
    }

    rec.model_dir = result_text_rec_model_dir.value();
    rec.lang = params_.lang;
    rec.ocr_version = params_.ocr_version;
    rec.vis_font_dir = params_.vis_font_dir;
    rec.device = params_.device;
    rec.precision = params_.precision;
    rec.enable_mkldnn = params_.enable_mkldnn;
    rec.mkldnn_cache_capacity = params_.mkldnn_cache_capacity;
    rec.cpu_threads = params_.cpu_threads;
    rec.batch_size = config_.GetInt("TextRecognition.batch_size", 1).value();

    text_rec_model_ = CreateModule<TextRecPredictor>(rec);
    text_rec_score_thresh_ =
        config_.GetFloat("TextRecognition.score_thresh", 0.0).value();
}

absl::StatusOr<std::vector<cv::Mat>>
_ImagePipeline::RotateImage(const std::vector<cv::Mat>& image_array_list,
    const std::vector<int>& rotate_angle_list) {
    if (image_array_list.size() != rotate_angle_list.size()) {
        return absl::InvalidArgumentError(
            "Length of image_array_list (" +
            std::to_string(image_array_list.size()) +
            ") must match length of rotate_angle_list (" +
            std::to_string(rotate_angle_list.size()) + ")");
    }
    std::vector<cv::Mat> rotated_images;
    rotated_images.reserve(image_array_list.size());
    for (std::size_t i = 0; i < image_array_list.size(); ++i) {
        int angle_indicator = rotate_angle_list[i];
        if (angle_indicator != 0 && angle_indicator != 1) {
            return absl::InvalidArgumentError(
                "rotate_angle must be 0 or 1, now it's: " +
                std::to_string(angle_indicator));
        }
        int rotate_angle = angle_indicator * 180;
        auto result_rotated_image =
            ComponentsProcessor::RotateImage(image_array_list[i], rotate_angle);
        if (!result_rotated_image.ok()) {
            return result_rotated_image.status();
        }
        rotated_images.push_back(result_rotated_image.value());
    }
    return rotated_images;
}

std::unordered_map<std::string, bool> _ImagePipeline::GetModelSettings() const {
    std::unordered_map<std::string, bool> model_settings = {};
    model_settings["use_doc_preprocessor"] = use_doc_preprocessor_;
    model_settings["use_textline_orientation"] = use_textline_orientation_;
    return model_settings;
}

std::vector<std::unique_ptr<BaseCVResult>>
_ImagePipeline::Predict(const std::vector<std::string>& input) {
    // This pipeline is designed for Mat inputs. We keep this method only to satisfy BasePipeline.
    // You can implement path->imread if needed, but we intentionally block usage here.
    INFOE("_ImagePipeline::Predict(vector<string>) is not supported. Use Predict(vector<cv::Mat>).");
    return {};
}

std::vector<std::unique_ptr<BaseCVResult>>
_ImagePipeline::Predict(const std::vector<cv::Mat>& input_images) {
    return PredictImpl_(input_images, nullptr);
}

std::vector<std::unique_ptr<BaseCVResult>>
_ImagePipeline::PredictImpl_(const std::vector<cv::Mat>& images,
    const std::vector<std::string>* input_paths) {
    auto model_settings = GetModelSettings();

    pipeline_result_vec_.clear();
    std::vector<std::unique_ptr<BaseCVResult>> base_results;
    base_results.reserve(images.size());

    // Process each image like "batch size = 1"
    for (size_t i = 0; i < images.size(); ++i) {
        const cv::Mat& in = images[i];
        if (in.empty()) {
            INFOW("Input image %d is empty, skip.", (int)i);
            OCRPipelineResult res;
            if (input_paths && i < input_paths->size()) res.input_path = (*input_paths)[i];
            res.model_settings = model_settings;
            res.text_det_params = text_det_params_;
            res.text_type = text_type_;
            res.text_rec_score_thresh = text_rec_score_thresh_;
            pipeline_result_vec_.push_back(res);
            base_results.push_back(std::unique_ptr<BaseCVResult>(new OCRResult(res)));
            continue;
        }

        std::vector<cv::Mat> origin_image = { in.clone() };

        // -------- Doc preprocessor --------
        std::vector<DocPreprocessorPipelineResult> doc_pre_results;
        if (use_doc_preprocessor_) {
            // IMPORTANT: current DocPreprocessor pipeline expects file paths, not cv::Mat.
            // We fallback to identity here to keep Mat pipeline workable.
            INFOW("DocPreprocessor is enabled in config, but Mat input is not supported. Fallback to identity.");
            DocPreprocessorPipelineResult r;
            r.output_image = in.clone();
            doc_pre_results.push_back(r);
        }
        else {
            DocPreprocessorPipelineResult r;
            r.output_image = in.clone();
            doc_pre_results.push_back(r);
        }

        std::vector<cv::Mat> pre_images;
        std::vector<cv::Mat> pre_images_copy;
        pre_images.reserve(doc_pre_results.size());
        pre_images_copy.reserve(doc_pre_results.size());
        for (auto& item : doc_pre_results) {
            pre_images.push_back(item.output_image);
            pre_images_copy.push_back(item.output_image.clone());
        }

        // -------- Detection --------
        text_det_model_->Predict(pre_images_copy);
        std::vector<TextDetPredictorResult> det_results =
            static_cast<TextDetPredictor*>(text_det_model_.get())->PredictorResult();

        std::vector<std::vector<std::vector<cv::Point2f>>> dt_polys_list;
        dt_polys_list.reserve(det_results.size());
        for (auto& item : det_results) {
            if (!item.dt_polys.empty()) {
                dt_polys_list.push_back(sort_boxes_(item.dt_polys));
            }
            else {
                dt_polys_list.push_back(std::vector<std::vector<cv::Point2f>>{});
            }
        }

        // Build OCRPipelineResult shell
        OCRPipelineResult res;
        if (input_paths && i < input_paths->size()) res.input_path = (*input_paths)[i];
        res.doc_preprocessor_res = doc_pre_results[0];
        res.dt_polys = dt_polys_list.empty() ? std::vector<std::vector<cv::Point2f>>{} : dt_polys_list[0];
        res.model_settings = model_settings;
        res.text_det_params = text_det_params_;
        res.text_type = text_type_;
        res.text_rec_score_thresh = text_rec_score_thresh_;

        // If no boxes, finalize empty result.
        if (dt_polys_list.empty() || dt_polys_list[0].empty()) {
            if (text_type_ == "general") {
                res.rec_boxes = ComponentsProcessor::ConvertPointsToBoxes(res.rec_polys);
            }
            pipeline_result_vec_.push_back(res);
            base_results.push_back(std::unique_ptr<BaseCVResult>(new OCRResult(res)));
            continue;
        }

        // -------- Crop polys -> sub images --------
        auto crop_status = (*crop_by_polys_)(pre_images[0], dt_polys_list[0]);
        if (!crop_status.ok()) {
            INFOE("Split image fail : %s", crop_status.status().ToString().c_str());
            exit(-1);
        }
        std::vector<cv::Mat> all_subs_of_img = crop_status.value();
        std::vector<cv::Mat> all_subs_of_img_copy;
        all_subs_of_img_copy.reserve(all_subs_of_img.size());
        for (auto& m : all_subs_of_img) all_subs_of_img_copy.push_back(m.clone());

        // -------- Textline orientation --------
        std::vector<int> angles;
        if (model_settings["use_textline_orientation"]) {
            textline_orientation_model_->Predict(all_subs_of_img_copy);
            auto angle_results =
                static_cast<ClasPredictor*>(textline_orientation_model_.get())->PredictorResult();
            angles.reserve(angle_results.size());
            for (auto& r : angle_results) angles.push_back(r.class_ids[0]);

            auto rotated_status = RotateImage(all_subs_of_img, angles);
            if (!rotated_status.ok()) {
                INFOE("Rotate images fail : %s", rotated_status.status().ToString().c_str());
                exit(-1);
            }
            all_subs_of_img = rotated_status.value();
        }
        else {
            angles = std::vector<int>(all_subs_of_img.size(), -1);
        }
        res.textline_orientation_angles = angles;

        // -------- Sort by aspect ratio (same as _OCRPipeline) --------
        std::vector<std::pair<std::pair<int, float>, TextRecPredictorResult>> sub_img_info_list;
        sub_img_info_list.reserve(all_subs_of_img.size());
        for (int m = 0; m < (int)all_subs_of_img.size(); ++m) {
            float ratio = (float)all_subs_of_img[m].size[1] / (float)all_subs_of_img[m].size[0];
            TextRecPredictorResult empty_rec;
            sub_img_info_list.push_back({ {m, ratio}, empty_rec });
        }

        std::vector<std::pair<int, float>> sorted_subs_info;
        sorted_subs_info.reserve(sub_img_info_list.size());
        for (auto& item : sub_img_info_list) sorted_subs_info.push_back(item.first);

        std::sort(sorted_subs_info.begin(), sorted_subs_info.end(),
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                return a.second < b.second;
            });

        std::vector<cv::Mat> sorted_subs_of_img;
        sorted_subs_of_img.reserve(sorted_subs_info.size());
        for (auto& item : sorted_subs_info) sorted_subs_of_img.push_back(all_subs_of_img[item.first]);

        // -------- Recognition --------
        text_rec_model_->Predict(sorted_subs_of_img);
        auto rec_results =
            static_cast<TextRecPredictor*>(text_rec_model_.get())->PredictorResult();

        for (int m = 0; m < (int)rec_results.size(); ++m) {
            int sub_img_id = sorted_subs_info[m].first;
            sub_img_info_list[sub_img_id].second = rec_results[m];
        }

        // -------- Collect outputs --------
        for (int sno = 0; sno < (int)sub_img_info_list.size(); ++sno) {
            auto rec_res = sub_img_info_list[sno].second;
            if (rec_res.rec_score >= text_rec_score_thresh_) {
                res.rec_texts.push_back(rec_res.rec_text);
                res.rec_scores.push_back(rec_res.rec_score);
                res.rec_polys.push_back(dt_polys_list[0][sno]);
                res.vis_fonts = rec_res.vis_font;
            }
        }

        if (text_type_ == "general") {
            res.rec_boxes = ComponentsProcessor::ConvertPointsToBoxes(res.rec_polys);
        }

        pipeline_result_vec_.push_back(res);
        base_results.push_back(std::unique_ptr<BaseCVResult>(new OCRResult(res)));
    }

    return base_results;
}

std::vector<std::unique_ptr<BaseCVResult>>
ImagePipeline::Predict(const std::vector<cv::Mat>& input) {
    if (thread_num_ == 1) {
        // Downcast to our impl that supports Mat
        return static_cast<_ImagePipeline*>(infer_.get())->Predict(input);
    }

    // Parallel mode: reuse base's threading helper (same pattern as OCRPipeline).
    // We keep it minimal: split by thread_num_ and push to PredictThread.
    int input_num = (int)input.size();
    if (thread_num_ > input_num) {
        INFOW("thread num exceed input num, will set %d", input_num);
        thread_num_ = input_num;
    }
    int infer_batch_num = input_num / thread_num_;
    if (infer_batch_num <= 0) infer_batch_num = 1;

    // Build batches (vector<vector<cv::Mat>>)
    std::vector<std::vector<cv::Mat>> batches;
    batches.reserve(thread_num_);
    for (int i = 0; i < input_num; i += infer_batch_num) {
        int end = std::min(i + infer_batch_num, input_num);
        std::vector<cv::Mat> b;
        b.reserve(end - i);
        for (int j = i; j < end; ++j) b.push_back(input[j]);
        batches.push_back(std::move(b));
    }

    std::vector<std::unique_ptr<BaseCVResult>> results;
    results.reserve(input.size());

    for (auto& b : batches) {
        auto st = AutoParallelSimpleInferencePipeline::PredictThread(b);
        if (!st.ok()) {
            INFOE("Infer fail : %s", st.ToString().c_str());
            exit(-1);
        }
    }
    for (size_t i = 0; i < batches.size(); ++i) {
        auto r = GetResult();
        if (!r.ok()) {
            INFOE("Get infer result fail : %s", r.status().ToString().c_str());
            exit(-1);
        }
        results.insert(results.end(),
            std::make_move_iterator(r.value().begin()),
            std::make_move_iterator(r.value().end()));
    }
    return results;
}
//
//void _ImagePipeline::OverrideConfig() {
//    // Copy your existing _OCRPipeline::OverrideConfig() logic directly.
//    // Keeping same keys guarantees identical behavior.
//    auto& data = config_.Data();
//
//    if (params_.doc_orientation_classify_model_name.has_value()) {
//        auto it = config_.FindKey("DocOrientationClassify.model_name");
//        if (!it.ok()) {
//            data["SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_name"] =
//                params_.doc_orientation_classify_model_name.value();
//        }
//        else {
//            auto key = it.value().first;
//            data.erase(data.find(key));
//            data[key] = params_.doc_orientation_classify_model_name.value();
//        }
//    }
//
//    if (params_.doc_orientation_classify_model_dir.has_value()) {
//        auto it = config_.FindKey("DocOrientationClassify.model_dir");
//        if (!it.ok()) {
//            data["SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_dir"] =
//                params_.doc_orientation_classify_model_dir.value();
//        }
//        else {
//            auto key = it.value().first;
//            data.erase(data.find(key));
//            data[key] = params_.doc_orientation_classify_model_dir.value();
//        }
//    }
//
//    if (params_.doc_unwarping_model_name.has_value()) {
//        auto it = config_.FindKey("DocUnwarping.model_name");
//        if (!it.ok()) {
//            data["SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_name"] =
//                params_.doc_unwarping_model_name.value();
//        }
//        else {
//            auto key = it.value().first;
//            data.erase(data.find(key));
//            data[key] = params_.doc_unwarping_model_name.value();
//        }
//    }
//
//    if (params_.doc_unwarping_model_dir.has_value()) {
//        auto it = config_.FindKey("DocUnwarping.model_dir");
//        if (!it.ok()) {
//            data["SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_dir"] =
//                params_.doc_unwarping_model_dir.value();
//        }
//        else {
//            auto key = it.value().first;
//            data.erase(data.find(key));
//            data[key] = params_.doc_unwarping_model_dir.value();
//        }
//    }
//
//    if (params_.text_detection_model_name.has_value()) {
//        auto it = config_.FindKey("TextDetection.model_name");
//        if (!it.ok()) {
//            data["Modules.TextDetection.model_name"] =
//                params_.text_detection_model_name.value();
//        }
//        else {
//            auto key = it.value().first;
//            data.erase(data.find(key));
//            data[key] = params_.text_detection_model_name.value();
//        }
//    }
//
//    if (params_.text_detection_model_dir.has_value()) {
//        auto it = config_.FindKey("TextDetection.model_dir");
//        if (!it.ok()) {
//            data["Modules.TextDetection.model_dir"] =
//                params_.text_detection_model_dir.value();
//        }
//        else {
//            auto key = it.value().first;
//            data.erase(data.find(key));
//            data[key] = params_.text_detection_model_dir.value();
//        }
//    }
//
//    if (params_.textline_orientation_model_name.has_value()) {
//        auto it = config_.FindKey("TextLineOrientation.model_name");
//        if (!it.ok()) {
//            data["Modules.TextLineOrientation.model_name"] =
//                params_.textline_orientation_model_name.value();
//        }
//        else {
//            auto key = it.value().first;
//            data.erase(data.find(key));
//            data[key] = params_.textline_orientation_model_name.value();
//        }
//    }
//
//    if (params_.textline_orientation_model_dir.has_value()) {
//        auto it = config_.FindKey("TextLineOrientation.model_dir");
//        if (!it.ok()) {
//            data["Modules.TextLineOrientation.model_dir"] =
//                params_.textline_orientation_model_dir.value();
//        }
//        else {
//            auto key = it.value().first;
//            data.erase(data.find(key));
//            data[key] = params_.textline_orientation_model_dir.value();
//        }
//    }
//
//    if (params_.text_recognition_model_name.has_value()) {
//        auto it = config_.FindKey("TextRecognition.model_name");
//        if (!it.ok()) {
//            data["Modules.TextRecognition.model_name"] =
//                params_.text_recognition_model_name.value();
//        }
//        else {
//            auto key = it.value().first;
//            data.erase(data.find(key));
//            data[key] = params_.text_recognition_model_name.value();
//        }
//    }
//
//    if (params_.text_recognition_model_dir.has_value()) {
//        auto it = config_.FindKey("TextRecognition.model_dir");
//        if (!it.ok()) {
//            data["Modules.TextRecognition.model_dir"] =
//                params_.text_recognition_model_dir.value();
//        }
//        else {
//            auto key = it.value().first;
//            data.erase(data.find(key));
//            data[key] = params_.text_recognition_model_dir.value();
//        }
//    }
//
//    if (params_.use_doc_orientation_classify.has_value()) {
//        data["Global.use_doc_orientation_classify"] =
//            params_.use_doc_orientation_classify.value();
//    }
//    if (params_.use_doc_unwarping.has_value()) {
//        data["Global.use_doc_unwarping"] = params_.use_doc_unwarping.value();
//    }
//    if (params_.use_textline_orientation.has_value()) {
//        data["Global.use_textline_orientation"] =
//            params_.use_textline_orientation.value();
//    }
//
//    if (params_.text_det_limit_side_len.has_value()) {
//        data["Modules.TextDetection.limit_side_len"] =
//            params_.text_det_limit_side_len.value();
//    }
//    if (params_.text_det_limit_type.has_value()) {
//        data["Modules.TextDetection.limit_type"] =
//            params_.text_det_limit_type.value();
//    }
//    if (params_.text_det_thresh.has_value()) {
//        data["Modules.TextDetection.thresh"] = params_.text_det_thresh.value();
//    }
//    if (params_.text_det_box_thresh.has_value()) {
//        data["Modules.TextDetection.box_thresh"] =
//            params_.text_det_box_thresh.value();
//    }
//    if (params_.text_det_unclip_ratio.has_value()) {
//        data["Modules.TextDetection.unclip_ratio"] =
//            params_.text_det_unclip_ratio.value();
//    }
//    if (params_.text_det_input_shape.has_value()) {
//        data["Modules.TextDetection.input_shape"] =
//            Utility::VecToString(params_.text_det_input_shape.value());
//    }
//
//    if (params_.text_rec_score_thresh.has_value()) {
//        data["Modules.TextRecognition.score_thresh"] =
//            params_.text_rec_score_thresh.value();
//    }
//    if (params_.text_rec_input_shape.has_value()) {
//        data["Modules.TextRecognition.input_shape"] =
//            Utility::VecToString(params_.text_rec_input_shape.value());
//    }
//
//    if (params_.lang.has_value()) {
//        data["Global.lang"] = params_.lang.value();
//    }
//    if (params_.ocr_version.has_value()) {
//        data["Global.ocr_version"] = params_.ocr_version.value();
//    }
//    if (params_.vis_font_dir.has_value()) {
//        data["Global.vis_font_dir"] = params_.vis_font_dir.value();
//    }
//    if (params_.device.has_value()) {
//        data["Global.device"] = params_.device.value();
//    }
//
//    // Keep these fixed params
//    data["Global.enable_mkldnn"] = params_.enable_mkldnn;
//    data["Global.mkldnn_cache_capacity"] = params_.mkldnn_cache_capacity;
//    data["Global.precision"] = params_.precision;
//    data["Global.cpu_threads"] = params_.cpu_threads;
//}

void _ImagePipeline::OverrideConfig() {
    auto& data = config_.Data();
    if (params_.doc_orientation_classify_model_name.has_value()) {
        auto it = config_.FindKey("DocOrientationClassify.model_name");
        if (!it.ok()) {
            data["SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify."
                "model_name"] = params_.doc_orientation_classify_model_name.value();
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.doc_orientation_classify_model_name.value();
        }
    }
    if (params_.doc_orientation_classify_model_dir.has_value()) {
        auto it = config_.FindKey("DocOrientationClassify.model_dir");
        if (!it.ok()) {
            data["SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify."
                "model_dir"] = params_.doc_orientation_classify_model_dir.value();
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.doc_orientation_classify_model_dir.value();
        }
    }
    if (params_.doc_unwarping_model_name.has_value()) {
        auto it = config_.FindKey("DocUnwarping.model_name");
        if (!it.ok()) {
            data["SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_name"] =
                params_.doc_unwarping_model_name.value();
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.doc_unwarping_model_name.value();
        }
    }
    if (params_.doc_unwarping_model_dir.has_value()) {
        auto it = config_.FindKey("DocUnwarping.model_dir");
        if (!it.ok()) {
            data["SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_dir"] =
                params_.doc_unwarping_model_dir.value();
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.doc_unwarping_model_dir.value();
        }
    }
    if (params_.text_detection_model_name.has_value()) {
        auto it = config_.FindKey("TextDetection.model_name");
        if (!it.ok()) {
            data["SubModules.TextDetection.model_name"] =
                params_.text_detection_model_name.value();
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.text_detection_model_name.value();
        }
    }
    if (params_.text_detection_model_dir.has_value()) {
        auto it = config_.FindKey("TextDetection.model_dir");
        if (!it.ok()) {
            data["SubModules.TextDetection.model_dir"] =
                params_.text_detection_model_dir.value();
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.text_detection_model_dir.value();
        }
    }
    if (params_.textline_orientation_model_name.has_value()) {
        auto it = config_.FindKey("TextLineOrientation.model_name");
        if (!it.ok()) {
            data["SubModules.TextLineOrientation.model_name"] =
                params_.textline_orientation_model_name.value();
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.textline_orientation_model_name.value();
        }
    }
    if (params_.textline_orientation_model_dir.has_value()) {
        auto it = config_.FindKey("TextLineOrientation.model_dir");
        if (!it.ok()) {
            data["SubModules.TextLineOrientation.model_dir"] =
                params_.textline_orientation_model_dir.value();
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.textline_orientation_model_dir.value();
        }
    }
    if (params_.textline_orientation_batch_size.has_value()) {
        auto it = config_.FindKey("TextLineOrientation.batch_size");
        if (!it.ok()) {
            data["SubModules.TextLineOrientation.batch_size"] =
                std::to_string(params_.textline_orientation_batch_size.value());
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] =
                std::to_string(params_.textline_orientation_batch_size.value());
        }
    }

    if (params_.text_recognition_model_name.has_value()) {
        auto it = config_.FindKey("TextRecognition.model_name");
        if (!it.ok()) {
            data["SubModules.TextRecognition.model_name"] =
                params_.text_recognition_model_name.value();
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.text_recognition_model_name.value();
        }
    }
    if (params_.text_recognition_model_dir.has_value()) {
        auto it = config_.FindKey("TextRecognition.model_dir");
        if (!it.ok()) {
            data["SubModules.TextRecognition.model_dir"] =
                params_.text_recognition_model_dir.value();
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.text_recognition_model_dir.value();
        }
    }
    if (params_.text_recognition_batch_size.has_value()) {
        auto it = config_.FindKey("TextRecognition.batch_size");
        if (!it.ok()) {
            data["SubModules.TextRecognition.batch_size"] =
                std::to_string(params_.text_recognition_batch_size.value());
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = std::to_string(params_.text_recognition_batch_size.value());
        }
    }

    if (params_.use_doc_orientation_classify.has_value()) {
        auto it = config_.FindKey("DocPreprocessor.use_doc_orientation_classify");
        if (!it.ok()) {
            data["SubPipelines.DocPreprocessor.use_doc_orientation_classify"] =
                params_.use_doc_orientation_classify.value() ? "true" : "false";
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] =
                params_.use_doc_orientation_classify.value() ? "true" : "false";
        }
    }
    if (params_.use_doc_unwarping.has_value()) {
        auto it = config_.FindKey("DocPreprocessor.use_doc_unwarping");
        if (!it.ok()) {
            data["SubPipelines.DocPreprocessor.use_doc_unwarping"] =
                params_.use_doc_unwarping.value() ? "true" : "false";
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.use_doc_unwarping.value() ? "true" : "false";
        }
    }
    if (params_.use_textline_orientation.has_value()) {
        auto it = config_.FindKey("use_textline_orientation");
        if (!it.ok()) {
            data["use_textline_orientation"] =
                params_.use_textline_orientation.value() ? "true" : "false";
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.use_textline_orientation.value() ? "true" : "false";
        }
    }
    if (params_.text_det_limit_side_len.has_value()) {
        auto it = config_.FindKey("TextDetection.limit_side_len");
        if (!it.ok()) {
            data["SubModules.TextDetection.limit_side_len"] =
                std::to_string(params_.text_det_limit_side_len.value());
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = std::to_string(params_.text_det_limit_side_len.value());
        }
    }
    if (params_.text_det_limit_type.has_value()) {
        auto it = config_.FindKey("TextDetection.limit_type");
        if (!it.ok()) {
            data["SubModules.TextDetection.limit_type"] =
                params_.text_det_limit_type.value();
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = params_.text_det_limit_type.value();
        }
    }
    if (params_.text_det_thresh.has_value()) {
        auto it = config_.FindKey("TextDetection.thresh");
        if (!it.ok()) {
            data["SubModules.TextDetection.thresh"] =
                std::to_string(params_.text_det_thresh.value());
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = std::to_string(params_.text_det_thresh.value());
        }
    }
    if (params_.text_det_box_thresh.has_value()) {
        auto it = config_.FindKey("TextDetection.box_thresh");
        if (!it.ok()) {
            data["SubModules.TextDetection.box_thresh"] =
                std::to_string(params_.text_det_box_thresh.value());
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = std::to_string(params_.text_det_box_thresh.value());
        }
    }
    if (params_.text_det_unclip_ratio.has_value()) {
        auto it = config_.FindKey("TextDetection.unclip_ratio");
        if (!it.ok()) {
            data["SubModules.TextDetection.unclip_ratio"] =
                std::to_string(params_.text_det_unclip_ratio.value());
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = std::to_string(params_.text_det_unclip_ratio.value());
        }
    }
    if (params_.text_det_input_shape.has_value()) {
        auto it = config_.FindKey("TextDetection.input_shape");
        if (!it.ok()) {
            data["SubModules.TextDetection.input_shape"] =
                Utility::VecToString(params_.text_det_input_shape.value());
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = Utility::VecToString(params_.text_det_input_shape.value());
        }
    }
    if (params_.text_rec_score_thresh.has_value()) {
        auto it = config_.FindKey("TextRecognition.score_thresh");
        if (!it.ok()) {
            data["SubModules.TextRecognition.score_thresh"] =
                std::to_string(params_.text_rec_score_thresh.value());
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = std::to_string(params_.text_rec_score_thresh.value());
        }
    }
    if (params_.text_rec_input_shape.has_value()) {
        auto it = config_.FindKey("TextRecognition.input_shape");
        if (!it.ok()) {
            data["SubModules.TextRecognition.input_shape"] =
                Utility::VecToString(params_.text_rec_input_shape.value());
        }
        else {
            auto key = it.value().first;
            data.erase(data.find(key));
            data[key] = Utility::VecToString(params_.text_rec_input_shape.value());
        }
    }
}