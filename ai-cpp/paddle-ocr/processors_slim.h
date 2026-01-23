// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "absl_shim.h"

// -----------------------------
// Minimal BaseProcessor (header-only)
// -----------------------------
// NOTE: This keeps the original pipeline style (Apply returns vector<cv::Mat>)
// without introducing external absl dependency. It relies on absl_shim.h only.
class BaseProcessor {
public:
  BaseProcessor() = default;
  virtual ~BaseProcessor() = default;

  virtual absl::StatusOr<std::vector<cv::Mat>>
  Apply(std::vector<cv::Mat> &input, const void *param_ptr = nullptr) const = 0;
};

// -----------------------------
// DetResizeForTest
// -----------------------------
// Keep this param because it carries the resize information required for det postprocess.
struct DetResizeForTestParam {
  std::optional<std::vector<int>> input_shape;   // [H, W] if provided
  std::optional<int> max_side_limit;             // optional
  std::optional<std::vector<int>> image_shape;   // output: [src_h, src_w]
  std::optional<bool> keep_ratio;                // optional
  std::optional<int> limit_side_len;             // optional
  std::optional<std::string> limit_type;         // "min" / "max"
  std::optional<int> resize_long;                // resize long side
};

class DetResizeForTest : public BaseProcessor {
public:
  explicit DetResizeForTest(const DetResizeForTestParam &params);

  absl::StatusOr<std::vector<cv::Mat>>
  Apply(std::vector<cv::Mat> &input,
        const void *param_ptr = nullptr) const override;

private:
  int resize_type_ = 0;
  std::vector<int> input_shape_;
  int limit_side_len_ = 736;
  std::string limit_type_ = "min";
  int resize_long_ = 960;
  int max_side_limit_ = 4000;
  bool keep_ratio_ = true;
};

// -----------------------------
// DBPostProcess (simplified)
// -----------------------------
// IMPORTANT: DBPostProcess does NOT inherit from BaseProcessor by request.
// It is a standalone class. The interface is kept similar to original:
// - constructor takes DBPostProcessParams
// - operator()(preds, img_shapes, optional overrides)
// - Apply(...) helper that fills result to param_ptr if you still use pipeline
struct DBPostProcessParams {
  std::optional<float> thresh;        // binarize threshold
  std::optional<float> box_thresh;    // filter threshold
  std::optional<float> unclip_ratio;  // expand ratio
  int max_candidates = 1000;
  int min_size = 3;
  bool use_dilation = false;
  std::string score_mode = "fast"; // "fast" or "slow"
  std::string box_type = "quad";   // keep "quad"; "poly" not supported in lite
};

struct DBPostProcessResult {
  std::vector<std::vector<cv::Point2f>> boxes; // quads in original image coords
  std::vector<float> scores;
};

class DBPostProcess {
public:
  explicit DBPostProcess(const DBPostProcessParams &params);

  // Kept: similar signature as original operator().
  // preds: probability map, can be (H,W) or (1,1,H,W)/(1,H,W)
  // img_shapes: [src_h, src_w]
  absl::StatusOr<std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
  operator()(const cv::Mat &preds,
             const std::vector<int> &img_shapes,
             std::optional<float> thresh = std::nullopt,
             std::optional<float> box_thresh = std::nullopt,
             std::optional<float> unclip_ratio = std::nullopt) const;

  // Optional helper to keep old pipeline style: input[0]=preds, and param_ptr=DBPostProcessResult*
  absl::StatusOr<std::vector<cv::Mat>>
  Apply(std::vector<cv::Mat> &input, const void *param_ptr = nullptr,
        const std::vector<int> *img_shapes = nullptr) const;

private:
  absl::StatusOr<std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
  Process(const cv::Mat &pred, const std::vector<int> &img_shape,
          float thresh, float box_thresh, float unclip_ratio) const;

  absl::StatusOr<std::pair<std::vector<std::vector<cv::Point2f>>, std::vector<float>>>
  BoxesFromBitmap(const cv::Mat &pred, const cv::Mat &bitmap,
                  int dest_width, int dest_height,
                  float box_thresh, float unclip_ratio) const;

  float BoxScoreFast(const cv::Mat &pred, const std::vector<cv::Point> &box) const;
  float BoxScoreSlow(const cv::Mat &pred, const std::vector<cv::Point> &box) const;

  std::vector<cv::Point2f> UnclipLite(const std::vector<cv::Point2f> &poly,
                                      float unclip_ratio) const;

  static std::vector<cv::Point2f> OrderQuad(const std::vector<cv::Point2f> &pts);
  static void ClipQuad(std::vector<cv::Point2f> &quad, int w, int h);

private:
  float thresh_ = 0.3f;
  float box_thresh_ = 0.7f;
  float unclip_ratio_ = 2.0f;
  int max_candidates_ = 1000;
  int min_size_ = 3;
  bool use_dilation_ = false;
  std::string score_mode_ = "fast";
  std::string box_type_ = "quad";
};
