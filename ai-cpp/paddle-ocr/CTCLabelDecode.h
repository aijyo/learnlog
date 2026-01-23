#pragma once

#include <opencv2/core.hpp>
#include <list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./utils/absl_shim.h"

// A lightweight CTC greedy decoder (PaddleOCR-style).
// - Input: preds [N, T, C], CV_32F
// - For each time step: argmax over C
// - Remove duplicates (optional) and ignore tokens (blank id)
// - Map ids to characters and output {text, mean_confidence}
class CTCLabelDecode {
public:
    // character_list: list of characters from yaml config.
    // use_space_char: whether to append an extra space char to dict.
    explicit CTCLabelDecode(const std::vector<std::string>& character_list,
        bool use_space_char = true);

    // preds must be CV_32F with dims=3: [N,T,C]
    // return: per-sample {text, confidence}
    absl::StatusOr<std::vector<std::pair<std::string, float>>>
        Apply(const cv::Mat& preds) const;

private:
    absl::StatusOr<std::pair<std::string, float>>
        Process(const cv::Mat& pred_data) const;

    absl::StatusOr<std::pair<std::string, float>>
        Decode(std::list<int>& text_index, std::list<float>& text_prob,
            bool is_remove_duplicate) const;

    void AddSpecialChar();

private:
    // PaddleOCR convention: blank token id is 0
    static constexpr int kBlankId = 0;

    // Ignore tokens (e.g. blank)
    const std::vector<int> IGNORE_TOKEN = { kBlankId };

    std::vector<std::string> character_list_;
    std::unordered_map<int, std::string> dict_;
    bool use_space_char_ = true;
};
