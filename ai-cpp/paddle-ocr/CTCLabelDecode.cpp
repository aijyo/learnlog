#include "CTCLabelDecode.h"

#include <algorithm>
#include <cstring>

// ---- CTCLabelDecode (no external absl, no Utility::SplitBatch) ----

CTCLabelDecode::CTCLabelDecode(const std::vector<std::string>& character_list,
    bool use_space_char)
    : character_list_(character_list), use_space_char_(use_space_char) {
    // If config gives empty dict, fallback to a minimal default set.
    if (character_list_.empty()) {
        const std::string normal = "0123456789abcdefghijklmnopqrstuvwxyz";
        for (const auto& item : normal) {
            character_list_.emplace_back(std::string(1, item));
        }
    }

    if (use_space_char_) {
        character_list_.emplace_back(std::string(" "));
    }

    // Insert blank token at index 0 (PaddleOCR convention)
    AddSpecialChar();

    dict_.reserve(character_list_.size());
    for (int i = 0; i < (int)character_list_.size(); ++i) {
        dict_[i] = character_list_[i];
    }
}

absl::StatusOr<std::vector<std::pair<std::string, float>>>
CTCLabelDecode::Apply(const cv::Mat& preds) const {
    if (preds.empty()) {
        return absl::InvalidArgumentError("CTCLabelDecode::Apply: preds is empty.");
    }
    if (preds.type() != CV_32F) {
        return absl::InvalidArgumentError(
            absl::StrCat("CTCLabelDecode::Apply: preds must be CV_32F, got type=", preds.type()));
    }
    if (preds.dims != 3) {
        return absl::InvalidArgumentError(
            absl::StrCat("CTCLabelDecode::Apply: preds.dims must be 3 [N,T,C], got dims=", preds.dims));
    }

    const int N = preds.size[0];
    if (N <= 0) {
        return absl::InvalidArgumentError("CTCLabelDecode::Apply: batch N <= 0.");
    }

    std::vector<std::pair<std::string, float>> results;
    results.reserve((size_t)N);

    // Split batch by slicing [n:n+1, :, :] without relying on Utility::SplitBatch.
    for (int n = 0; n < N; ++n) {
        std::vector<cv::Range> ranges(3);
        ranges[0] = cv::Range(n, n + 1);
        ranges[1] = cv::Range::all();
        ranges[2] = cv::Range::all();

        // Clone to ensure contiguous buffer and avoid view-lifetime issues.
        cv::Mat one = preds(ranges).clone();

        auto r = Process(one);
        if (!r.ok()) {
            return r.status();
        }
        results.push_back(r.value());
    }

    return results;
}

absl::StatusOr<std::pair<std::string, float>>
CTCLabelDecode::Process(const cv::Mat& pred_data) const {
    if (pred_data.empty()) {
        return absl::InvalidArgumentError("CTCLabelDecode::Process: pred_data is empty.");
    }
    if (pred_data.type() != CV_32F) {
        return absl::InvalidArgumentError("CTCLabelDecode::Process: pred_data must be CV_32F.");
    }
    if (pred_data.dims != 3) {
        return absl::InvalidArgumentError(
            absl::StrCat("CTCLabelDecode::Process: pred_data.dims must be 3, got dims=", pred_data.dims));
    }

    // pred_data: [1, T, C]
    // Squeeze batch dim => [T, C]
    std::vector<int> shape_squeeze;
    shape_squeeze.reserve((size_t)pred_data.dims - 1);
    for (int i = 1; i < pred_data.dims; ++i) {
        shape_squeeze.push_back(pred_data.size[i]);
    }

    cv::Mat pred_tc = pred_data.reshape(1, (int)shape_squeeze.size(), shape_squeeze.data());
    if (pred_tc.dims != 2) {
        return absl::InternalError("CTCLabelDecode::Process: reshape to [T,C] failed.");
    }

    const int T = pred_tc.size[0];
    const int C = pred_tc.size[1];
    if (T <= 0 || C <= 0) {
        return absl::InvalidArgumentError("CTCLabelDecode::Process: invalid [T,C] after reshape.");
    }

    std::list<int> text_index;
    std::list<float> text_prob;

    // Greedy decode
    for (int t = 0; t < T; ++t) {
        const float* row = pred_tc.ptr<float>(t);
        float max_val = row[0];
        int max_idx = 0;

        for (int c = 1; c < C; ++c) {
            const float v = row[c];
            if (v > max_val) {
                max_val = v;
                max_idx = c;
            }
        }

        text_index.push_back(max_idx);
        text_prob.push_back(max_val);
    }

    return Decode(text_index, text_prob, /*is_remove_duplicate=*/true);
}

absl::StatusOr<std::pair<std::string, float>>
CTCLabelDecode::Decode(std::list<int>& text_index, std::list<float>& text_prob,
    bool is_remove_duplicate) const {
    if (text_index.empty()) {
        return std::pair<std::string, float>("", 0.f);
    }

    std::vector<bool> selection(text_index.size(), true);

    // Remove duplicate tokens (CTC rule).
    if (is_remove_duplicate && text_index.size() > 1) {
        auto prev = text_index.begin();
        auto curr = std::next(prev);
        size_t idx = 1;
        for (; curr != text_index.end(); ++curr, ++prev, ++idx) {
            if (*curr == *prev) selection[idx] = false;
        }
    }

    // Remove ignore tokens (blank etc.).
    for (const auto& ignore_item : IGNORE_TOKEN) {
        size_t idx = 0;
        for (auto it = text_index.begin(); it != text_index.end(); ++it, ++idx) {
            if (*it == ignore_item) selection[idx] = false;
        }
    }

    // Apply selection to text_index.
    {
        auto sel_it = selection.begin();
        for (auto it = text_index.begin(); it != text_index.end();) {
            if (!(*sel_it)) {
                it = text_index.erase(it);
            }
            else {
                ++it;
            }
            ++sel_it;
        }
    }

    // Apply selection to text_prob.
    {
        auto sel_it = selection.begin();
        for (auto it = text_prob.begin(); it != text_prob.end();) {
            if (!(*sel_it)) {
                it = text_prob.erase(it);
            }
            else {
                ++it;
            }
            ++sel_it;
        }
    }

    // Map ids to chars.
    std::string text;
    text.reserve(text_index.size() * 2);

    for (auto it = text_index.begin(); it != text_index.end(); ++it) {
        const int id = *it;
        auto dit = dict_.find(id);
        if (dit != dict_.end()) {
            text += dit->second;
        }
        else {
            // Out of range => append a space to keep behavior tolerant.
            text += " ";
        }
    }

    // Mean confidence.
    float mean = 0.f;
    if (!text_prob.empty()) {
        float sum = 0.f;
        for (float p : text_prob) sum += p;
        mean = sum / (float)text_prob.size();
    }

    return std::pair<std::string, float>(text, mean);
}

void CTCLabelDecode::AddSpecialChar() {
    // PaddleOCR uses blank at index 0.
    character_list_.insert(character_list_.begin(), "blank");
}
