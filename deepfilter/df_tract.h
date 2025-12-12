#pragma once

// df_tract.h
// High-level DeepFilterNet tract wrapper in C++
// Comments are in English as requested.

#include <vector>
#include <deque>
#include <complex>
#include <cstddef>
#include <optional>
#include <memory>

#include "./../common_def.h"
#include "df_state.h"
#include "model_runner.h" // Your ONNX-MLIR runner wrapper
#include "df_config_loader.h"

//using Complex32 = std::complex<float>;

namespace df {
    class MMS_EXPORT DfTractCpp {
    public:
        struct RawOut {
            float lsnr{ 0.0f };
            std::optional<Tensor> gains; // [ch, nb_erb]
            std::optional<Tensor> coefs; // [ch, nb_df, df_order, 2]
        };

        explicit DfTractCpp(const DfTractConfig& cfg);

        // Initialize internal buffers (must be called once after construction).
        void init();

        // Process one hop of audio (interleaved or per-channel contiguous frames)
        // noisy: shape [ch * hop_size]
        // enh:   shape [ch * hop_size]
        // returns local SNR estimate (lsnr).
        float process(const float* noisy, float* enh);

        // Also expose raw process (for debugging)
        RawOut process_raw();

        // Access DFState per channel
        std::shared_ptr<DFState> state(std::size_t ch_index) const {
            return df_states_.at(ch_index);
        }

        const DfTractConfig& cfg() const { return cfg_; }

    private:
        // Apply DeepFiltering using a deque of spectra and DF coefficients.
        void df_apply(const std::deque<Tensor>& spec_x,
            const Tensor& coefs,
            std::size_t nb_df,
            std::size_t df_order,
            std::size_t n_freqs,
            Tensor& spec_out) const;

    private:
        DfTractConfig cfg_;

        std::shared_ptr<ModelRunnerShim> enc_;
        std::shared_ptr<ModelRunnerShim> erb_dec_;
        std::shared_ptr<ModelRunnerShim> df_dec_;

        std::size_t n_freqs_{ 0 };
        std::size_t skip_counter_{ 0 };

        std::vector<std::shared_ptr<DFState>> df_states_;

        // Buffers matching Rust shapes:
        // spec_buf_: [ch, 1, 1, n_freqs, 2] (flattened)
        // erb_buf_:  [ch, 1, 1, nb_erb]
        // cplx_buf_: [ch, 1, nb_df, 2]
        Tensor spec_buf_;
        Tensor erb_buf_;
        Tensor cplx_buf_;

        // Rolling buffers for DeepFiltering:
        std::deque<Tensor> rolling_spec_buf_y_;
        std::deque<Tensor> rolling_spec_buf_x_;

        // Pre-allocated zero ERB mask
        std::vector<float> m_zeros_;
    };

} // namespace df
