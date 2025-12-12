#pragma once

// df_state.h
// C++ DFState implementation (pocketfft-backed)
// Comments inside code are in English as requested.

#include <vector>
#include <complex>
#include <cstddef>
#include <functional>
#include <memory>

#include "./../common_def.h"
#include "model_shim/fft/kiss_fft_cxx.h"

namespace df {

    //using Complex32 = std::complex<float>;

    //// Forward declaration of pocketfft plans.
    //// You may adapt this to your concrete pocketfft API.
    //struct FftPlan {
    //    std::size_t n = 0;
    //    // You can store any additional fields needed by your FFT backend.
    //};

    //// Simple wrapper for a 1D real FFT using pocketfft (or other library).
    //class RealFft {
    //public:
    //    RealFft() = default;
    //    RealFft(std::size_t n);

    //    std::size_t size() const { return n_; }

    //    // Forward: real input -> complex spectrum (size n_/2+1)
    //    void forward(const float* in, Complex32* out) const;

    //    // Inverse: complex spectrum -> real output (size n_)
    //    void inverse(const Complex32* in, float* out) const;

    //private:
    //    std::size_t n_{ 0 };
    //    std::size_t n_freqs_{ 0 };
    //    std::shared_ptr<FftPlan> plan_fwd_;
    //    std::shared_ptr<FftPlan> plan_inv_;
    //};

    // DFState replicates the behavior of the Rust DFState used in DeepFilterNet.
    class MMS_EXPORT DFState {
    public:
        DFState(std::size_t sr,
            std::size_t fft_size,
            std::size_t hop_size,
            std::size_t nb_erb,
            std::size_t min_nb_freqs);

        // Initialize mean_norm_state and unit_norm_state
        // nb_df is the number of low-frequency bins used for DF.
        void init_norm_states(std::size_t nb_df);

        // Feature extraction for ERB bands: writes nb_erb floats into out_feat
        void feat_erb(const float* spectrum_interleaved,
            float alpha,
            float* out_feat);

        // Feature extraction for complex features: outputs nb_df complex values
        void feat_cplx(const float* spectrum_interleaved,
            float alpha,
            float* out_feat);

        // STFT analysis: time-domain input -> complex spectrum (size = fft_size/2+1)
        void analysis(const  float* input, Complex32* spectrum);

        // ISTFT synthesis: complex spectrum -> time-domain output (hop_size samples)
        void synthesis(const Complex32* spectrum, float* output);

        // Apply ERB mask m (size nb_erb) to the complex spectrum in-place
        //void apply_mask(Complex32* spectrum, const float* m);
        void apply_mask(float* spectrum_interleaved, const float* m);

        // Post filter as in DeepFilterNet: noisy + enhanced spectra, band-wise attenuation
        void post_filter(const std::vector<Complex32>& noisy,
            std::vector<Complex32>& enh,
            float beta) const;

        std::size_t sample_rate() const { return sr_; }
        std::size_t fft_size() const { return fft_size_; }
        std::size_t hop_size() const { return hop_size_; }
        std::size_t nb_erb() const { return nb_erb_; }

    private:
        void compute_band_corr(float* out_feat, const float* spectrum_interleaved);
        void log_transform(float* out_feat);
        void band_mean_norm_erb(float* out_feat, float alpha);

        // Internal helpers
        static float freq2erb(float freq_hz);
        static float erb2freq(float n_erb);

        // Build ERB filterbank bands (0-based indices into [0..freq_size_))
        static std::vector<std::size_t> erb_fb(std::size_t sr,
            std::size_t fft_size,
            std::size_t nb_bands,
            std::size_t min_nb_freqs);

        // window
        static std::vector<float> build_window(std::size_t win_size);

        // Post-filter implementation
        static void post_filter_impl(const std::vector<Complex32>& noisy,
            std::vector<Complex32>& enh,
            float beta);

    private:
        std::size_t sr_;            // sample rate
        std::size_t fft_size_;      // FFT size
        std::size_t hop_size_;      // frame size
        std::size_t freq_size_;     // fft_size/2 + 1
        std::size_t nb_erb_;

        // window + normalization
        std::vector<float> window_;
        float wnorm_; // analysis normalization factor

        // overlap buffers
        std::vector<float> analysis_mem_;  // size = fft_size - hop_size
        std::vector<float> synthesis_mem_; // size = fft_size - hop_size

        // ERB band map: vector of band sizes and cumulative indices
        std::vector<std::size_t> erb_;

        // Normalization states
        std::vector<float> mean_norm_state_; // per ERB band
        std::vector<float> unit_norm_state_; // per DF band

        // FFT engine
        //RealFft fft_;
        x_fft::KissFFT fft_forward_;
        x_fft::KissFFT fft_inverse_;
        std::vector<float> time_buf_;
        std::vector<Complex32> freq_buf_;
    };

} // namespace df
