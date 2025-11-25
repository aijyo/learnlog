// df_state.cpp
// DFState implementation using pocketfft
// Comments in English.

#include "df_state.h"

#define _USE_MATH_DEFINES

#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cstring>

// Include your pocketfft header here. Adjust the name if necessary.
// Common header names: "pocketfft_hdronly.h", "pocketfft.hpp", etc.
// This example assumes a header-only pocketfft with C++ API.
#include "3rd/pocketfft_hdronly.h"

namespace df {

    using namespace pocketfft;

    // ========================= RealFft implementation =========================

    RealFft::RealFft(std::size_t n)
        : n_(n)
    {
        if (n_ == 0) throw std::runtime_error("RealFft: size must be > 0");
        n_freqs_ = n_ / 2 + 1;
        plan_fwd_ = std::make_shared<FftPlan>();
        plan_inv_ = std::make_shared<FftPlan>();
        plan_fwd_->n = n_;
        plan_inv_->n = n_;
    }

    void RealFft::forward(const float* in, Complex32* out) const {
        if (!in || !out) throw std::runtime_error("RealFft::forward: null pointer");

        shape_t shape{ n_ };
        stride_t stride_in{ static_cast<ptrdiff_t>(sizeof(float)) };
        stride_t stride_out{ static_cast<ptrdiff_t>(sizeof(std::complex<float>)) };
        std::size_t axis = 0;

        //std::vector<double> vin(n_);
        //for (std::size_t i = 0; i < n_; ++i)
        //    vin[i] = static_cast<double>(in[i]);

        auto vin = in;
        auto vout = out;

        //std::vector<Complex32> vout(n_freqs_);
        r2c(shape, stride_in, stride_out,
            axis, FORWARD,
            vin, out, (float)1.0);

        //for (std::size_t i = 0; i < n_freqs_; ++i) {
        //    out[i] = Complex32(static_cast<float>(vout[i].real()),
        //        static_cast<float>(vout[i].imag()));
        //}
    }

    void RealFft::inverse(const Complex32* in, float* out) const {
        if (!in || !out) throw std::runtime_error("RealFft::inverse: null pointer");

        shape_t shape{ n_ };
        stride_t stride_in{ static_cast<ptrdiff_t>(sizeof(std::complex<float>)) };
        stride_t stride_out{ static_cast<ptrdiff_t>(sizeof(float)) };
        std::size_t axis = 0;

        //std::vector<std::complex<double>> vin(n_freqs_);
        //for (std::size_t i = 0; i < n_freqs_; ++i) {
        //    vin[i] = std::complex<double>(in[i].real(), in[i].imag());
        //}
        //std::vector<double> vout(n_);

        auto vin = in;
        auto vout = out;
        c2r(shape, stride_in, stride_out,
            axis, BACKWARD,
            vin, vout, (float)1.0);

        //for (std::size_t i = 0; i < n_; ++i) {
        //    out[i] = static_cast<float>(vout[i]);
        //}

        // double scale = 1.0 / static_cast<double>(n_);
        // out[i] = static_cast<float>(vout[i] * scale);
    }

    // ========================= Helper functions =========================

    namespace {
        inline float freq2erb_impl(float freq_hz) {
            // Matching typical ERB formula used in DeepFilterNet:
            // ERB(f) = 9.265 * ln(1 + f / (24.7 * 9.265))
            return 9.265f * std::log1pf(freq_hz / (24.7f * 9.265f));
        }

        inline float erb2freq_impl(float n_erb) {
            // Inverse of freq2erb
            return 24.7f * 9.265f * (std::exp(n_erb / 9.265f) - 1.0f);
        }

        inline std::vector<float> build_window_impl(std::size_t win_size) {
            std::vector<float> w(win_size, 0.0f);
            if (win_size == 0 ) return w;

            const double half_pi = 0.5 * M_PI;
            const double window_size_h = static_cast<double>(win_size) / 2.0;
            for (std::size_t i = 0; i < win_size; ++i) {
                double s = std::sin(half_pi *
                    (static_cast<double>(i) + 0.5) /
                    static_cast<double>(window_size_h));
                double inner = half_pi * s * s;
                w[i] = static_cast<float>(std::sin(inner));
            }
            return w;
        }

        inline void apply_window_in_place(std::vector<float>& buf, const std::vector<float>& window) {
            std::size_t n = std::min(buf.size(), window.size());
            for (std::size_t i = 0; i < n; ++i) buf[i] *= window[i];
        }

        // Band-wise correlation compute: out[i] = mean( real(x * conj(p)) ) over band_size
        inline void compute_band_corr(const Complex32* x,
            const Complex32* p,
            const std::vector<std::size_t>& erb,
            std::vector<float>& out) {
            std::size_t erb_len = erb.size();
            out.assign(erb_len, 0.0f);
            //if (x.size() != p.size()) throw std::runtime_error("compute_band_corr size mismatch");
            std::size_t bcsum = 0;
            for (std::size_t i = 0; i < erb_len; ++i) {
                std::size_t band_size = erb[i];
                if (band_size == 0) continue;
                float acc = 0.0f;
                for (std::size_t j = 0; j < band_size; ++j) {
                    const Complex32& xv = x[bcsum + j];
                    const Complex32& pv = p[bcsum + j];
                    acc += (xv.real() * pv.real() + xv.imag() * pv.imag());
                }
                out[i] = acc / static_cast<float>(band_size);
                bcsum += band_size;
            }
        }
    } // namespace

    // ========================= DFState implementation =========================

    DFState::DFState(std::size_t sr,
        std::size_t fft_size,
        std::size_t hop_size,
        std::size_t nb_erb,
        std::size_t min_nb_freqs)
        : sr_(sr)
        , fft_size_(fft_size)
        , hop_size_(hop_size)
        , nb_erb_(nb_erb)
        , fft_(fft_size)
    {
        if (hop_size_ == 0 || fft_size_ == 0) {
            throw std::runtime_error("DFState: hop_size and fft_size must be > 0");
        }
        freq_size_ = fft_size_ / 2 + 1;

        if (hop_size_ > fft_size_) {
            throw std::runtime_error("DFState: hop_size cannot exceed fft_size");
        }
        //auto window_size_h = fft_size / 2.0;
        window_ = build_window(fft_size_);

        float window_size_f = static_cast<float>(fft_size_);
        float frame_size_f = static_cast<float>(hop_size_);

        // Rust: wnorm = 1. / (window_size.pow(2) as f32 / (2 * frame_size) as f32);
        // -> 2 * frame_size / (window_size * window_size)
        wnorm_ = 1.0f / ((window_size_f * window_size_f) / (2.0f * frame_size_f));

        // Overlap buffers have size fft_size - hop_size
        std::size_t overlap = fft_size_ - hop_size_;
        analysis_mem_.assign(overlap, 0.0f);
        synthesis_mem_.assign(overlap, 0.0f);

        // Build ERB bands
        erb_ = erb_fb(sr_, fft_size_, nb_erb_, min_nb_freqs);
        //erb_cum_.resize(erb_.size());
        //std::partial_sum(erb_.begin(), erb_.end(), erb_cum_.begin());

        time_buf_.assign(fft_size_, 0.0f);
        freq_buf_.assign(freq_size_, Complex32(0.0f, 0.0f));
    }

    void DFState::init_norm_states(std::size_t nb_df) {
        // Rust uses MEAN_NORM_INIT and UNIT_NORM_INIT which are linear ramp values.
        static constexpr float MEAN_NORM_INIT_MIN = -60.0f;
        static constexpr float MEAN_NORM_INIT_MAX = -90.0f;
        std::size_t nb_erb = nb_erb_;
        mean_norm_state_.resize(nb_erb);
        if (nb_erb <= 1) {
            mean_norm_state_.assign(nb_erb, MEAN_NORM_INIT_MIN);
        }
        else {
            float step = (MEAN_NORM_INIT_MAX - MEAN_NORM_INIT_MIN)
                / static_cast<float>(nb_erb - 1);
            for (std::size_t i = 0; i < nb_erb; ++i) {
                mean_norm_state_[i] = MEAN_NORM_INIT_MIN + static_cast<float>(i) * step;
            }
        }

        static constexpr float UNIT_NORM_INIT_MIN = 0.001f;
        static constexpr float UNIT_NORM_INIT_MAX = 0.0001f;
        unit_norm_state_.resize(nb_df);
        if (nb_df <= 1) {
            unit_norm_state_.assign(nb_df, UNIT_NORM_INIT_MIN);
        }
        else {
            float step = (UNIT_NORM_INIT_MAX - UNIT_NORM_INIT_MIN)
                / static_cast<float>(nb_df - 1);
            for (std::size_t i = 0; i < nb_df; ++i) {
                unit_norm_state_[i] = UNIT_NORM_INIT_MIN + static_cast<float>(i) * step;
            }
        }
    }

    float DFState::freq2erb(float freq_hz) {
        return freq2erb_impl(freq_hz);
    }

    float DFState::erb2freq(float n_erb) {
        return erb2freq_impl(n_erb);
    }

    std::vector<std::size_t> DFState::erb_fb(std::size_t sr,
        std::size_t fft_size,
        std::size_t nb_bands,
        std::size_t min_nb_freqs) {
        std::size_t nyq_freq = sr / 2;
        float freq_width = static_cast<float>(sr) / static_cast<float>(fft_size);
        float erb_low = freq2erb(0.0f);
        float erb_high = freq2erb(static_cast<float>(nyq_freq));
        float step = (erb_high - erb_low) / static_cast<float>(nb_bands);

        std::vector<std::size_t> erb(nb_bands, 0);
        int freq_over = 0;   // how many bins to subtract in this band due to previous over-allocation
        int prev_freq = 0;   // or whatever initial value as Rust

        for (std::size_t i = 1; i <= nb_bands; ++i) {
            float f = erb2freq(erb_low + static_cast<float>(i) * step);
            int fb = static_cast<int>(std::round(f / freq_width));

            int nb_freqs = fb - prev_freq - freq_over;

            if (nb_freqs < static_cast<int>(min_nb_freqs)) {
                // Not enough freq bins in current erb band
                freq_over = static_cast<int>(min_nb_freqs) - nb_freqs; // remember how many extra we enforced
                nb_freqs = static_cast<int>(min_nb_freqs);             // enforce min_nb_freqs
            }
            else {
                freq_over = 0;
            }

            erb[i - 1] = static_cast<std::size_t>(nb_freqs);
            prev_freq = fb;
        }

        erb[nb_bands - 1] += 1; // since we have WINDOW_SIZE/2+1 frequency bins
        // Adjust last band to cover up to nyq
        std::size_t sum = std::accumulate(erb.begin(), erb.end(), static_cast<std::size_t>(0));
        auto too_large = sum - (fft_size / 2 + 1);
        if (too_large > 0) {
            erb[nb_bands - 1] -= too_large;
        }
        //assert!(sum == fft_size / 2 + 1);
        return erb;
    }

    std::vector<float> DFState::build_window(std::size_t win_size) {
        return build_window_impl(win_size);
    }

    // input  : time-domain frame, length = frame_size (hop_size_)
    // spectrum: complex spectrum, length = freq_size_
    void DFState::analysis(const float* input, Complex32* spectrum) {
        if (!input || !spectrum) {
            throw std::runtime_error("DFState::analysis: null pointer");
        }

        const std::size_t win = fft_size_;      // window_size
        const std::size_t h = hop_size_;      // frame_size
        const std::size_t win_h = win - h;        // window_size - frame_size

        // Sanity checks mirroring Rust debug_asserts
        if (analysis_mem_.size() < h) {
            throw std::runtime_error("DFState::analysis: analysis_mem_ too small");
        }
        if (time_buf_.size() < win) {
            throw std::runtime_error("DFState::analysis: time_buf_ too small");
        }
        if (freq_buf_.size() < freq_size_) {
            throw std::runtime_error("DFState::analysis: freq_buf_ too small");
        }

        // buffer used as Rust: let mut buf = state.fft_forward.make_input_vec();
        float* buf = time_buf_.data();

        // ---- 1) First part of the window on the previous frame ----
        // Rust:
        // let (buf_first, buf_second) = buf.split_at_mut(window_size - frame_size);
        // let (window_first, window_second) = state.window.split_at(window_size - frame_size);
        // for (&y, &w, x) in izip!(state.analysis_mem.iter(), window_first.iter(), buf_first.iter_mut()) {
        //     *x = y * w;
        // }
        //
        // => use first win_h samples of analysis_mem_ with first win_h window taps
        for (std::size_t i = 0; i < win_h; ++i) {
            buf[i] = analysis_mem_[i] * window_[i];
        }

        // ---- 2) Second part of the window on the new input frame ----
        // Rust:
        // for ((&y, &w), x) in input.iter().zip(window_second.iter()).zip(buf_second.iter_mut()) {
        //     *x = y * w;
        // }
        //
        // window_second starts at window_[win_h], length = h
        const float* win2 = window_.data() + win_h;
        for (std::size_t i = 0; i < h; ++i) {
            buf[win_h + i] = input[i] * win2[i];
        }

        // ---- 3) Shift analysis_mem (rotate_left(frame_size)) ----
        // Rust:
        // let analysis_split = state.analysis_mem.len() - state.frame_size;
        // if analysis_split > 0 {
        //     // hop_size is < window_size / 2
        //     state.analysis_mem.rotate_left(state.frame_size);
        // }
        const std::size_t len_mem = analysis_mem_.size();
        const std::size_t analysis_split = len_mem - h;
        if (analysis_split > 0) {
            std::rotate(analysis_mem_.begin(),
                analysis_mem_.begin() + h,
                analysis_mem_.end());
        }

        // ---- 4) Copy input to analysis_mem tail for next iteration ----
        // Rust:
        // for (x, &y) in state.analysis_mem[analysis_split..].iter_mut().zip(input) {
        //     *x = y
        // }
        //
        // i.e. analysis_mem_[analysis_split .. analysis_split + h] = input[0 .. h]
        for (std::size_t i = 0; i < h; ++i) {
            analysis_mem_[analysis_split + i] = input[i];
        }

        // ---- 5) Forward FFT: buf -> spectrum (via freq_buf_) ----
        // Rust:
        // state.fft_forward.process_with_scratch(&mut buf, output, &mut state.analysis_scratch)
        fft_.forward(buf, freq_buf_.data());

        // ---- 6) Apply normalization in analysis only (Rust: *x *= norm) ----
        // Rust:
        // let norm = state.wnorm;
        // for x in output.iter_mut() { *x *= norm; }
        for (std::size_t i = 0; i < freq_size_; ++i) {
            freq_buf_[i] *= wnorm_;
        }

        // ---- 7) Copy to output spectrum ----
        if (spectrum != freq_buf_.data()) {
            std::memcpy(spectrum,
                freq_buf_.data(),
                freq_size_ * sizeof(Complex32));
        }
    }

    void DFState::synthesis(const Complex32* spectrum, float* output) {
        if (!spectrum || !output) {
            throw std::runtime_error("DFState::synthesis: null pointer");
        }

        const std::size_t win = fft_size_;   // window_size
        const std::size_t h = hop_size_;   // frame_size (same as Rust's state.frame_size)

        // time_buf_ must hold 'win' samples
        if (time_buf_.size() < win) {
            throw std::runtime_error("DFState::synthesis: time_buf_ too small");
        }

        float* x = time_buf_.data(); // Rust: let mut x = state.fft_inverse.make_output_vec();

        // 1) Inverse FFT
        // Rust:
        // match state.fft_inverse.process_with_scratch(input, &mut x, &mut state.synthesis_scratch) { ... }
        // Here we assume fft_.inverse throws or handles internal errors.
        fft_.inverse(spectrum, x);

        // 2) Apply window in place
        // Rust: apply_window_in_place(&mut x, &state.window);
        for (std::size_t i = 0; i < win; ++i) {
            x[i] *= window_[i];
        }

        // 3) Overlap-add: x_first + synthesis_mem -> output
        // Rust:
        // let (x_first, x_second) = x.split_at(state.frame_size);
        // for ((&xi, &mem), out) in x_first.iter().zip(state.synthesis_mem.iter()).zip(output.iter_mut()) {
        //     *out = xi + mem;
        // }
        //
        // Here x_first is x[0 .. h), x_second is x[h .. win)
        for (std::size_t i = 0; i < h; ++i) {
            output[i] = x[i] + synthesis_mem_[i];
        }

        // 4) Update synthesis_mem for next frame
        //
        // Rust:
        // let split = state.synthesis_mem.len() - state.frame_size;
        // if split > 0 {
        //     state.synthesis_mem.rotate_left(state.frame_size);
        // }
        // let (s_first, s_second) = state.synthesis_mem.split_at_mut(split);
        // let (xs_first, xs_second) = x_second.split_at(split);
        // for (&xi, mem) in xs_first.iter().zip(s_first.iter_mut()) {
        //     *mem += xi;
        // }
        // for (&xi, mem) in xs_second.iter().zip(s_second.iter_mut()) {
        //     *mem = xi;
        // }

        const std::size_t len_mem = synthesis_mem_.size();  // should be win - h or larger
        if (len_mem < h) {
            throw std::runtime_error("DFState::synthesis: synthesis_mem_ too small");
        }

        const std::size_t split = len_mem - h;  // may be 0 if win == 2 * h

        // Rotate-left by h if split > 0 (same as Rust's rotate_left(frame_size))
        if (split > 0) {
            std::rotate(synthesis_mem_.begin(),
                synthesis_mem_.begin() + h,
                synthesis_mem_.end());
        }

        // Now we create the same logical splits as in Rust:
        // synthesis_mem = [s_first (len=split), s_second (len=h)]
        float* s_first = synthesis_mem_.data();         // length = split
        float* s_second = synthesis_mem_.data() + split; // length = len_mem - split (= h)

        const float* x_second = x + h;                   // length = win - h

        // x_second is split into [xs_first (len=split), xs_second (len=h)]
        const float* xs_first = x_second;               // first 'split' samples
        const float* xs_second = x_second + split;       // remaining 'h' samples

        // Overlap-add for next frame (first part)
        // for (&xi, mem) in xs_first.iter().zip(s_first.iter_mut()) { *mem += xi; }
        for (std::size_t i = 0; i < split; ++i) {
            s_first[i] += xs_first[i];
        }

        // Override left-shifted buffer (second part)
        // for (&xi, mem) in xs_second.iter().zip(s_second.iter_mut()) { *mem = xi; }
        for (std::size_t i = 0; i < h; ++i) {
            s_second[i] = xs_second[i];
        }
    }

    void DFState::feat_erb(const Complex32* spectrum,
        float alpha,
        float* out_feat) {
        if (!spectrum || !out_feat) {
            throw std::runtime_error("DFState::feat_erb: null pointer");
        }

        if (erb_.size() != nb_erb_ || mean_norm_state_.size() != nb_erb_) {
            throw std::runtime_error("DFState::feat_erb: size mismatch");
        }

        // === compute_band_corr(out_feat, spectrum, spectrum, &erb_) ===
        // Initialize output bands to 0
        for (std::size_t i = 0; i < nb_erb_; ++i) {
            out_feat[i] = 0.0f;
        }

        std::size_t bcsum = 0;
        for (std::size_t b = 0; b < nb_erb_; ++b) {
            std::size_t band_size = erb_[b];
            if (band_size == 0) continue;

            float acc = 0.0f;
            // Auto-correlation of spectrum with itself: Re*Re + Im*Im
            for (std::size_t j = 0; j < band_size; ++j) {
                const Complex32& x = spectrum[bcsum + j];
                acc += x.real() * x.real() + x.imag() * x.imag();
            }

            float k = 1.0f / static_cast<float>(band_size);
            out_feat[b] += acc * k;  // average per band

            bcsum += band_size;
        }

        // === 10 * log10(out + 1e-10) ===
        const float eps = 1e-10f;
        for (std::size_t i = 0; i < nb_erb_; ++i) {
            out_feat[i] = 10.0f * std::log10(out_feat[i] + eps);
        }

        // === band_mean_norm_erb(output, &mut self.mean_norm_state, alpha) ===
        // Rust:
        // for (x, s) in xs.iter_mut().zip(state.iter_mut()) {
        //     *s = *x * (1. - alpha) + *s * alpha;
        //     *x -= *s;
        //     *x /= 40.;
        // }
        for (std::size_t i = 0; i < nb_erb_; ++i) {
            float& x = out_feat[i];
            float& s = mean_norm_state_[i];

            // Exponential moving average of log-magnitude
            s = x * (1.0f - alpha) + s * alpha;

            // Mean normalization and scaling by 1/40
            x -= s;
            x /= 40.0f;
        }
    }

    void DFState::feat_cplx(const Complex32* spectrum,
        float alpha,
        Complex32* out_feat) {
        if (!spectrum || !out_feat) {
            throw std::runtime_error("DFState::feat_cplx: null pointer");
        }

        // In Rust: output.clone_from_slice(input);
        // Here we copy spectrum to out_feat first.
        std::size_t nb_df = unit_norm_state_.size();
        for (std::size_t i = 0; i < nb_df; ++i) {
            out_feat[i] = spectrum[i];
        }

        // band_unit_norm(output, &mut self.unit_norm_state, alpha)
        //
        // Rust:
        // for (x, s) in xs.iter_mut().zip(state.iter_mut()) {
        //     *s = x.norm() * (1. - alpha) + *s * alpha;
        //     *x /= s.sqrt();
        // }
        for (std::size_t i = 0; i < nb_df; ++i) {
            Complex32& x = out_feat[i];
            float& s = unit_norm_state_[i];

            // x.norm() in Rust (num_complex) is magnitude: sqrt(re^2 + im^2)
            float mag = std::abs(x);

            // Exponential moving average of magnitude
            s = mag * (1.0f - alpha) + s * alpha;

            // Divide by sqrt(s); guard zero to avoid NaN
            float denom = std::sqrt(std::max(s, 0.0f));
            if (denom > 0.0f) {
                x /= denom;
            }
            else {
                // If s is zero (e.g. all inputs are exactly zero), output zero
                x = Complex32(0.0f, 0.0f);
            }
        }
    }

    void DFState::apply_mask(Complex32* spectrum, const float* m) {
        if (!spectrum || !m) {
            throw std::runtime_error("DFState::apply_mask: null pointer");
        }
        std::size_t bcsum = 0;
        for (std::size_t i = 0; i < nb_erb_; ++i) {
            std::size_t band_size = erb_[i];
            float g = m[i];

            for (std::size_t j = 0; j < band_size; ++j) {
                std::size_t idx = bcsum + j;

                if (idx >= freq_size_) {
                    break;
                }
                spectrum[idx] *= g;
            }
            bcsum += band_size;
        }
    }

    void DFState::post_filter_impl(const std::vector<Complex32>& noisy,
        std::vector<Complex32>& enh,
        float beta) {
        // Size check
        if (noisy.size() != enh.size()) {
            throw std::runtime_error("post_filter_impl: size mismatch");
        }

        const std::size_t N = noisy.size();
        if (N == 0) return;

        const float eps = 1e-12f;
        const float pi = std::acos(-1.0f);  // PI
        const float beta_p1 = beta + 1.0f;

        float g[4];
        float g_sin[4];
        float pf[4];

        // Process in groups of 4, same as Rust noisy.chunks_exact(4)
        const std::size_t n_groups = N / 4;

        for (std::size_t group = 0; group < n_groups; ++group) {
            const std::size_t base = group * 4;

            const std::size_t i0 = base + 0;
            const std::size_t i1 = base + 1;
            const std::size_t i2 = base + 2;
            const std::size_t i3 = base + 3;

            // --- compute g[0..3] ---
            // g[k] = clamp(|e[k]| / (|n[k]| + eps), eps, 1.0)
            {
                float n_mag0 = std::abs(noisy[i0]);
                float e_mag0 = std::abs(enh[i0]);
                float ratio0 = e_mag0 / (n_mag0 + eps);
                if (ratio0 < eps)  ratio0 = eps;
                if (ratio0 > 1.0f) ratio0 = 1.0f;
                g[0] = ratio0;

                float n_mag1 = std::abs(noisy[i1]);
                float e_mag1 = std::abs(enh[i1]);
                float ratio1 = e_mag1 / (n_mag1 + eps);
                if (ratio1 < eps)  ratio1 = eps;
                if (ratio1 > 1.0f) ratio1 = 1.0f;
                g[1] = ratio1;

                float n_mag2 = std::abs(noisy[i2]);
                float e_mag2 = std::abs(enh[i2]);
                float ratio2 = e_mag2 / (n_mag2 + eps);
                if (ratio2 < eps)  ratio2 = eps;
                if (ratio2 > 1.0f) ratio2 = 1.0f;
                g[2] = ratio2;

                float n_mag3 = std::abs(noisy[i3]);
                float e_mag3 = std::abs(enh[i3]);
                float ratio3 = e_mag3 / (n_mag3 + eps);
                if (ratio3 < eps)  ratio3 = eps;
                if (ratio3 > 1.0f) ratio3 = 1.0f;
                g[3] = ratio3;
            }

            // --- compute g_sin[0..3] (完全按 Rust 展开形式) ---
            // g_sin[k] = g[k] * sin(g[k] * pi / 2.0)
            g_sin[0] = g[0] * std::sin(g[0] * pi * 0.5f);
            g_sin[1] = g[1] * std::sin(g[1] * pi * 0.5f);
            g_sin[2] = g[2] * std::sin(g[2] * pi * 0.5f);
            g_sin[3] = g[3] * std::sin(g[3] * pi * 0.5f);

            // --- compute pf[0..3] ---
            // pf[k] = (beta_p1 * g[k] / (1 + beta * (g[k]/g_sin[k])^2)) / g[k]
            //       = beta_p1 / (1 + beta * (g[k]/g_sin[k])^2)
            {
                float ratio_g0 = g[0] / g_sin[0];
                float denom0 = 1.0f + beta * (ratio_g0 * ratio_g0);
                pf[0] = beta_p1 / denom0;

                float ratio_g1 = g[1] / g_sin[1];
                float denom1 = 1.0f + beta * (ratio_g1 * ratio_g1);
                pf[1] = beta_p1 / denom1;

                float ratio_g2 = g[2] / g_sin[2];
                float denom2 = 1.0f + beta * (ratio_g2 * ratio_g2);
                pf[2] = beta_p1 / denom2;

                float ratio_g3 = g[3] / g_sin[3];
                float denom3 = 1.0f + beta * (ratio_g3 * ratio_g3);
                pf[3] = beta_p1 / denom3;
            }

            // --- apply pf to enh[i0..i3] ---
            // e[0] *= pf[0]; e[1] *= pf[1]; e[2] *= pf[2]; e[3] *= pf[3];
            enh[i0] *= pf[0];
            enh[i1] *= pf[1];
            enh[i2] *= pf[2];
            enh[i3] *= pf[3];
        }

        // Tail (N % 4 != 0) is ignored, same as Rust chunks_exact(4)
    }

    void DFState::post_filter(const std::vector<Complex32>& noisy,
        std::vector<Complex32>& enh,
        float beta) const {
        const_cast<DFState*>(this)->post_filter_impl(noisy, enh, beta);
    }


} // namespace df
