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
#include <corecrt_math_defines.h>
//
//// Include your pocketfft header here. Adjust the name if necessary.
//// Common header names: "pocketfft_hdronly.h", "pocketfft.hpp", etc.
//// This example assumes a header-only pocketfft with C++ API.
//#include "3rd/pocketfft_hdronly.h"

namespace df {

    //using namespace pocketfft;

    //// ========================= RealFft implementation =========================

    //RealFft::RealFft(std::size_t n)
    //    : n_(n)
    //{
    //    if (n_ == 0) throw std::runtime_error("RealFft: size must be > 0");
    //    n_freqs_ = n_ / 2 + 1;
    //    plan_fwd_ = std::make_shared<FftPlan>();
    //    plan_inv_ = std::make_shared<FftPlan>();
    //    plan_fwd_->n = n_;
    //    plan_inv_->n = n_;
    //}

    //void RealFft::forward(const float* in, Complex32* out) const {
    //    if (!in || !out) throw std::runtime_error("RealFft::forward: null pointer");

    //    shape_t shape{ n_ };
    //    stride_t stride_in{ static_cast<ptrdiff_t>(sizeof(float)) };
    //    stride_t stride_out{ static_cast<ptrdiff_t>(sizeof(std::complex<float>)) };
    //    std::size_t axis = 0;

    //    //std::vector<double> vin(n_);
    //    //for (std::size_t i = 0; i < n_; ++i)
    //    //    vin[i] = static_cast<double>(in[i]);

    //    auto vin = in;
    //    auto vout = out;

    //    //std::vector<Complex32> vout(n_freqs_);
    //    r2c(shape, stride_in, stride_out,
    //        axis, FORWARD,
    //        vin, out, (float)1.0);

    //    //for (std::size_t i = 0; i < n_freqs_; ++i) {
    //    //    out[i] = Complex32(static_cast<float>(vout[i].real()),
    //    //        static_cast<float>(vout[i].imag()));
    //    //}
    //}

    //void RealFft::inverse(const Complex32* in, float* out) const {
    //    if (!in || !out) throw std::runtime_error("RealFft::inverse: null pointer");

    //    shape_t shape{ n_ };
    //    stride_t stride_in{ static_cast<ptrdiff_t>(sizeof(std::complex<float>)) };
    //    stride_t stride_out{ static_cast<ptrdiff_t>(sizeof(float)) };
    //    std::size_t axis = 0;

    //    //std::vector<std::complex<double>> vin(n_freqs_);
    //    //for (std::size_t i = 0; i < n_freqs_; ++i) {
    //    //    vin[i] = std::complex<double>(in[i].real(), in[i].imag());
    //    //}
    //    //std::vector<double> vout(n_);

    //    auto vin = in;
    //    auto vout = out;
    //    c2r(shape, stride_in, stride_out,
    //        axis, BACKWARD,
    //        vin, vout, (float)1.0);

    //    //for (std::size_t i = 0; i < n_; ++i) {
    //    //    out[i] = static_cast<float>(vout[i]);
    //    //}

    //    // double scale = 1.0 / static_cast<double>(n_);
    //    // out[i] = static_cast<float>(vout[i] * scale);
    //}

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
                    acc += (xv.r * pv.r + xv.i * pv.i);
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
        , fft_forward_(fft_size,false)
        , fft_inverse_(fft_size, true)
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

    // input_noisy_ch  : time-domain frame, length = frame_size (hop_size_)
    // spectrum: complex spectrum, length = freq_size_
    void DFState::analysis(const float* input_noisy_ch, Complex32* spectrum) {
        if (!input_noisy_ch || !spectrum) {
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
        auto& buf = time_buf_;

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

        // ---- 2) Second part of the window on the new input_noisy_ch frame ----
        // Rust:
        // for ((&y, &w), x) in input_noisy_ch.iter().zip(window_second.iter()).zip(buf_second.iter_mut()) {
        //     *x = y * w;
        // }
        //
        // window_second starts at window_[win_h], length = h
        const float* win2 = window_.data() + win_h;
        for (std::size_t i = 0; i < h; ++i) {
            buf[win_h + i] = input_noisy_ch[i] * win2[i];
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

        // ---- 4) Copy input_noisy_ch to analysis_mem tail for next iteration ----
        // Rust:
        // for (x, &y) in state.analysis_mem[analysis_split..].iter_mut().zip(input_noisy_ch) {
        //     *x = y
        // }
        //
        // i.e. analysis_mem_[analysis_split .. analysis_split + h] = input_noisy_ch[0 .. h]
        for (std::size_t i = 0; i < h; ++i) {
            analysis_mem_[analysis_split + i] = input_noisy_ch[i];
        }

        //{
        //    const std::size_t len_mem = analysis_mem_.size();
        //    const std::size_t analysis_split = len_mem - h;

        //    if (analysis_split > 0) {
        //        // Move [h .. len_mem) to [0 .. analysis_split)
        //        std::memmove(analysis_mem_.data(),
        //            analysis_mem_.data() + h,
        //            analysis_split * sizeof(float));
        //    }

        //    // Fill tail with new input
        //    std::memcpy(analysis_mem_.data() + analysis_split,
        //        input_noisy_ch,
        //        h * sizeof(float));
        //}
        // ---- 5) Forward FFT: buf -> spectrum (via freq_buf_) ----
        // Rust:
        // state.fft_forward.process_with_scratch(&mut buf, output, &mut state.analysis_scratch)
//        {
//            std::vector<float> input = { -0.971467, -0.836983, -0.311486, -0.542014, -0.237995, -0.071220, -0.607078, 0.585647, 0.245415, 0.494258,
//0.415047, 0.272217, -0.158734, -0.147301, -0.918045, 0.771090, -0.282587, -0.127091, 0.714520, -0.666487,
//-0.458414, -0.598150, 0.531485, -0.752943, -0.145886, -0.217106, -0.092927, -0.545293, 0.254764, 0.716065,
//-0.945555, 0.219893, -0.557607, 0.885014, -0.651030, -0.833218, 0.033763, 0.118779, -0.799234, 0.521978,
//0.905697, 0.628161, 0.482018, -0.529617, -0.217603, 0.261545, -0.953752, 0.473895, -0.270291, -0.968591,
//0.509615, -0.221260, -0.038057, 0.848077, -0.290924, -0.096964, -0.624794, -0.804215, -0.581622, 0.974861,
//-0.274816, 0.873410, 0.886905, 0.872418, -0.123131, -0.011496, -0.561262, 0.381468, 0.649490, -0.764268,
//0.052787, -0.026374, -0.341202, -0.555603, 0.025748, -0.580355, -0.110662, -0.375707, -0.607700, -0.077212,
//0.349648, 0.420058, -0.363123, -0.525394, -0.135884, 0.802761, -0.403015, -0.955083, -0.656275, -0.648737,
//-0.170041, -0.182081, -0.747231, 0.117589, -0.393214, 0.341794, -0.814152, 0.751148, 0.105244, -0.697393,
//0.286081, -0.998790, 0.491288, 0.013810, 0.850576, -0.061585, -0.404132, 0.504742, 0.794194, -0.205317,
//0.517893, 0.825812, -0.633936, -0.062422, 0.680768, -0.065124, 0.438965, -0.410707, 0.173116, 0.638977,
//0.476335, 0.282812, -0.102703, 0.839189, 0.369105, 0.383393, -0.107805, -0.856198, 0.705368, -0.300538,
//-0.690725, 0.949818, 0.795539, -0.839849, 0.258355, -0.998774, -0.631311, -0.165941, 0.811981, 0.683488,
//0.530055, -0.652618, -0.507509, 0.267257, -0.745532, -0.748487, -0.935376, -0.628042, -0.872742, 0.114836,
//-0.336078, -0.468256, 0.749548, 0.104462, -0.968305, 0.821721, 0.485387, -0.306309, -0.314802, 0.845973,
//0.964045, -0.961616, -0.049390, 0.971605, -0.968269, -0.438088, 0.748545, -0.415504, -0.844375, -0.071921,
//0.278491, 0.110720, 0.018285, 0.326458, -0.429818, 0.826932, -0.633623, 0.866778, -0.878608, -0.395474,
//-0.848014, -0.757833, 0.858385, -0.064795, 0.360095, 0.019444, -0.274730, 0.316735, -0.024120, 0.471183,
//0.260556, 0.713524, 0.963948, 0.338205, -0.797713, 0.331436, 0.218333, -0.160100, 0.829714, -0.464869,
//0.443157, -0.094495, -0.855384, -0.522051, 0.463327, 0.225621, -0.323704, 0.159772, -0.669504, -0.906509,
//-0.135225, -0.773146, 0.213115, 0.966773, 0.910138, -0.593888, -0.401840, 0.657080, 0.619583, -0.659293,
//0.954217, -0.722298, 0.115125, 0.808519, 0.483381, 0.167916, -0.518550, -0.116015, -0.630193, 0.165096,
//0.764598, 0.915701, 0.602705, -0.483910, 0.080125, -0.366774, -0.502355, 0.240039, -0.058784, -0.822536,
//-0.621393, -0.346672, 0.617749, 0.874738, -0.191537, -0.615960, -0.064308, 0.563562, -0.557735, 0.100572,
//-0.238993, -0.977807, -0.052014, 0.576320, 0.651244, 0.600691, -0.837017, -0.522468, 0.814784, -0.450877,
//0.637218, -0.979223, 0.343912, -0.022630, -0.090523, -0.248848, -0.861447, 0.117097, -0.880783, -0.465142,
//0.367951, 0.895349, 0.228887, 0.325783, -0.307877, 0.013306, 0.846911, 0.753519, -0.736941, 0.219727,
//-0.493795, 0.614285, -0.574720, 0.037815, 0.005830, 0.830379, 0.260653, -0.966311, -0.180002, -0.565914,
//0.542177, -0.801677, -0.536030, -0.838662, 0.779317, -0.874563, -0.996738, 0.019183, -0.127939, -0.648687,
//-0.749833, -0.163417, 0.426255, -0.739225, 0.818855, -0.730453, 0.578146, -0.813435, 0.997234, -0.467208,
//0.909940, 0.833616, 0.654100, 0.952050, 0.582283, 0.510624, 0.554261, 0.622704, 0.739952, -0.608229,
//-0.762787, -0.845100, -0.492065, 0.549185, 0.465780, 0.928100, 0.372377, 0.500930, -0.341959, 0.913389,
//-0.811137, -0.127362, 0.825183, -0.706741, 0.956984, 0.995292, -0.571467, -0.922757, -0.060592, 0.928307,
//-0.081507, 0.107249, -0.232993, 0.325593, 0.540802, -0.709142, -0.465789, 0.766690, -0.899310, -0.912429,
//0.583956, 0.476415, -0.932458, -0.925750, -0.024221, -0.055949, 0.021369, 0.116797, 0.041610, -0.820673,
//0.242607, -0.206270, 0.648398, 0.888567, -0.671690, -0.362275, -0.364285, -0.282766, -0.633868, 0.385194,
//-0.380873, 0.411937, -0.817926, -0.912108, 0.572620, 0.275653, 0.343479, 0.668362, -0.651477, 0.601395,
//0.124695, -0.478943, 0.855006, -0.463945, 0.688884, -0.929656, 0.226222, -0.755989, 0.326696, -0.947025,
//-0.858488, 0.571966, -0.590581, -0.152385, 0.452845, 0.179858, 0.142325, 0.177298, -0.196820, 0.769032,
//0.577910, -0.211884, 0.813795, 0.754224, -0.997556, 0.811309, -0.921242, 0.589084, 0.818035, 0.199297,
//0.317996, 0.631513, -0.769960, -0.697869, -0.211535, 0.213515, 0.086348, -0.762381, 0.793136, 0.145379,
//-0.046641, 0.484618, 0.149384, 0.063954, 0.080567, 0.722821, 0.084161, 0.490256, -0.080386, 0.432561,
//-0.922209, -0.734571, 0.922147, -0.616726, 0.886835, 0.983070, 0.044589, -0.955561, -0.953102, 0.077182,
//0.397012, 0.364972, 0.510026, 0.513964, -0.781604, 0.288017, -0.613166, -0.306014, -0.630512, -0.004291,
//0.659416, -0.501629, -0.152617, -0.811007, 0.836721, 0.999932, -0.849293, 0.612553, -0.723852, -0.810584,
//-0.423389, -0.007234, 0.374100, -0.021420, -0.515320, -0.830416, 0.216000, 0.219004, 0.972257, 0.190006,
//0.740663, 0.183053, 0.905174, 0.724725, -0.846939, 0.208451, 0.006366, -0.746392, -0.854587, 0.123686,
//-0.862781, 0.649791, -0.577687, -0.273038, -0.565458, -0.693919, -0.197746, 0.586409, -0.263490, -0.324505,
//0.264471, 0.582380, 0.943386, -0.025029, 0.880545, -0.646640, -0.162000, -0.840282, 0.732917, -0.032176,
//-0.592403, -0.249518, 0.324437, -0.427935, -0.623063, 0.256742, -0.457705, 0.707110, -0.905605, 0.910543,
//0.534479, -0.177516, -0.748573, 0.117003, -0.405428, -0.176639, 0.089915, -0.857718, 0.398850, 0.190791,
//0.350861, -0.609077, -0.000873, 0.050886, 0.459360, -0.183503, 0.690458, 0.973291, 0.965651, -0.096377,
//0.329965, -0.258764, 0.060945, 0.456813, 0.019969, -0.482056, 0.424036, -0.085552, 0.785535, -0.602166,
//-0.734057, -0.040855, 0.267284, -0.845679, -0.694553, -0.950711, -0.744441, -0.331005, 0.655869, -0.979860,
//0.437976, 0.637977, 0.639022, 0.113144, 0.233416, -0.939753, -0.256317, -0.310878, 0.192462, -0.243173,
//0.598919, -0.460268, -0.217693, 0.356946, -0.406205, -0.910436, -0.464145, -0.095719, 0.660053, 0.482938,
//0.658818, -0.091352, -0.758629, -0.208485, -0.055278, 0.439421, 0.608320, 0.666662, -0.580186, -0.368142,
//0.771117, 0.355531, -0.318950, -0.946592, -0.379599, 0.828547, 0.834639, 0.730545, 0.729376, -0.360270,
//0.709274, 0.039941, 0.611675, 0.422268, 0.285893, 0.348130, -0.707483, 0.527615, -0.559241, -0.496046,
//0.952092, -0.343409, 0.513119, 0.513481, -0.503982, -0.530909, -0.642582, 0.941917, -0.594957, -0.127164,
//0.576785, -0.818295, 0.987154, 0.755168, -0.420763, 0.002353, -0.283069, -0.725613, -0.013352, 0.856205,
//-0.964370, 0.784129, -0.806056, 0.004589, -0.674321, -0.405592, 0.903680, 0.365870, -0.083558, 0.897307,
//0.359782, 0.111979, -0.288194, 0.976365, -0.645243, -0.991172, 0.891361, 0.647209, 0.957308, -0.122321,
//0.293796, -0.225580, 0.454782, 0.783318, -0.928559, -0.107938, -0.475008, -0.205495, 0.836784, -0.641184,
//-0.608986, -0.407484, 0.221304, -0.331367, 0.164235, -0.017137, 0.984999, 0.179944, -0.333812, 0.217325,
//0.119415, 0.493675, 0.470820, -0.662007, -0.044982, -0.067785, -0.230330, -0.974924, -0.788825, -0.986183,
//0.283370, -0.940675, 0.467706, -0.558148, -0.154818, -0.063745, 0.723158, -0.239840, -0.702407, -0.308945,
//0.376020, 0.633283, 0.501382, 0.107699, -0.221780, 0.959323, -0.745902, -0.276087, -0.729933, 0.206264,
//-0.595277, -0.755115, 0.439849, 0.112572, -0.416361, -0.358474, 0.403793, -0.752076, 0.029789, -0.938490,
//0.557640, 0.291792, -0.023249, -0.815858, 0.406008, 0.938825, -0.222955, 0.322753, 0.885972, -0.259911,
//-0.003748, -0.042207, 0.651335, -0.244993, -0.795041, 0.850163, -0.112597, 0.713237, 0.121947, -0.585279,
//-0.994654, 0.800968, 0.187303, 0.198210, 0.206024, 0.576055, 0.230657, -0.397966, 0.584231, 0.841582,
//0.197810, -0.451430, -0.166472, 0.376508, -0.509758, -0.149020, -0.326548, -0.719212, -0.039985, -0.372216,
//-0.665271, 0.586050, 0.295278, -0.101140, 0.735531, 0.670876, -0.982409, -0.351729, -0.874151, 0.164681,
//0.540969, 0.147008, -0.335800, 0.261638, -0.183666, -0.854537, 0.208363, -0.451089, -0.360029, -0.513327,
//-0.891132, 0.632845, -0.228126, 0.842219, -0.445247, -0.704732, -0.958891, -0.491316, -0.039993, 0.555152,
//0.293472, -0.872875, -0.355546, 0.652179, -0.917948, -0.056478, 0.772243, 0.515489, 0.186111, -0.786543,
//-0.119904, 0.516440, -0.272371, -0.053200, -0.018184, -0.023125, 0.325546, -0.066408, -0.338325, -0.953862,
//-0.001011, 0.043684, 0.906023, 0.663790, -0.839577, -0.022054, 0.479309, -0.320553, -0.335137, -0.014988,
//-0.407677, -0.916167, 0.291876, 0.240475, -0.846122, 0.450254, -0.627375, -0.824998, 0.348604, 0.273807,
//-0.888803, -0.817860, -0.007260, -0.657587, 0.171432, 0.967710, -0.999544, -0.492329, 0.813943, 0.381600,
//-0.889070, -0.302350, -0.659126, -0.099585, 0.924058, 0.538402, -0.386443, 0.234197, -0.974399, -0.593393,
//-0.928546, -0.557743, -0.335570, 0.643268, -0.934546, 0.420635, -0.245465, 0.198442, 0.784370, 0.135876,
//-0.923415, 0.964456, -0.673418, -0.745127, 0.933095, -0.769638, 0.776860, -0.935355, -0.041139, 0.770368,
//0.261830, -0.933963, -0.031948, 0.892751, 0.538810, -0.201043, -0.570681, -0.784623, -0.504114, 0.669335,
//0.261741, -0.400773, -0.340664, 0.602287, 0.944705, 0.656084, -0.106443, -0.702837, -0.104878, -0.586409,
//-0.111512, 0.461527, 0.787465, -0.385529, -0.180557, -0.009078, 0.132104, -0.067690, -0.154027, 0.318933,
//0.322729, -0.912350, -0.702156, -0.956036, -0.484498, 0.442235, 0.685006, 0.638473, 0.591888, 0.147212,
//0.995293, 0.980544, -0.636622, -0.653022, 0.039792, -0.799880, 0.224783, 0.378136, -0.970402, -0.419425,
//0.006910, -0.983803, -0.241367, 0.984494, 0.795292, 0.284499, -0.249095, 0.135791, -0.962310, -0.877045,
//0.323957, 0.197040, 0.046139, 0.293560, 0.894892, -0.256949, 0.440937, -0.595911, -0.373141, 0.767704,
//-0.594113, -0.565673, 0.772859, 0.894833, 0.290155, -0.249084, -0.259949, 0.116119, 0.156361, -0.340675,
//-0.581966, -0.094199, -0.077673, -0.372114, -0.321775, -0.247741, -0.493430, 0.422530, -0.622811, -0.302132,
//-0.474918, -0.717294, 0.618780, 0.244131, -0.646266, 0.856835, -0.799863, -0.519203, -0.586155, -0.203217,
//0.133456, -0.633873, 0.943298, -0.754744, -0.419991, 0.558337, 0.818532, 0.638954, 0.537355, 0.390483,
//            };
//            std::vector<Complex32> output(freq_size_);
//
//            fft_forward_.forward(input.data(), output.data());
//        }
        fft_forward_.forward(buf, freq_buf_);
        // ---- 6) Apply normalization in analysis only (Rust: *x *= norm) ----
        // Rust:
        // let norm = state.wnorm;
        // for x in output.iter_mut() { *x *= norm; }
        for (std::size_t i = 0; i < freq_size_; ++i) {
            //freq_buf_[i] *= wnorm_;
            freq_buf_[i].r *= wnorm_;
            freq_buf_[i].i *= wnorm_;
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
//
//        {
//            std::vector<Complex32> input = { Complex32(-18.995966f, 0.000000f), Complex32(10.338626f, 5.593056f), Complex32(6.882241f, 5.345215f), Complex32(-5.135488f, 2.516920f), Complex32(-8.982934f, -24.808613f),
//Complex32(0.509266f, -11.853664f), Complex32(-5.451043f, -14.765564f), Complex32(-2.941125f, 2.734736f), Complex32(7.058029f, 19.129047f), Complex32(15.092484f, 6.299122f),
//Complex32(-22.805189f, 10.267513f), Complex32(-13.786385f, 8.168575f), Complex32(7.019880f, 2.565522f), Complex32(-26.985023f, 12.558376f), Complex32(8.980804f, -31.038933f),
//Complex32(-19.690868f, 4.975602f), Complex32(-10.238283f, -2.954839f), Complex32(-17.261032f, -9.257509f), Complex32(-19.660198f, 18.899908f), Complex32(-9.572515f, -10.586405f),
//Complex32(9.674056f, -16.640806f), Complex32(-3.162091f, 34.675365f), Complex32(-10.049966f, -6.023996f), Complex32(-15.119382f, 5.451357f), Complex32(-11.212821f, -3.324176f),
//Complex32(-6.034213f, -5.211294f), Complex32(0.167655f, -10.007454f), Complex32(11.013232f, -0.890051f), Complex32(9.458846f, 2.932532f), Complex32(-1.189357f, -23.859108f),
//Complex32(1.514477f, -8.809504f), Complex32(-1.831510f, 28.502365f), Complex32(-4.497005f, 13.193583f), Complex32(-19.316200f, 7.150238f), Complex32(-25.160006f, 11.000460f),
//Complex32(-22.522467f, -17.077656f), Complex32(-3.765234f, 16.581554f), Complex32(2.965302f, 11.929659f), Complex32(-1.976437f, 5.349058f), Complex32(6.725644f, -11.498652f),
//Complex32(-10.378979f, 13.149912f), Complex32(6.879305f, 17.011826f), Complex32(17.805641f, -30.964178f), Complex32(9.564600f, -6.644442f), Complex32(-28.213036f, 1.717505f),
//Complex32(5.051019f, -7.107278f), Complex32(-9.081902f, 7.423851f), Complex32(2.792136f, -11.694791f), Complex32(-0.576015f, 6.577173f), Complex32(9.656082f, -3.070610f),
//Complex32(5.903786f, 1.519445f), Complex32(24.298958f, 21.224642f), Complex32(-2.894897f, 23.853170f), Complex32(-14.276148f, -15.556356f), Complex32(12.422008f, -5.332211f),
//Complex32(-3.966918f, 0.177342f), Complex32(7.381366f, -6.767553f), Complex32(0.482228f, 15.979431f), Complex32(-9.195655f, 10.497074f), Complex32(34.579945f, -11.346473f),
//Complex32(1.643150f, -20.679638f), Complex32(-5.942488f, -25.753391f), Complex32(-14.575768f, -23.994755f), Complex32(-2.003407f, 17.942839f), Complex32(-11.665757f, -18.259535f),
//Complex32(-1.495786f, 11.228166f), Complex32(-23.411671f, 24.611465f), Complex32(-0.811281f, -13.092474f), Complex32(-15.820556f, -0.945435f), Complex32(2.886477f, 12.563347f),
//Complex32(15.451542f, 13.627192f), Complex32(-17.252775f, -5.314473f), Complex32(3.942097f, -16.919046f), Complex32(20.429501f, -2.358697f), Complex32(15.412832f, 13.697265f),
//Complex32(-6.469134f, 2.541164f), Complex32(19.214581f, -14.862696f), Complex32(-0.019651f, -0.558748f), Complex32(2.073602f, 6.590636f), Complex32(-8.028654f, -15.562874f),
//Complex32(-10.635180f, -14.652043f), Complex32(-9.663191f, -10.919343f), Complex32(6.939662f, 17.889454f), Complex32(-0.693790f, -1.783221f), Complex32(10.844641f, 10.874880f),
//Complex32(-9.372159f, 15.876196f), Complex32(2.478801f, -12.012575f), Complex32(-7.024889f, -2.393691f), Complex32(10.771797f, -4.334272f), Complex32(25.684254f, 26.383865f),
//Complex32(-6.620694f, 3.413881f), Complex32(-6.181921f, 17.728767f), Complex32(-14.935834f, -3.657888f), Complex32(-12.933439f, 7.618006f), Complex32(-7.121202f, 7.253653f),
//Complex32(5.108355f, 10.539791f), Complex32(-7.692217f, 2.177085f), Complex32(-6.011131f, -12.688927f), Complex32(12.999207f, -17.805218f), Complex32(12.982077f, 10.620550f),
//Complex32(-8.658020f, 2.665409f), Complex32(-8.467497f, 15.169535f), Complex32(-7.424043f, 25.021748f), Complex32(-7.469159f, -23.125315f), Complex32(36.263069f, 4.193930f),
//Complex32(-0.839141f, 0.011906f), Complex32(-17.571217f, 7.118480f), Complex32(8.181891f, 24.651699f), Complex32(-1.556839f, 35.336964f), Complex32(9.866760f, -6.243740f),
//Complex32(-2.714465f, 4.995208f), Complex32(10.626816f, 17.183743f), Complex32(30.431728f, 0.412702f), Complex32(8.824977f, -0.733456f), Complex32(15.293226f, -7.438459f),
//Complex32(1.535516f, 3.205484f), Complex32(-7.959931f, -15.720963f), Complex32(11.697523f, 12.084542f), Complex32(-15.329835f, 7.352215f), Complex32(-5.460382f, -13.928609f),
//Complex32(17.240223f, -0.914454f), Complex32(-3.449856f, -16.425959f), Complex32(-1.656821f, -0.904994f), Complex32(-11.184221f, 0.267418f), Complex32(6.457870f, 5.448761f),
//Complex32(1.504107f, -2.416211f), Complex32(-2.683831f, -11.078596f), Complex32(-15.526917f, 9.223602f), Complex32(-0.579926f, -10.384674f), Complex32(-8.008957f, -11.078047f),
//Complex32(1.403600f, 6.898623f), Complex32(-8.072185f, 12.802371f), Complex32(11.116571f, -25.619061f), Complex32(-8.498357f, 5.679674f), Complex32(-1.202536f, 4.953169f),
//Complex32(4.710228f, -10.162991f), Complex32(-8.204884f, -11.774630f), Complex32(-0.889470f, -18.135616f), Complex32(-0.051537f, -11.728650f), Complex32(-7.631756f, -6.816689f),
//Complex32(2.570525f, 5.371425f), Complex32(1.326265f, 2.149452f), Complex32(8.751574f, -14.964953f), Complex32(20.224937f, -11.446556f), Complex32(-30.251144f, 0.018289f),
//Complex32(-15.703173f, -7.648595f), Complex32(20.785261f, -7.814899f), Complex32(31.149700f, -20.172510f), Complex32(-6.989610f, -20.040022f), Complex32(3.914135f, -1.383373f),
//Complex32(-19.510122f, 4.193727f), Complex32(10.808090f, -15.751785f), Complex32(-15.099289f, -24.986160f), Complex32(18.146343f, 1.782876f), Complex32(10.489021f, 19.231720f),
//Complex32(-3.051666f, 30.946665f), Complex32(-9.649801f, 3.265321f), Complex32(-9.234617f, 11.543508f), Complex32(3.063021f, 6.575866f), Complex32(0.029593f, 0.138569f),
//Complex32(0.692862f, 5.378758f), Complex32(-3.814054f, 12.010751f), Complex32(-3.347909f, 3.940201f), Complex32(-12.364731f, -9.701759f), Complex32(1.793211f, -2.520886f),
//Complex32(-0.920295f, 16.687391f), Complex32(-8.965861f, 1.012744f), Complex32(7.207948f, 3.410513f), Complex32(-2.388365f, 13.816368f), Complex32(-2.114711f, 1.434843f),
//Complex32(26.578205f, -3.240787f), Complex32(9.583123f, 10.657747f), Complex32(-16.932392f, -5.269003f), Complex32(-3.755199f, 12.889533f), Complex32(6.202152f, 3.881835f),
//Complex32(3.164091f, 12.912457f), Complex32(27.828827f, 6.606972f), Complex32(12.774553f, 13.788788f), Complex32(18.584026f, -16.473139f), Complex32(21.837456f, 15.820013f),
//Complex32(-15.372646f, -18.919603f), Complex32(14.592844f, -12.417215f), Complex32(-12.070583f, -11.765101f), Complex32(1.479272f, -20.471727f), Complex32(15.089823f, 8.306201f),
//Complex32(2.831573f, -8.640285f), Complex32(13.297886f, -17.629642f), Complex32(-5.088376f, -5.271697f), Complex32(-3.641232f, -2.138423f), Complex32(-13.101028f, 0.656264f),
//Complex32(-14.272242f, -10.546221f), Complex32(9.070318f, 10.162844f), Complex32(-14.468657f, -2.350607f), Complex32(-7.866366f, 1.703583f), Complex32(-4.564072f, 10.517690f),
//Complex32(-10.859041f, 20.148441f), Complex32(5.879182f, -6.403038f), Complex32(-11.732446f, -5.971218f), Complex32(-15.180093f, -9.594255f), Complex32(4.438309f, -17.603077f),
//Complex32(17.903725f, 11.172745f), Complex32(-3.057494f, 16.506020f), Complex32(2.237533f, 11.135016f), Complex32(7.717194f, 8.837324f), Complex32(-14.991104f, 4.794185f),
//Complex32(2.608862f, -2.864330f), Complex32(-5.043706f, -4.741631f), Complex32(-20.038311f, -12.314047f), Complex32(21.892956f, 6.853544f), Complex32(-5.408468f, -8.449220f),
//Complex32(-14.653867f, 2.696403f), Complex32(1.230082f, -21.142399f), Complex32(0.209244f, 1.199733f), Complex32(16.473602f, -11.669542f), Complex32(-1.134071f, 5.097834f),
//Complex32(1.851458f, 16.773521f), Complex32(18.367908f, 12.646514f), Complex32(6.792896f, -8.224891f), Complex32(-13.753762f, -11.722740f), Complex32(-7.651226f, 14.426937f),
//Complex32(-2.646406f, 2.798453f), Complex32(1.428522f, -18.979124f), Complex32(-6.032392f, -2.967410f), Complex32(-4.402033f, -6.668971f), Complex32(-10.162131f, -15.168908f),
//Complex32(-6.809722f, -12.585546f), Complex32(11.841277f, -4.180064f), Complex32(5.339121f, 4.478897f), Complex32(3.682972f, -10.310080f), Complex32(19.703041f, -21.236650f),
//Complex32(13.855760f, 6.997259f), Complex32(11.881507f, -9.672845f), Complex32(25.854568f, -6.564649f), Complex32(8.067533f, 12.211246f), Complex32(-6.389883f, -24.183140f),
//Complex32(-10.429645f, 6.007164f), Complex32(3.472137f, 10.159141f), Complex32(16.531277f, -8.419689f), Complex32(21.827400f, 17.178745f), Complex32(-43.377605f, -11.777424f),
//Complex32(5.316872f, 9.833263f), Complex32(-20.616283f, -6.426792f), Complex32(13.955218f, -11.110432f), Complex32(13.544878f, 10.557861f), Complex32(17.037195f, -10.729316f),
//Complex32(17.567375f, 16.427076f), Complex32(-6.349456f, -1.692344f), Complex32(-5.782385f, 13.018527f), Complex32(2.872228f, -4.943972f), Complex32(10.866231f, -7.952844f),
//Complex32(4.543845f, 4.223866f), Complex32(-15.011513f, 0.332499f), Complex32(38.855206f, -1.667819f), Complex32(-7.818738f, -3.548594f), Complex32(12.639758f, 1.809979f),
//Complex32(31.194618f, -8.602148f), Complex32(-8.414973f, -0.846654f), Complex32(-10.470600f, 7.259093f), Complex32(-5.101623f, 12.783716f), Complex32(13.763848f, 20.656771f),
//Complex32(3.038507f, 8.112400f), Complex32(-6.550691f, -0.180262f), Complex32(1.379949f, 4.677025f), Complex32(17.077126f, -1.429615f), Complex32(-0.277019f, -7.489450f),
//Complex32(-16.030849f, -10.670128f), Complex32(-2.169555f, 6.919556f), Complex32(-3.135915f, 5.846470f), Complex32(6.028348f, -2.829443f), Complex32(21.594534f, -4.200839f),
//Complex32(-8.844970f, 3.624506f), Complex32(10.323676f, -2.935778f), Complex32(1.094913f, 0.708078f), Complex32(3.720529f, -4.420267f), Complex32(-18.406927f, 10.868652f),
//Complex32(-5.811287f, 12.596964f), Complex32(9.868202f, 3.437005f), Complex32(-15.120626f, -26.071249f), Complex32(-12.548417f, 2.456870f), Complex32(10.117840f, 11.361111f),
//Complex32(-9.189079f, -6.419964f), Complex32(-8.496268f, -2.266954f), Complex32(23.784798f, -4.089324f), Complex32(22.631866f, -37.044483f), Complex32(3.794535f, 4.006637f),
//Complex32(-20.400591f, -5.373325f), Complex32(-2.166588f, -19.721756f), Complex32(-16.041332f, -7.027935f), Complex32(-17.075874f, 13.148663f), Complex32(3.170208f, -15.049601f),
//Complex32(15.188181f, 12.025333f), Complex32(-0.785129f, 1.662568f), Complex32(7.720432f, 14.137958f), Complex32(33.917953f, -12.032286f), Complex32(0.517809f, 3.968319f),
//Complex32(-14.887161f, -12.642902f), Complex32(5.009511f, -13.339349f), Complex32(-4.515266f, 10.880301f), Complex32(-16.130795f, 5.310618f), Complex32(-6.124988f, 26.341322f),
//Complex32(-30.012829f, 10.198683f), Complex32(18.575905f, 26.482126f), Complex32(-5.043038f, 7.487442f), Complex32(-15.325748f, 11.691898f), Complex32(-18.883690f, 6.531654f),
//Complex32(9.877169f, 6.325613f), Complex32(-6.977391f, 7.182150f), Complex32(18.251280f, 9.743962f), Complex32(-3.114694f, 10.520168f), Complex32(-1.139370f, 5.881979f),
//Complex32(-3.991862f, -16.483402f), Complex32(-3.975019f, -10.149296f), Complex32(-15.428574f, 3.222096f), Complex32(3.315690f, 5.829948f), Complex32(-10.286578f, -16.314087f),
//Complex32(10.319742f, 7.527228f), Complex32(-5.623059f, -13.720368f), Complex32(-16.787989f, 7.165363f), Complex32(7.685066f, -2.145141f), Complex32(-11.887889f, -0.931315f),
//Complex32(-4.347286f, -5.263075f), Complex32(-20.508415f, 5.143772f), Complex32(-25.398535f, 28.613197f), Complex32(9.350100f, 4.295239f), Complex32(7.859222f, -2.733483f),
//Complex32(-8.947104f, 6.704062f), Complex32(0.395180f, 14.963646f), Complex32(-9.106251f, -14.306326f), Complex32(7.682825f, 12.262165f), Complex32(-9.925310f, -27.755928f),
//Complex32(-9.762703f, 4.795362f), Complex32(26.896976f, -22.915592f), Complex32(-8.901529f, -11.034403f), Complex32(16.397747f, 23.429960f), Complex32(-11.957567f, -24.274879f),
//Complex32(7.344589f, -6.147251f), Complex32(-2.519200f, 2.456314f), Complex32(2.346200f, 1.664595f), Complex32(10.928629f, -14.788579f), Complex32(-6.398448f, 19.274694f),
//Complex32(9.406891f, -21.786644f), Complex32(-9.787096f, 4.426725f), Complex32(3.993280f, 2.311984f), Complex32(-2.639194f, -4.739604f), Complex32(-11.847584f, 0.289275f),
//Complex32(12.104115f, -0.886709f), Complex32(-4.536924f, 0.576758f), Complex32(-1.790301f, 20.495424f), Complex32(-21.563150f, -0.893249f), Complex32(8.287963f, 7.631278f),
//Complex32(-9.189438f, -0.546430f), Complex32(-4.101891f, 13.473387f), Complex32(-4.020948f, 13.410675f), Complex32(-29.595257f, -4.825782f), Complex32(-27.116772f, 11.917549f),
//Complex32(17.226000f, 1.891850f), Complex32(4.716357f, -9.223271f), Complex32(-12.169151f, 2.183771f), Complex32(9.560408f, -18.037361f), Complex32(-0.332184f, 15.814116f),
//Complex32(12.640635f, -5.429434f), Complex32(-6.280210f, 4.653987f), Complex32(8.324470f, -6.287958f), Complex32(-15.620733f, 14.616656f), Complex32(6.893698f, -25.428589f),
//Complex32(9.270827f, 7.863677f), Complex32(11.893032f, 6.251169f), Complex32(34.853989f, 1.605757f), Complex32(-17.378460f, -1.036637f), Complex32(-11.253311f, -4.626488f),
//Complex32(0.049177f, -11.061677f), Complex32(-6.603865f, -3.236461f), Complex32(-6.721542f, 1.391113f), Complex32(-10.412097f, 5.926887f), Complex32(12.186501f, -2.009395f),
//Complex32(-12.350296f, -18.665028f), Complex32(-8.123112f, -1.121279f), Complex32(1.796694f, -10.128237f), Complex32(-7.711598f, 0.501770f), Complex32(1.081403f, 19.216635f),
//Complex32(-21.802845f, 2.126010f), Complex32(-23.889170f, 7.853556f), Complex32(1.629589f, 19.411766f), Complex32(13.170692f, -3.022301f), Complex32(-16.507830f, -2.426587f),
//Complex32(9.839910f, -1.633538f), Complex32(16.979591f, -15.577158f), Complex32(-10.906731f, 9.377270f), Complex32(5.992490f, 21.842760f), Complex32(-21.610857f, -13.001671f),
//Complex32(10.963738f, 15.167917f), Complex32(-8.915173f, 4.286758f), Complex32(-14.617117f, -3.315684f), Complex32(4.913463f, -26.569767f), Complex32(-2.881917f, -11.210131f),
//Complex32(-9.145579f, 0.470936f), Complex32(-3.191015f, -4.003400f), Complex32(-14.388769f, 8.836884f), Complex32(-2.650331f, 11.915078f), Complex32(-16.079889f, 4.241789f),
//Complex32(-13.827288f, 13.468589f), Complex32(-12.917662f, 9.134556f), Complex32(-7.280602f, -10.690281f), Complex32(11.841969f, -11.557888f), Complex32(2.595768f, 0.499313f),
//Complex32(-9.244325f, -11.596861f), Complex32(-2.418322f, 10.727063f), Complex32(4.448723f, 7.263704f), Complex32(-2.789968f, 13.557726f), Complex32(-15.724458f, -7.524868f),
//Complex32(4.032002f, 5.145826f), Complex32(20.719978f, 23.992744f), Complex32(-3.726792f, -19.583878f), Complex32(11.900520f, -9.058455f), Complex32(3.679649f, -5.857621f),
//Complex32(-15.766325f, 7.837670f), Complex32(8.099175f, 14.706988f), Complex32(-10.375119f, -9.046114f), Complex32(5.586123f, -7.736117f), Complex32(13.901676f, -5.851308f),
//Complex32(-5.710063f, -9.370528f), Complex32(-28.694494f, 1.597398f), Complex32(13.593324f, 11.325317f), Complex32(4.926898f, 3.526036f), Complex32(10.534347f, 8.569002f),
//Complex32(-11.064763f, -30.311649f), Complex32(-24.461433f, -3.545957f), Complex32(0.915052f, 14.348822f), Complex32(-6.258981f, -13.702570f), Complex32(2.006510f, -6.917696f),
//Complex32(4.134982f, -21.888144f), Complex32(6.945263f, 4.280350f), Complex32(25.656246f, -1.043373f), Complex32(0.538037f, 21.536907f), Complex32(-18.896849f, 7.520863f),
//Complex32(-15.654448f, -15.161401f), Complex32(13.825020f, -7.257401f), Complex32(-2.004929f, 23.799156f), Complex32(-9.731557f, -9.723558f), Complex32(9.145565f, -11.667186f),
//Complex32(2.406937f, -14.490524f), Complex32(-27.836166f, 17.427849f), Complex32(7.173460f, -18.401205f), Complex32(-16.274506f, -18.573473f), Complex32(23.932053f, 7.507150f),
//Complex32(-2.196497f, 1.420769f), Complex32(17.525755f, -21.470552f), Complex32(7.567956f, -3.368389f), Complex32(5.553819f, -24.190039f), Complex32(4.820426f, -5.975075f),
//Complex32(-19.976944f, 29.102270f), Complex32(3.205982f, -4.754959f), Complex32(7.758222f, -0.537813f), Complex32(4.195755f, -10.594639f), Complex32(11.581671f, 4.269434f),
//Complex32(-13.243146f, -31.970602f), Complex32(29.078300f, -7.674439f), Complex32(-3.035251f, -11.765720f), Complex32(1.022672f, -5.123981f), Complex32(9.362080f, 8.822839f),
//Complex32(3.480300f, -16.360479f), Complex32(-4.144984f, 7.554344f), Complex32(-12.904140f, -15.244960f), Complex32(-5.238771f, 12.975737f), Complex32(5.579574f, 15.281708f),
//Complex32(-3.235431f, 3.112813f), Complex32(7.489837f, 8.981285f), Complex32(6.580761f, -0.663384f), Complex32(1.451899f, 14.418619f), Complex32(4.835931f, 6.998347f),
//Complex32(-1.031672f, -1.139980f), Complex32(-3.048790f, -11.511011f), Complex32(-8.214084f, -10.678775f), Complex32(-12.326331f, 3.587109f), Complex32(-31.430874f, 5.771149f),
//Complex32(-13.430630f, -12.848139f), Complex32(-5.472738f, -3.368484f), Complex32(-9.210053f, -5.933405f), Complex32(29.860775f, 22.513193f), Complex32(-1.752654f, -13.108831f),
//Complex32(12.093386f, 0.000000f), };
//
//            std::vector<float> output(fft_size_, 0.0f);
//            fft_inverse_.inverse(input.data(), output.data());
//        }
        // 1) Inverse FFT
        // Rust:
        // match state.fft_inverse.process_with_scratch(input_noisy_ch, &mut x, &mut state.synthesis_scratch) { ... }
        // Here we assume fft_.inverse throws or handles internal errors.
        fft_inverse_.inverse(spectrum, x);

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
    void DFState::compute_band_corr(float* out_feat, const float* spectrum_interleaved) {
        // Initialize output bands to 0
        for (std::size_t i = 0; i < nb_erb_; ++i) {
            out_feat[i] = 0.0f;
        }

        std::size_t bcsum = 0;
        for (std::size_t b = 0; b < nb_erb_; ++b) {
            std::size_t band_size = erb_[b];
            if (band_size == 0) {
                continue;
            }

            float k = 1.0f / static_cast<float>(band_size);  // Weight factor for each band
            // Auto-correlation of spectrum with itself: Re*Re + Im*Im
            for (std::size_t j = 0; j < band_size; ++j) {
                std::size_t idx = bcsum + j;  // complex bin index

                // Pointer to [Re, Im] of this bin in interleaved layout
                const float* re_im = spectrum_interleaved + 2 * idx;
                float re = re_im[0];
                float im = re_im[1];

                // Directly accumulate the weighted correlation
                out_feat[b] += (re * re + im * im) * k;  // Weighted sum (magnitude squared)
            }

            bcsum += band_size;
        }
    }

    void DFState::log_transform(float* out_feat) {
        const float eps = 1e-10f;  // Prevent log(0)
        for (std::size_t i = 0; i < nb_erb_; ++i) {
            out_feat[i] = 10.0f * std::log10(out_feat[i] + eps);  // 10 * log10(x)
        }
    }

    void DFState::band_mean_norm_erb(float* out_feat, float alpha) {
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

    void DFState::feat_erb(const float* spectrum_interleaved, float alpha, float* out_feat) {
        if (!spectrum_interleaved || !out_feat) {
            throw std::runtime_error("DFState::feat_erb: null pointer");
        }

        if (erb_.size() != nb_erb_ || mean_norm_state_.size() != nb_erb_) {
            throw std::runtime_error("DFState::feat_erb: size mismatch");
        }

        // Step 1: Compute band correlations (auto-correlation of the spectrum)
        compute_band_corr(out_feat, spectrum_interleaved);

        // Step 2: Apply log10 transformation
        log_transform(out_feat);

        // Step 3: Apply exponential mean normalization
        band_mean_norm_erb(out_feat, alpha);
    }

    //// spectrum_interleaved: [re0, im0, re1, im1, ...],
    //// length = 2 * freq_bins (at least sum(erb_)).
    //void DFState::feat_erb(const float* spectrum_interleaved,
    //    float alpha,
    //    float* out_feat) {
    //    if (!spectrum_interleaved || !out_feat) {
    //        throw std::runtime_error("DFState::feat_erb: null pointer");
    //    }

    //    if (erb_.size() != nb_erb_ || mean_norm_state_.size() != nb_erb_) {
    //        throw std::runtime_error("DFState::feat_erb: size mismatch");
    //    }

    //    // Initialize output bands to 0
    //    for (std::size_t i = 0; i < nb_erb_; ++i) {
    //        out_feat[i] = 0.0f;
    //    }

    //    std::size_t bcsum = 0;
    //    for (std::size_t b = 0; b < nb_erb_; ++b) {
    //        std::size_t band_size = erb_[b];
    //        if (band_size == 0) {
    //            continue;
    //        }

    //        float acc = 0.0f;

    //        // Auto-correlation of spectrum with itself: Re*Re + Im*Im
    //        for (std::size_t j = 0; j < band_size; ++j) {
    //            std::size_t idx = bcsum + j;  // complex bin index

    //            // Pointer to [Re, Im] of this bin in interleaved layout
    //            const float* re_im = spectrum_interleaved + 2 * idx;
    //            float re = re_im[0];
    //            float im = re_im[1];

    //            acc += re * re + im * im;
    //        }

    //        float k = 1.0f / static_cast<float>(band_size);
    //        out_feat[b] += acc * k;  // average per band

    //        bcsum += band_size;
    //    }

    //    // 10 * log10(out + 1e-10)
    //    const float eps = 1e-10f;
    //    for (std::size_t i = 0; i < nb_erb_; ++i) {
    //        out_feat[i] = 10.0f * std::log10(out_feat[i] + eps);
    //    }

    //    // band_mean_norm_erb(output, &mut self.mean_norm_state, alpha)
    //    for (std::size_t i = 0; i < nb_erb_; ++i) {
    //        float& x = out_feat[i];
    //        float& s = mean_norm_state_[i];

    //        // Exponential moving average of log-magnitude
    //        s = x * (1.0f - alpha) + s * alpha;

    //        // Mean normalization and scaling by 1/40
    //        x -= s;
    //        x /= 40.0f;
    //    }
    //}

    //void DFState::feat_cplx(const Complex32* spectrum,
    //    float alpha,
    //    Complex32* out_feat) {
    //    if (!spectrum || !out_feat) {
    //        throw std::runtime_error("DFState::feat_cplx: null pointer");
    //    }

    //    // In Rust: output.clone_from_slice(input_noisy_ch);
    //    // Here we copy spectrum to out_feat first.
    //    std::size_t nb_df = unit_norm_state_.size();
    //    for (std::size_t i = 0; i < nb_df; ++i) {
    //        out_feat[i] = spectrum[i];
    //    }

    //    // band_unit_norm(output, &mut self.unit_norm_state, alpha)
    //    //
    //    // Rust:
    //    // for (x, s) in xs.iter_mut().zip(state.iter_mut()) {
    //    //     *s = x.norm() * (1. - alpha) + *s * alpha;
    //    //     *x /= s.sqrt();
    //    // }
    //    for (std::size_t i = 0; i < nb_df; ++i) {
    //        Complex32& x = out_feat[i];
    //        float& s = unit_norm_state_[i];

    //        // x.norm() in Rust (num_complex) is magnitude: sqrt(re^2 + im^2)
    //        float mag = std::abs(x);

    //        // Exponential moving average of magnitude
    //        s = mag * (1.0f - alpha) + s * alpha;

    //        // Divide by sqrt(s); guard zero to avoid NaN
    //        float denom = std::sqrt(std::max(s, 0.0f));
    //        if (denom > 0.0f) {
    //            x /= denom;
    //        }
    //        else {
    //            // If s is zero (e.g. all inputs are exactly zero), output zero
    //            x = Complex32(0.0f, 0.0f);
    //        }
    //    }
    //}

    // spectrum_interleaved: [re0, im0, re1, im1, ...],
    // length = 2 * nb_df (nb_df = unit_norm_state_.size()).
    void DFState::feat_cplx(const float* spectrum_interleaved,
        float alpha,
        float* out_feat) 
    {
        if (!spectrum_interleaved || !out_feat) {
            throw std::runtime_error("DFState::feat_cplx: null pointer");
        }

        std::size_t nb_df = unit_norm_state_.size();

        // Step 1: Copy the interleaved spectrum into out_feat as [Re0, Im0, Re1, Im1, ...]
        for (std::size_t i = 0; i < nb_df; ++i) {
            const float* re_im = spectrum_interleaved + 2 * i;
            out_feat[2 * i] = re_im[0];  // real part
            out_feat[2 * i + 1] = re_im[1];  // imaginary part
        }

        // Step 2: Apply unit normalization on the output feature array
        for (std::size_t i = 0; i < nb_df; ++i) {
            float& re = out_feat[2 * i];
            float& im = out_feat[2 * i + 1];
            float& s = unit_norm_state_[i];

            // Compute magnitude: sqrt(re^2 + im^2)
            float mag = std::sqrt(re * re + im * im);

            // Apply exponential moving average on magnitude
            s = mag * (1.0f - alpha) + s * alpha;

            // Normalize by dividing by the square root of s
            float denom = std::sqrt(std::max(s, 0.0f)); // Prevent sqrt of negative values
            if (denom > 0.0f) {
                // Normalize real and imaginary parts by the magnitude
                re /= denom;
                im /= denom;
            }
            else {
                // If s is zero (all inputs are zero), set output to zero
                re = 0.0f;
                im = 0.0f;
            }
        }
    }

    //void DFState::apply_mask(Complex32* spectrum, const float* gains) {
    //    if (!spectrum || !gains) {
    //        throw std::runtime_error("DFState::apply_mask: null pointer");
    //    }
    //    std::size_t bcsum = 0;
    //    for (std::size_t i = 0; i < nb_erb_; ++i) {
    //        std::size_t band_size = erb_[i];
    //        float g = gains[i];

    //        for (std::size_t j = 0; j < band_size; ++j) {
    //            std::size_t idx = bcsum + j;

    //            if (idx >= freq_size_) {
    //                break;
    //            }
    //            spectrum[idx] *= g;
    //        }
    //        bcsum += band_size;
    //    }
    //}
    // spectrum_interleaved: length = 2 * freq_size_
    // layout: [re0, im0, re1, im1, ...]
    void DFState::apply_mask(float* spectrum_interleaved, const float* gains) {
        if (!spectrum_interleaved || !gains) {
            throw std::runtime_error("DFState::apply_mask: null pointer");
        }

        std::size_t bcsum = 0;  // accumulated band size
        for (std::size_t i = 0; i < nb_erb_; ++i) {
            std::size_t band_size = erb_[i];
            float g = gains[i];  // gain for the current band

            // Apply gain to each element in the current band
            for (std::size_t j = 0; j < band_size; ++j) {
                std::size_t idx = bcsum + j;  // complex index (bin index)

                if (idx >= freq_size_) {
                    break;
                }

                // For interleaved layout, bin 'idx' occupies two floats: real and imaginary parts
                float* re_im = spectrum_interleaved + 2 * idx;
                re_im[0] *= g;  // real part
                re_im[1] *= g;  // imaginary part
            }
            bcsum += band_size;
        }
    }


    // Calculate the magnitude (absolute value) of a complex number
    static float magnitude(const kiss_fft_cpx& cpx) {
        return std::sqrt(cpx.r * cpx.r + cpx.i * cpx.i);
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
                float n_mag0 = magnitude(noisy[i0]);
                float e_mag0 = magnitude(enh[i0]);
                float ratio0 = e_mag0 / (n_mag0 + eps);
                if (ratio0 < eps)  ratio0 = eps;
                if (ratio0 > 1.0f) ratio0 = 1.0f;
                g[0] = ratio0;

                float n_mag1 = magnitude(noisy[i1]);
                float e_mag1 = magnitude(enh[i1]);
                float ratio1 = e_mag1 / (n_mag1 + eps);
                if (ratio1 < eps)  ratio1 = eps;
                if (ratio1 > 1.0f) ratio1 = 1.0f;
                g[1] = ratio1;

                float n_mag2 = magnitude(noisy[i2]);
                float e_mag2 = magnitude(enh[i2]);
                float ratio2 = e_mag2 / (n_mag2 + eps);
                if (ratio2 < eps)  ratio2 = eps;
                if (ratio2 > 1.0f) ratio2 = 1.0f;
                g[2] = ratio2;

                float n_mag3 = magnitude(noisy[i3]);
                float e_mag3 = magnitude(enh[i3]);
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
            enh[i0].r *= pf[0];
            enh[i0].i *= pf[0];
            enh[i1].r *= pf[1];
            enh[i1].i *= pf[1];
            enh[i2].r *= pf[2];
            enh[i2].i *= pf[2];
            enh[i3].r *= pf[3];
            enh[i3].i *= pf[3];
        }

        // Tail (N % 4 != 0) is ignored, same as Rust chunks_exact(4)
    }


    void DFState::post_filter(const std::vector<Complex32>& noisy,
        std::vector<Complex32>& enh,
        float beta) const {
        const_cast<DFState*>(this)->post_filter_impl(noisy, enh, beta);
    }


} // namespace df
