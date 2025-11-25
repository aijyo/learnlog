#include "df_tract.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace df;

// Helper: reinterpret complex float view into real interleaved buffer [re,im] per bin
static inline void complex_vec_to_interleaved(const std::vector<std::complex<float>>& src,
    float* dst, size_t bins) {
    for (size_t i = 0; i < bins; ++i) {
        dst[2 * i + 0] = src[i].real();
        dst[2 * i + 1] = src[i].imag();
    }
}

static inline void interleaved_to_complex_vec(const float* src,
    std::vector<std::complex<float>>& dst,
    size_t bins) {
    dst.resize(bins);
    for (size_t i = 0; i < bins; ++i) {
        dst[i] = std::complex<float>(src[2 * i + 0], src[2 * i + 1]);
    }
}

// -------------------- DfTractCpp implementation --------------------

DfTractCpp::DfTractCpp(
    const DfTractConfig& cfg)
    : cfg_(cfg)
{
    auto enc_runner = std::make_shared<ModelRunnerShim>(MODEL_ENC);
    auto erb_runner = std::make_shared<ModelRunnerShim>(MODEL_ERB_DEC);
    auto df_runner = std::make_shared<ModelRunnerShim>(MODEL_DF_DEC);

    enc_ = enc_runner;
    erb_dec_ = erb_runner;
    df_dec_ = df_runner;

    n_freqs_ = cfg_.fft_size / 2 + 1;

    // DFState::new(sr, fft_size, hop_size, nb_erb, min_nb_erb_freqs)
    for (size_t c = 0; c < cfg_.ch; ++c) {
        auto state = std::make_shared<DFState>(
            cfg_.sr, cfg_.fft_size, cfg_.hop_size,
            cfg_.nb_erb, cfg_.min_nb_erb_freqs);
        state->init_norm_states(cfg_.nb_df);
        df_states_.push_back(std::move(state));
    }
}

void DfTractCpp::init() {
    const size_t ch = cfg_.ch;
    const size_t n_freqs = n_freqs_;

    const size_t y_len = cfg_.df_order + cfg_.conv_lookahead;
    const size_t x_len = std::max(cfg_.df_order, cfg_.lookahead);

    // spec_buf_: [ch, 1, 1, n_freqs, 2]
    spec_buf_ = Tensor({ ch, 1, 1, n_freqs, 2 });

    // erb_buf_: [ch, 1, 1, nb_erb]
    erb_buf_ = Tensor({ ch, 1, 1, cfg_.nb_erb });

    // cplx_buf_: [ch, 1, nb_df, 2]
    cplx_buf_ = Tensor({ ch, 1, cfg_.nb_df, 2 });

    // Pre-allocated zero ERB mask (same as Rust m_zeros)
    m_zeros_.assign(cfg_.nb_erb, 0.0f);

    rolling_spec_buf_y_.clear();
    rolling_spec_buf_x_.clear();

    // rolling_spec_buf_y: capacity df_order + conv_lookahead
    for (size_t i = 0; i < y_len; ++i) {
        rolling_spec_buf_y_.emplace_back(Tensor({ ch, 1, 1, n_freqs, 2 }));
    }
    // rolling_spec_buf_x: capacity max(df_order, lookahead)
    for (size_t i = 0; i < x_len; ++i) {
        rolling_spec_buf_x_.emplace_back(Tensor({ ch, 1, 1, n_freqs, 2 }));
    }
}
// spec_x: deque of spectrogram frames, oldest at front, newest at back.
// coefs:  complex DF coefficients, layout [ch, nb_df, df_order, 2] (re/im interleaved)
// nb_df:  number of DF bins
// df_order: DF filter order
// n_freqs: total FFT bins
// spec_out: on input: ERB-stage spectrum [ch, n_freqs, 2]
//           on output: low freq [0..nb_df-1] replaced by DF result, high freq unchanged.
void DfTractCpp::df_apply(const std::deque<Tensor>& spec_x,
    const Tensor& coefs,
    size_t nb_df,
    size_t df_order,
    size_t n_freqs,
    Tensor& spec_out) const
{
    const size_t ch = cfg_.ch;

    // Runtime checks roughly mirroring Rust debug_asserts.
    if (spec_x.size() < df_order) {
        throw std::runtime_error("df_apply: not enough history frames");
    }

    // Pointer to complex DF coefficients: [ch, nb_df, df_order, 2]
    const float* coef_ptr = coefs.ptr();

    // 1) Zero relevant frequency bins (0..nb_df-1) of spec_out, like:
    //    o_f.slice_mut(s![.., ..nb_df]).fill(Complex32::default());
    for (size_t c = 0; c < ch; ++c) {
        float* out_ch = spec_out.ptr() + c * n_freqs * 2;
        for (size_t f = 0; f < nb_df; ++f) {
            out_ch[2 * f + 0] = 0.0f;
            out_ch[2 * f + 1] = 0.0f;
        }
        // bins >= nb_df remain as-is (carry ERB-stage spectrum)
    }

    // 2) Iterate over DF frames (time) and accumulate x * h into spec_out
    //
    // Rust:
    // let spec_iter = spec.iter().map(...);
    // for (s_f, c_f) in spec_iter.zip(coefs_arr.axis_iter(Axis(2))) {
    //     for (s_ch, c_ch, o_ch) in izip!(...) {
    //         for (&s, &c, o) in izip!(s_ch, c_ch, o_ch.iter_mut()) {
    //             *o += s * c;
    //         }
    //     }
    // }
    //
    // o -> c -> f
    for (size_t o = 0; o < df_order; ++o) {
        // spec_x[o]: oldest frame first, matching Rust's spec.iter() zip with coefs axis 2
        const Tensor& frame = spec_x[o];
        const float* frame_ptr = frame.ptr(); // layout [ch, n_freqs, 2]

        // Iterate over channels
        for (size_t c = 0; c < ch; ++c) {
            const float* hist_ch = frame_ptr + c * n_freqs * 2;
            float* out_ch = spec_out.ptr() + c * n_freqs * 2;

            // Iterate over frequency bins up to nb_df
            for (size_t f = 0; f < nb_df; ++f) {
                // coef index: [c, f, o, re/im] with dims [ch, nb_df, df_order, 2]
                const size_t coef_index = ((c * nb_df + f) * df_order + o) * 2;
                const float re_h = coef_ptr[coef_index + 0];
                const float im_h = coef_ptr[coef_index + 1];

                const float re_x = hist_ch[2 * f + 0];
                const float im_x = hist_ch[2 * f + 1];

                // complex multiply x * h
                const float o_re = re_x * re_h - im_x * im_h;
                const float o_im = re_x * im_h + im_x * re_h;

                // accumulate: o_f[c, f] += s * c
                out_ch[2 * f + 0] += o_re;
                out_ch[2 * f + 1] += o_im;
            }
        }
    }

    // bins >= nb_df remain unchanged (ERB-only spectrum), same as Rust: only
    // slice(.., ..nb_df) is zeroed and updated, the rest of o_f is not touched.
}

DfTractCpp::RawOut DfTractCpp::process_raw() {
    RawOut out;
    const size_t ch = cfg_.ch;

    // 1) Feature extraction per channel from spec_buf_
    for (size_t c = 0; c < ch; ++c) {
        float* spec_ptr = spec_buf_.ptr() + c * (n_freqs_ * 2);
        std::vector<Complex32> spec_c(n_freqs_);
        interleaved_to_complex_vec(spec_ptr, spec_c, n_freqs_);

        // ERB features into erb_buf_ at [c, 0, 0, :]
        float* erb_ptr = erb_buf_.ptr() + c * cfg_.nb_erb;
        df_states_[c]->feat_erb(spec_c.data(), cfg_.alpha, erb_ptr);

        // Complex low-frequency features into cplx_buf_ at [c, 0, :, :]
        std::vector<Complex32> cplx_out(cfg_.nb_df);
        df_states_[c]->feat_cplx(spec_c.data(), cfg_.alpha, cplx_out.data());
        float* cplx_ptr = cplx_buf_.ptr() + c * (cfg_.nb_df * 2);
        complex_vec_to_interleaved(cplx_out, cplx_ptr, cfg_.nb_df);
    }

    // 2) Build encoder inputs
    // enc feat_erb: same shape as erb_buf_ -> [ch, 1, 1, nb_erb]
    Tensor enc_in_erb = erb_buf_;

    // enc feat_spec: cplx_buf_ permuted to [ch, 2, 1, nb_df]
    // [ch, 1, self.nb_df, 2] ==> [ch, 2, 1, self.nb_df]
    Tensor enc_in_cplx({ ch, 2, 1, cfg_.nb_df });
    for (size_t c = 0; c < ch; ++c) {
        const float* src = cplx_buf_.ptr() + c * (cfg_.nb_df * 2);
        float* dst = enc_in_cplx.ptr() + c * (2 * cfg_.nb_df);
        // real part
        for (size_t f = 0; f < cfg_.nb_df; ++f) {
            dst[0 * cfg_.nb_df + f] = src[2 * f + 0];
        }
        // imag part
        for (size_t f = 0; f < cfg_.nb_df; ++f) {
            dst[1 * cfg_.nb_df + f] = src[2 * f + 1];
        }
    }

    std::vector<Tensor> enc_inputs = { enc_in_erb, enc_in_cplx };

    std::vector<Tensor> enc_outputs = enc_->run(enc_inputs);
    if (enc_outputs.size() < 3) {
        throw std::runtime_error("Encoder returned insufficient outputs");
    }

    // Pop lsnr, c0, emb from the back (same order as Rust)
    //{ e0, e1, e2, e3, c0, emb, lsnr }
    Tensor t_lsnr = std::move(enc_outputs[6]);

    if (t_lsnr.size() < 1) {
        throw std::runtime_error("Encoder-lsnr empty");
    }
    out.lsnr = t_lsnr.data[0];

    Tensor c0 = std::move(enc_outputs[5]);

    Tensor emb = std::move(enc_outputs[4]);

    // Stage decisions
    bool apply_gains = false;
    bool apply_gain_zeros = false;
    bool apply_df = false;
    if (out.lsnr < cfg_.min_db_thresh) {
        apply_gains = false;
        apply_gain_zeros = true;
        apply_df = false;
    }
    else if (out.lsnr > cfg_.max_db_erb_thresh) {
        apply_gains = false;
        apply_gain_zeros = false;
        apply_df = false;
    }
    else if (out.lsnr > cfg_.max_db_df_thresh) {
        apply_gains = true;
        apply_gain_zeros = false;
        apply_df = false;
    }
    else {
        apply_gains = true;
        apply_gain_zeros = false;
        apply_df = true;
    }

    // 3) ERB decoder
    if (apply_gains) {
        // Remaining enc_outputs hold e0..e3
        if (enc_outputs.size() < 4) {
            throw std::runtime_error("Encoder missing e0..e3 features");
        }
        Tensor e3 = std::move(enc_outputs[3]);
        Tensor e2 = std::move(enc_outputs[2]);
        Tensor e1 = std::move(enc_outputs[1]);
        Tensor e0 = std::move(enc_outputs[0]);

        Tensor emb_for_erb = emb;  // copy for erb_dec
        std::vector<Tensor> erb_inputs = { std::move(emb_for_erb), std::move(e3), std::move(e2), std::move(e1), std::move(e0)};


        std::vector<Tensor> erb_outputs = erb_dec_->run(erb_inputs);
        if (erb_outputs.empty()) {
            throw std::runtime_error("erb_dec returned no outputs");
        }
        // Rust uses last output as gains
        //{m(gains) }
        Tensor m = std::move(erb_outputs[0]);
        out.gains = std::move(m);
    }
    else if (apply_gain_zeros) {
        Tensor zeros({ cfg_.ch, cfg_.nb_erb });
        std::fill(zeros.data.begin(), zeros.data.end(), 0.0f);
        out.gains = std::move(zeros);
    }
    else {
        out.gains = std::nullopt;
    }

    // 4) DF decoder
    if (apply_df) {
        Tensor emb_for_df = std::move(emb);  // copy for df_dec
        std::vector<Tensor> df_inputs = { std::move(emb_for_df), std::move(c0) };

        std::vector<Tensor> df_outputs = df_dec_->run(df_inputs);
        if (df_outputs.empty()) {
            throw std::runtime_error("df_dec returned no outputs");
        }
        // Rust: last output reshaped as [ch, nb_df, df_order, 2]
        //{coefs, 302}
        Tensor coefs = std::move(df_outputs[0]);
        out.coefs = std::move(coefs);
    }
    else {
        out.coefs = std::nullopt;
    }

    return out;
}
float DfTractCpp::process(const float* noisy, float* enh) {
    assert(noisy && enh);

    const size_t ch = cfg_.ch;
    const size_t hop = cfg_.hop_size;

    // 0) Frame energy and "RMS" (actually mean square, same as Rust)
    float  max_a = 0.0f;
    double e = 0.0;
    for (size_t c = 0; c < ch; ++c) {
        const float* chptr = noisy + c * hop;
        for (size_t i = 0; i < hop; ++i) {
            const float v = chptr[i];
            max_a = std::max(max_a, std::fabs(v));
            e += static_cast<double>(v) * static_cast<double>(v);
        }
    }
    const float rms = static_cast<float>(e / static_cast<double>(ch * hop));

    // --- Silence / skip counter logic (Rust-aligned) ---
    // Rust:
    // if rms < 1e-7 { skip_counter += 1 } else { skip_counter = 0 }
    // if skip_counter > 5 { enh.fill(0.); return Ok(-15.); }
    if (rms < 1e-7f) {
        ++skip_counter_;
    }
    else {
        skip_counter_ = 0;
    }
    if (skip_counter_ > 5) {
        std::fill(enh, enh + ch * hop, 0.0f);
        return -15.0f;
    }

    // --- Clipping warning (same as Rust) ---
    if (max_a > 0.9999f) {
        std::cerr << "Warning: possible clipping detected: " << max_a << "\n";
    }

    // 1) Update rolling buffers (drop oldest)
    if (!rolling_spec_buf_y_.empty())
        rolling_spec_buf_y_.pop_front();
    if (!rolling_spec_buf_x_.empty())
        rolling_spec_buf_x_.pop_front();

    // 2) Analysis: time-domain -> complex spectrum, write into spec_buf_
    for (size_t c = 0; c < ch; ++c) {
        const float* in_ptr = noisy + c * hop;
        float* spec_ptr = spec_buf_.ptr() + c * (n_freqs_ * 2);

        // Temporary complex buffer for this channel
        std::vector<Complex32> spec_c(n_freqs_);
        df_states_[c]->analysis(in_ptr, spec_c.data());
        complex_vec_to_interleaved(spec_c, spec_ptr, n_freqs_);
    }

    // Push current spectrum into both rolling buffers (signal model y = f(x))
    rolling_spec_buf_y_.push_back(spec_buf_);
    rolling_spec_buf_x_.push_back(spec_buf_);

    // 3) Fast bypass when attenuation limit == 1.0 (no noise reduction)
    // Rust: if self.atten_lim.unwrap_or_default() == 1. { enh.assign(&noisy); return Ok(35.); }
    if (cfg_.atten_lim.has_value() &&
        std::fabs(cfg_.atten_lim.value() - 1.0f) < 1e-6f) {
        std::memcpy(enh, noisy, sizeof(float) * ch * hop);
        return 35.0f;
    }

    // 4) High-level model pass: encoder + erb_dec + df_dec
    RawOut raw = process_raw();  // (lsnr, gains, coefs)

    // 5) Decide whether to apply ERB / post stages
    // Rust: let (apply_erb, _, _) = self.apply_stages(lsnr);
    bool apply_erb = true;
    {
        const float lsnr = raw.lsnr;
        if(lsnr < cfg_.min_db_thresh || lsnr > cfg_.max_db_erb_thresh) {
            apply_erb = false;
        }
        //else
        //{
        //    apply_erb = true;
        //}
    }

    // 6) Apply ERB gains on rolling_spec_buf_y_[df_order - 1]
    const size_t spec_idx = cfg_.df_order - 1;
    Tensor& spec_target = rolling_spec_buf_y_.at(spec_idx);

    if (raw.gains.has_value()) {
        Tensor gains = raw.gains.value();
        const size_t g_ndim = gains.shape.size();
        const size_t g_ch = g_ndim >= 1 ? gains.shape[0] : 1;

        if (g_ch < ch) {
            // Single-channel mask replicated to all channels (Rust: reduced to single channel)
            float* gptr = gains.ptr();  // length = nb_erb
            for (size_t c = 0; c < ch; ++c) {
                float* spec_ptr = spec_target.ptr() + c * (n_freqs_ * 2);
                std::vector<Complex32> spec_c(n_freqs_);
                interleaved_to_complex_vec(spec_ptr, spec_c, n_freqs_);
                // Rust: self.df_states[0].apply_mask(...)
                df_states_[0]->apply_mask(spec_c.data(), gptr);
                complex_vec_to_interleaved(spec_c, spec_ptr, n_freqs_);
            }
        }
        else {
            // Per-channel mask (same number of channels)
            for (size_t c = 0; c < ch; ++c) {
                float* spec_ptr = spec_target.ptr() + c * (n_freqs_ * 2);
                std::vector<Complex32> spec_c(n_freqs_);
                interleaved_to_complex_vec(spec_ptr, spec_c, n_freqs_);

                float* gptr = gains.ptr() + c * cfg_.nb_erb;
                // Rust: self.df_states[0].apply_mask(...)
                df_states_[0]->apply_mask(spec_c.data(), gptr);
                complex_vec_to_interleaved(spec_c, spec_ptr, n_freqs_);
            }
        }

        // Rust: self.skip_counter = 0;
        skip_counter_ = 0;
    }
    else {
        // gains are None => skipped due to LSNR
        // Rust: self.skip_counter += 1;
        ++skip_counter_;
    }

    // 7) Copy ERB-masked spectrum into spec_buf_ (spec_out for DF)
    // Rust:
    // let spec = self.rolling_spec_buf_y.get_mut(self.df_order - 1).unwrap();
    // self.spec_buf.clone_from(spec);
    spec_buf_ = spec_target;

    // 8) Deep Filtering stage
    if (raw.coefs.has_value()) {
        df_apply(rolling_spec_buf_x_,
            raw.coefs.value(),
            cfg_.nb_df,
            cfg_.df_order,
            n_freqs_,
            spec_buf_);
    }

    // 9) Prepare spec_noisy and spec_enh for post-filter and attenuation mix
    // Rust:
    // idx = max(lookahead, df_order) - lookahead - 1
    size_t noisy_idx =
        (std::max(cfg_.lookahead, cfg_.df_order) - cfg_.lookahead - 1);

    const Tensor& spec_noisy = rolling_spec_buf_x_.at(noisy_idx);
    Tensor& spec_enh = spec_buf_;

    // 10) Optional post filter
    // Rust:
    // if apply_erb && self.post_filter {
    //     post_filter(spec_noisy, spec_enh, post_filter_beta);
    // }
    if (apply_erb && cfg_.post_filter) {
        for (size_t c = 0; c < ch; ++c) {
            std::vector<Complex32> noisy_c(n_freqs_);
            std::vector<Complex32> enh_c(n_freqs_);
            interleaved_to_complex_vec(spec_noisy.ptr() + c * n_freqs_ * 2,
                noisy_c, n_freqs_);
            interleaved_to_complex_vec(spec_enh.ptr() + c * n_freqs_ * 2,
                enh_c, n_freqs_);

            df_states_[c]->post_filter(noisy_c, enh_c, cfg_.post_filter_beta);

            complex_vec_to_interleaved(enh_c,
                spec_enh.ptr() + c * n_freqs_ * 2,
                n_freqs_);
        }
    }

    // 11) Limit attenuation by mixing back some of the noisy signal
    // Rust:
    // spec_enh *= (1. - lim); spec_enh += lim * spec_noisy;
    if (cfg_.atten_lim.has_value()) {
        const float lim = cfg_.atten_lim.value();
        const float one_minus_lim = 1.0f - lim;
        for (size_t c = 0; c < ch; ++c) {
            float* enh_ptr = spec_enh.ptr() + c * n_freqs_ * 2;
            const float* noisy_ptr = spec_noisy.ptr() + c * n_freqs_ * 2;
            for (size_t i = 0; i < n_freqs_ * 2; ++i) {
                enh_ptr[i] = enh_ptr[i] * one_minus_lim + noisy_ptr[i] * lim;
            }
        }
    }

    // 12) Synthesis: spectrum -> time domain
    // Rust: for (state, spec_ch, enh_ch) in izip!(...) { state.synthesis(spec_ch, enh_ch) }
    for (size_t c = 0; c < ch; ++c) {
        std::vector<Complex32> spec_c(n_freqs_);
        interleaved_to_complex_vec(spec_enh.ptr() + c * n_freqs_ * 2,
            spec_c, n_freqs_);

        float* out_ptr = enh + c * hop;
        df_states_[c]->synthesis(spec_c.data(), out_ptr);
    }

    // Return lsnr estimated by the encoder (same as Rust)
    return raw.lsnr;
}
