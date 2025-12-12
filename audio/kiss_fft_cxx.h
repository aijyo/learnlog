#ifndef __X_KISS_FFT_H__
#define __X_KISS_FFT_H__

#include <vector>
#include <iostream>
#include <stdexcept>

#include "model_shim/3rd/kissfft/include/kiss_fftr.h"

using Complex32 = kiss_fft_cpx;

namespace x_fft {

    class KissFFT {
    public:
        // inverse=false: forward-only (real->complex)
        // inverse=true : inverse-only (complex->real)
        KissFFT(int fft_size, bool inverse);

        ~KissFFT();

        int forward(const std::vector<float>& input, std::vector<Complex32>& output);
        int inverse(const std::vector<Complex32>& input, std::vector<float>& output);

        int forward(const float* input, kiss_fft_cpx* output);
        int inverse(const kiss_fft_cpx* input, float* output);

        int fft_size() const { return fft_size_; }
        int freq_size() const { return freq_size_; }

    private:
        bool inverse_{ false };
        kiss_fftr_cfg cfg_{ nullptr };  // Real FFT config (forward or inverse)

        std::vector<float> time_buf_;       // size = fft_size_
        std::vector<Complex32> freq_buf_;   // size = freq_size_

        int fft_size_{ 0 };   // N (must be even for kiss_fftr)
        int freq_size_{ 0 };  // N/2 + 1
    };

} // namespace x_fft

#endif // __X_KISS_FFT_H__
