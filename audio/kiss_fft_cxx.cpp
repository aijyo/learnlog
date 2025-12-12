#include "kiss_fft_cxx.h"

namespace x_fft {

    KissFFT::KissFFT(int fft_size, bool inverse)
        : inverse_(inverse), fft_size_(fft_size) {

        if (fft_size_ <= 0 || (fft_size_ & 1) != 0) {
            throw std::runtime_error("KissFFT: fft_size must be positive and even for kiss_fftr");
        }

        freq_size_ = fft_size_ / 2 + 1;

        // Allocate real FFT config: inverse flag selects forward vs inverse
        cfg_ = kiss_fftr_alloc(fft_size_, inverse_ ? 1 : 0, nullptr, nullptr);
        if (!cfg_) {
            throw std::runtime_error("KissFFT: kiss_fftr_alloc failed");
        }

        time_buf_.resize(fft_size_);
        freq_buf_.resize(freq_size_);
    }

    KissFFT::~KissFFT() {
        if (cfg_) {
            kiss_fftr_free(cfg_);
            cfg_ = nullptr;
        }
    }

    int KissFFT::forward(const std::vector<float>& input, std::vector<Complex32>& output) {
        if (inverse_) {
            std::cerr << "KissFFT::forward not allowed: this instance is inverse-only.\n";
            return -1;
        }
        if (static_cast<int>(input.size()) != fft_size_) {
            std::cerr << "KissFFT::forward: input size must be fft_size.\n";
            return -1;
        }
        if (static_cast<int>(output.size()) != freq_size_) {
            output.resize(freq_size_);
        }

        // kiss_fftr reads N real points and writes N/2+1 complex points
        // (time_buf_ not strictly needed, but keeps symmetry with raw-pointer overload)
        std::copy(input.begin(), input.end(), time_buf_.begin());
        kiss_fftr(cfg_, time_buf_.data(), freq_buf_.data());

        std::copy(freq_buf_.begin(), freq_buf_.end(), output.begin());
        return 0;
    }

    int KissFFT::inverse(const std::vector<Complex32>& input, std::vector<float>& output) {
        if (!inverse_) {
            std::cerr << "KissFFT::inverse not allowed: this instance is forward-only.\n";
            return -1;
        }
        if (static_cast<int>(input.size()) != freq_size_) {
            std::cerr << "KissFFT::inverse: input size must be freq_size (N/2+1).\n";
            return -1;
        }
        if (static_cast<int>(output.size()) != fft_size_) {
            output.resize(fft_size_);
        }

        std::copy(input.begin(), input.end(), freq_buf_.begin());
        kiss_fftri(cfg_, freq_buf_.data(), time_buf_.data());

        std::copy(time_buf_.begin(), time_buf_.end(), output.begin());

        // Note: Many FFT implementations return unnormalized inverse.
        // If you need exact round-trip amplitude match with Rust, you may need:
        // for (auto& v : output) v *= (1.0f / fft_size_);

        return 0;
    }

    int KissFFT::forward(const float* input, kiss_fft_cpx* output) {
        if (inverse_) {
            std::cerr << "KissFFT::forward not allowed: this instance is inverse-only.\n";
            return -1;
        }
        if (!input || !output) {
            std::cerr << "KissFFT::forward: null pointer.\n";
            return -1;
        }

        // Copy N real points
        std::memcpy(time_buf_.data(), input, sizeof(float) * fft_size_);

        kiss_fftr(cfg_, time_buf_.data(), freq_buf_.data());

        // Copy N/2+1 complex bins
        std::memcpy(output, freq_buf_.data(), sizeof(kiss_fft_cpx) * freq_size_);
        return 0;
    }

    int KissFFT::inverse(const kiss_fft_cpx* input, float* output) {
        if (!inverse_) {
            std::cerr << "KissFFT::inverse not allowed: this instance is forward-only.\n";
            return -1;
        }
        if (!input || !output) {
            std::cerr << "KissFFT::inverse: null pointer.\n";
            return -1;
        }

        // Copy N/2+1 complex bins
        std::memcpy(freq_buf_.data(), input, sizeof(kiss_fft_cpx) * freq_size_);

        kiss_fftri(cfg_, freq_buf_.data(), time_buf_.data());

        // Copy N real points
        std::memcpy(output, time_buf_.data(), sizeof(float) * fft_size_);

        // Optional normalization (see note in vector overload)
        // for (int i = 0; i < fft_size_; ++i) output[i] *= (1.0f / fft_size_);

        return 0;
    }

} // namespace x_fft
