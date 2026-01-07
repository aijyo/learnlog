#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    typedef void* DFHandle;

    typedef enum DFDType {
        DF_DTYPE_F32 = 1,
    } DFDType;

    typedef struct DFTensor {
        DFDType dtype;
        int32_t rank;

        // shape: length=rank, int64
        const int64_t* shape;

        // data pointer (contiguous by default). For F32, points to float.
        void* data;

        // bytes of data buffer (optional but recommended)
        size_t byte_len;

        // stride in elements (optional). If NULL, treated as contiguous row-major.
        const int64_t* strides;
    } DFTensor;

    // Create/destroy model handle
    int df_plan_create(const char* model_path, DFHandle* out_handle);
    void df_plan_destroy(DFHandle handle);
    const char* df_last_error_message(void);

    // Run: inputs are views (Rust will not free input buffers).
    // outputs are allocated by Rust; caller must free each output tensor via df_tensor_free.
    int df_plan_run(DFHandle handle,
        const DFTensor* inputs, size_t input_count,
        DFTensor* outputs, size_t output_count);

    // Free one tensor returned by df_plan_run (frees shape/strides/data allocated by Rust)
    void df_tensor_free(DFTensor* t);

#ifdef __cplusplus
}
#endif
