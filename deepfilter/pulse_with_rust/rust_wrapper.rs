// [package]
// name = "df_test"
// version = "0.1.0"
// edition = "2021"

// [lib]
// # If you don't specify `path`, Cargo will look for `src/lib.rs`
// name = "tract_stream_test"
// path = "src/pulse_model_wrapper.rs"

// # Export both:
// # - cdylib: dynamic library for C/C++ (Windows: .dll, Linux: .so, macOS: .dylib)
// # - staticlib: static library for C/C++ (Windows: .lib, Linux/macOS: .a)
// crate-type = ["cdylib", "staticlib"]

// [dependencies]
// anyhow = "1"
// ndarray = "0.15"
// tract-onnx = "0.21"
// tract-core = "0.21"
// tract-pulse = "0.21"

// log = "0.4"
// env_logger = "0.11"
// [profile.release]
// # Recommended for smaller/faster C ABI libs
// lto = true
// codegen-units = 1
// panic = "abort"


use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};

use tract_core::prelude::*;
use tract_onnx::prelude::*;
use tract_pulse::model::{PulsedModel, PulsedModelExt};

pub type DFHandle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub enum DFDType {
    DF_DTYPE_F32 = 1,
}

#[repr(C)]
pub struct DFTensor {
    pub dtype: DFDType,
    pub rank: i32,
    pub shape: *const i64,
    pub data: *mut c_void,
    pub byte_len: usize,
    pub strides: *const i64, // optional
}

thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<CString>> = const { std::cell::RefCell::new(None) };
}

fn set_last_error<E: std::fmt::Display>(e: E) {
    let msg = CString::new(e.to_string()).unwrap_or_else(|_| CString::new("Unknown error").unwrap());
    LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(msg));
}

#[no_mangle]
pub extern "C" fn df_last_error_message() -> *const c_char {
    LAST_ERROR.with(|cell| {
        if let Some(s) = cell.borrow().as_ref() { s.as_ptr() } else { std::ptr::null() }
    })
}

struct StreamingPlan {
    plan: SimplePlan<TypedFact, Box<dyn TypedOp>, TypedModel>,
}

// -------------------- your existing build --------------------

fn load_streaming_typed_model(model_path: &str) -> TractResult<TypedModel> {
    let mut m: InferenceModel = tract_onnx::onnx()
        .with_ignore_output_shapes(true)
        .model_for_path(model_path)?;

    let s = m.symbols.sym("S");

    let n_ch: usize = 1;
    let f0: usize = 32;
    let f1: usize = 96;

    let in0_shape = tvec![(n_ch as i64).to_dim(), 1.to_dim(), s.to_dim(), (f0 as i64).to_dim()];
    let in1_shape = tvec![(n_ch as i64).to_dim(), 2.to_dim(), s.to_dim(), (f1 as i64).to_dim()];

    let in0 = InferenceFact::dt_shape(f32::datum_type(), in0_shape);
    let in1 = InferenceFact::dt_shape(f32::datum_type(), in1_shape);

    m = m.with_input_fact(0, in0)?
        .with_input_fact(1, in1)?
        .with_input_names(["feat_erb", "feat_spec"])?
        .with_output_names(["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"])?;

    m.analyse(true)?;
    let mut typed = m.into_typed()?;
    typed.declutter()?;

    let pulsed: PulsedModel = PulsedModel::new(&typed, s, &1.to_dim())?;
    let typed_streaming = pulsed.into_typed()?.into_optimized()?;
    Ok(typed_streaming)
}

fn build_streaming_runnable(model_path: &str)
-> TractResult<SimplePlan<TypedFact, Box<dyn TypedOp>, TypedModel>> {
    let typed = load_streaming_typed_model(model_path)?;
    Ok(typed.into_runnable()?)
}

// -------------------- C API --------------------

#[no_mangle]
pub unsafe extern "C" fn df_plan_create(model_path: *const c_char, out_handle: *mut DFHandle) -> c_int {
    if out_handle.is_null() {
        set_last_error("out_handle is null");
        return -1;
    }
    *out_handle = std::ptr::null_mut();

    if model_path.is_null() {
        set_last_error("model_path is null");
        return -2;
    }
    let path = match CStr::from_ptr(model_path).to_str() {
        Ok(s) => s,
        Err(e) => { set_last_error(e); return -3; }
    };

    match build_streaming_runnable(path) {
        Ok(plan) => {
            let boxed = Box::new(StreamingPlan { plan });
            *out_handle = Box::into_raw(boxed) as *mut c_void;
            0
        }
        Err(e) => { set_last_error(e); -4 }
    }
}

#[no_mangle]
pub unsafe extern "C" fn df_plan_destroy(handle: DFHandle) {
    if handle.is_null() { return; }
    let _ = Box::from_raw(handle as *mut StreamingPlan);
}

/// Convert DFTensor view (F32, contiguous) to tract Tensor.
/// This copies data into an owned Tensor for safety.
/// If you want zero-copy, we can add a "borrowed tensor" path, but it is trickier with lifetimes across run().
unsafe fn dftensor_to_tract_tensor_f32(t: &DFTensor) -> Result<Tensor, String> {
    if t.dtype as i32 != DFDType::DF_DTYPE_F32 as i32 {
        return Err("only f32 supported".into());
    }
    if t.rank <= 0 {
        return Err("rank must be > 0".into());
    }
    if t.shape.is_null() || t.data.is_null() {
        return Err("shape/data is null".into());
    }
    if !t.strides.is_null() {
        // Keep it strict for now; we can support strided by copying with indexer later.
        return Err("strided tensor not supported yet; pass contiguous tensor".into());
    }

    let rank = t.rank as usize;
    let shape = std::slice::from_raw_parts(t.shape, rank);
    let shape_usize: TVec<usize> = shape.iter().map(|&d| d as usize).collect();

    let elem_count: usize = shape_usize.iter().product();
    let need_bytes = elem_count * std::mem::size_of::<f32>();
    if t.byte_len < need_bytes {
        return Err(format!("byte_len too small: need {need_bytes}, got {}", t.byte_len));
    }

    let src = std::slice::from_raw_parts(t.data as *const f32, elem_count);
    let owned = src.to_vec();

    // Build tract tensor
    let mut tensor = Tensor::zero_dt(f32::datum_type(), &shape_usize).map_err(|e| e.to_string())?;
    tensor.as_slice_mut::<f32>().map_err(|e| e.to_string())?.copy_from_slice(&owned);
    Ok(tensor)
}

/// Allocate output DFTensor from tract Tensor (f32 only).
fn tract_tensor_to_dftensor_f32(t: &Tensor) -> Result<DFTensor, String> {
    let shape_i64: Vec<i64> = t.shape().iter().map(|&d| d as i64).collect();

    let view = t.to_array_view::<f32>().map_err(|e| e.to_string())?;
    let mut data: Vec<f32> = view.iter().copied().collect();

    let shape_box = shape_i64.into_boxed_slice();
    let data_len_bytes = data.len() * std::mem::size_of::<f32>();

    let shape_ptr = Box::into_raw(shape_box) as *const i64;
    let data_ptr = data.as_mut_ptr();
    std::mem::forget(data);

    Ok(DFTensor {
        dtype: DFDType::DF_DTYPE_F32,
        rank: view.ndim() as i32,
        shape: shape_ptr,
        data: data_ptr as *mut c_void,
        byte_len: data_len_bytes,
        strides: std::ptr::null(), // contiguous
    })
}

#[no_mangle]
pub unsafe extern "C" fn df_plan_run(
    handle: DFHandle,
    inputs: *const DFTensor,
    input_count: usize,
    outputs: *mut DFTensor,
    output_count: usize,
) -> c_int {
    if handle.is_null() { set_last_error("handle is null"); return -1; }
    if inputs.is_null() || outputs.is_null() { set_last_error("inputs/outputs is null"); return -2; }
    if input_count == 0 || output_count == 0 { set_last_error("counts must be > 0"); return -3; }

    // Convert inputs
    let in_slice = std::slice::from_raw_parts(inputs, input_count);
    let mut tvalues: TVec<TValue> = tvec!();
    for (i, tin) in in_slice.iter().enumerate() {
        match dftensor_to_tract_tensor_f32(tin) {
            Ok(tensor) => tvalues.push(tensor.into_tvalue()),
            Err(e) => { set_last_error(format!("input[{i}] {e}")); return -4; }
        }
    }

    // Run
    let plan = &mut *(handle as *mut StreamingPlan);
    let outs = match plan.plan.run(tvalues) {
        Ok(v) => v,
        Err(e) => { set_last_error(e); return -5; }
    };

    if outs.len() < output_count {
        set_last_error(format!("model outputs {} < requested {}", outs.len(), output_count));
        return -6;
    }

    // Fill outputs[0..output_count]
    let out_slice = std::slice::from_raw_parts_mut(outputs, output_count);
    for i in 0..output_count {
        // Initialize to null in case of partial failure
        out_slice[i] = DFTensor {
            dtype: DFDType::DF_DTYPE_F32,
            rank: 0,
            shape: std::ptr::null(),
            data: std::ptr::null_mut(),
            byte_len: 0,
            strides: std::ptr::null(),
        };

        // let ten = match outs[i].clone().into_tensor() {
        //     Ok(t) => t,
        //     Err(e) => { set_last_error(format!("output[{i}] into_tensor failed: {e}")); return -7; }
        // };

        let ten: Tensor = outs[i].clone().into_tensor();
        match tract_tensor_to_dftensor_f32(&ten) {
            Ok(dt) => out_slice[i] = dt,
            Err(e) => { set_last_error(format!("output[{i}] {e}")); return -8; }
        }
    }

    0
}

#[no_mangle]
pub unsafe extern "C" fn df_tensor_free(t: *mut DFTensor) {
    if t.is_null() { return; }
    let tt = &mut *t;

    // Free shape
    if !tt.shape.is_null() && tt.rank > 0 {
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(tt.shape as *mut i64, tt.rank as usize));
    }

    // Free strides (if ever allocated)
    if !tt.strides.is_null() && tt.rank > 0 {
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(tt.strides as *mut i64, tt.rank as usize));
    }

    // Free data (f32)
    if !tt.data.is_null() && tt.byte_len > 0 {
        let elem = tt.byte_len / std::mem::size_of::<f32>();
        let _ = Vec::from_raw_parts(tt.data as *mut f32, elem, elem);
    }

    // Reset
    tt.shape = std::ptr::null();
    tt.strides = std::ptr::null();
    tt.data = std::ptr::null_mut();
    tt.byte_len = 0;
    tt.rank = 0;
}
