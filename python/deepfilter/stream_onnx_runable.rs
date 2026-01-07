// [package]
// name = "tract_stream_test"
// version = "0.1.0"
// edition = "2021"

// [dependencies]
// anyhow = "1"
// ndarray = "0.15"
// tract-onnx = "0.21"
// tract-core = "0.21"
// tract-pulse = "0.21"

// log = "0.4"
// env_logger = "0.11"

use ndarray::Array4;
use anyhow::Result;

use tract_onnx::prelude::*;
use tract_core::prelude::*;
use tract_core::plan::SimplePlan;
use tract_core::model::TypedModel;
use tract_core::plan::SimpleState ;
use tract_pulse::model::{PulsedModel, PulsedModelExt};

fn load_streaming_typed_model(model_path: &str) -> Result<TypedModel> {
    // Load ONNX as inference model
    let mut m: InferenceModel = tract_onnx::onnx()
        .with_ignore_output_shapes(true)
        .model_for_path(model_path)?;

    // Create symbolic dim S (streaming axis)
    let s = m.symbols.sym("S");

    // ---------------------------------------------------------------------
    // IMPORTANT:
    // Replace these with your real input shapes.
    // Example assumes:
    //   input0: [N, 1, S, F0]
    //   input1: [N, 2, S, F1]
    // ---------------------------------------------------------------------
    let n_ch: usize = 1;
    let f0: usize = 32;
    let f1: usize = 96;

    // Use shapefactoid! macro from tract_core::prelude::*
    // Build shape without shapefactoid! macro (more stable across tract versions)
    let in0_shape = tvec![
        (n_ch as i64).to_dim(),
        1.to_dim(),
        s.to_dim(),
        (f0 as i64).to_dim()
    ];

    let in1_shape = tvec![
        (n_ch as i64).to_dim(),
        2.to_dim(),
        s.to_dim(),
        (f1 as i64).to_dim()
    ];

    let in0 = InferenceFact::dt_shape(f32::datum_type(), in0_shape);
    let in1 = InferenceFact::dt_shape(f32::datum_type(), in1_shape);


    // Set input facts & names
    m = m
        .with_input_fact(0, in0)?
        .with_input_fact(1, in1)?
        .with_input_names(["feat_erb", "feat_spec"])?
        .with_output_names(["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"])?;

    // Analyze -> typed -> declutter
    m.analyse(true)?;
    let mut typed = m.into_typed()?;
    typed.declutter()?;

    // Pulse along symbolic axis `S` with pulse size 1
    // In tract-pulse 0.21, `new` is provided by PulsedModelExt trait.
    let pulsed: PulsedModel = PulsedModel::new(&typed, s, &1.to_dim())?;

    // Optimize after pulsing
    let typed_streaming = pulsed.into_typed()?.into_optimized()?;
    Ok(typed_streaming)
}

fn build_streaming_runnable(model_path: &str)
  -> Result<SimplePlan<TypedFact, Box<dyn TypedOp>, TypedModel>> {
    let typed = load_streaming_typed_model(model_path)?;
    Ok(typed.into_runnable()?)
}

fn print_cpp_vector(name: &str, data: &[f32]) {
    // Print like C++ std::vector<float>
    println!("std::vector<float> {} = {{", name);
    for (i, v) in data.iter().enumerate() {
        if i % 8 == 0 {
            print!("    ");
        }
        print!("{:.9e}f", v);
        if i + 1 != data.len() {
            print!(", ");
        }
        if i % 8 == 7 {
            println!();
        }
    }
    println!("\n}};\n");
}

fn tensor_to_f32_vec(t: &Tensor) -> TractResult<Vec<f32>> {
    // Convert tensor to a flat Vec<f32> (contiguous)
    let view = t.to_array_view::<f32>()?;
    Ok(view.iter().copied().collect())
}

fn make_enc_inputs(
    feat_erb: &[f32],
    feat_spec: &[f32],
) -> TractResult<TVec<TValue>> {
    let t0 = Tensor::from_shape(&[1, 1, 1, 32], feat_erb)?;
    let t1 = Tensor::from_shape(&[1, 2, 1, 96], feat_spec)?;
    Ok(tvec![t0.into(), t1.into()])
}

fn main() -> TractResult<()> {
    let model_path = r"D:\code\mycode\onnx_mlir_test\model_shim\3rd\deepfilter\lib\enc.onnx";


    // -----------------------------
    // Input 0: feat_erb (1, 1, 1, 32)
    // -----------------------------
    let enc_input00_data: Vec<f32> = vec![
    -9.870241880e-1f32, -9.582564235e-1f32, -7.621173859e-1f32, -6.845709085e-1f32, -7.189494967e-1f32, -8.269284964e-1f32, 
    -8.343971372e-1f32, -7.436777353e-1f32, -7.492841482e-1f32, -7.691404223e-1f32, -7.460414767e-1f32, -7.230672836e-1f32, 
    -6.998332739e-1f32, -6.765916944e-1f32, -6.533386111e-1f32, -6.292451620e-1f32, -6.057933569e-1f32, -5.799909830e-1f32, 
    -5.579267740e-1f32, -5.230569839e-1f32, -5.105007291e-1f32, -4.808557630e-1f32, -4.563953280e-1f32, -4.053146243e-1f32, 
    -4.019483626e-1f32, -3.911190033e-1f32, -3.672130704e-1f32, -3.432745039e-1f32, -3.193288743e-1f32, -2.953802049e-1f32, 
    -2.714347839e-1f32, -2.474842072e-1f32,
    ];

    let enc_input10_data: Vec<f32> = vec![
    -6.771491766e-1f32, -7.019680738e-1f32, -6.851059198e-1f32, -5.010404587e-1f32, -5.829856992e-1f32, -6.421140432e-1f32, 
    -7.570089102e-1f32, -5.909231305e-1f32, -6.382904053e-1f32, -7.184351087e-1f32, -7.222245932e-1f32, -7.184267044e-1f32, 
    -6.944219470e-1f32, -6.670005918e-1f32, -6.368633509e-1f32, -5.953594446e-1f32, -5.781076550e-1f32, -5.645904541e-1f32, 
    -5.456022024e-1f32, -4.207580686e-1f32, -4.897518158e-1f32, -2.618556917e-1f32, -3.235549852e-2f32, -4.211883619e-2f32, 
    -1.612718552e-1f32, -3.850931227e-1f32, -3.626285493e-1f32, -3.395469785e-1f32, -3.159580231e-1f32, -2.922182083e-1f32, 
    -2.686582506e-1f32, -2.449815720e-1f32,
    ];

    // let feat_erb =
    //     Array::from_shape_vec((1, 1, 1, 32), enc_input10_data)?;
    let feat_erb_arr_00: Array4<f32> = Array4::from_shape_vec((1, 1, 1, 32), enc_input00_data)?;
    let feat_erb_vec_0: Vec<f32> = feat_erb_arr_00.into_raw_vec();

    let feat_erb_arr_10: Array4<f32> = Array4::from_shape_vec((1, 1, 1, 32), enc_input10_data)?;
    let feat_erb_vec_1: Vec<f32> = feat_erb_arr_10.into_raw_vec();
    // let feat_erb: Tensor = Tensor::from_shape(&[1usize, 1usize, 1usize, 32usize], &feat_erb_vec)?;

    // -----------------------------
    // Input 1: feat_spec (1, 2, 1, 96)
    // -----------------------------
    
    let enc_input01_data: Vec<f32> = vec![
    5.608978609e-5f32, -4.347474896e-5f32, 6.358668907e-5f32, -3.843351078e-5f32, -8.016927313e-5f32, -5.761748180e-4f32, 
    1.026913291e-3f32, -5.899974494e-4f32, 3.791539057e-4f32, 3.495257988e-5f32, -2.001321409e-4f32, 4.330219781e-6f32, 
    -9.225792382e-6f32, -6.877525448e-5f32, 2.193586406e-4f32, -5.693211642e-5f32, -2.121816651e-4f32, 1.383454219e-4f32, 
    -4.102570165e-5f32, 3.543958519e-5f32, -2.710202898e-5f32, 2.246082659e-5f32, -1.828368295e-5f32, 1.599626194e-5f32, 
    -1.282310859e-5f32, 1.059764963e-5f32, -9.558402780e-6f32, 8.686864021e-6f32, -7.068610557e-6f32, 5.537151992e-6f32, 
    -5.295310075e-6f32, 4.163627182e-6f32, -3.654618922e-6f32, 4.008406904e-6f32, -2.838887212e-6f32, 2.923202828e-6f32, 
    3.094674412e-6f32, -4.932698630e-6f32, 2.836406793e-6f32, 1.320331648e-5f32, 4.804560831e-6f32, -4.035312668e-5f32, 
    -2.895317493e-6f32, 1.591035652e-5f32, 8.305859410e-6f32, -6.645963367e-6f32, 7.780853593e-6f32, 1.378697925e-5f32, 
    -4.441707461e-6f32, -1.633800457e-5f32, -3.119310577e-6f32, 9.306754691e-6f32, -4.955776603e-6f32, -4.169105068e-6f32, 
    2.615480844e-5f32, 4.148366497e-5f32, -1.161309046e-4f32, 2.725246486e-5f32, 6.823204603e-5f32, -2.750616841e-5f32, 
    -3.597101022e-5f32, -3.656224180e-6f32, 4.458432522e-5f32, 5.155150575e-6f32, -2.707873136e-5f32, -7.926018952e-6f32, 
    1.321801756e-5f32, 1.098966820e-7f32, 1.077381239e-5f32, -3.541422848e-5f32, 7.911276043e-5f32, 4.626373266e-5f32, 
    -1.657095127e-4f32, -1.946542470e-4f32, 2.797279740e-4f32, 1.733271056e-4f32, -9.339563258e-5f32, -1.476477482e-4f32, 
    -7.362264296e-5f32, 1.163858251e-4f32, 3.483941327e-5f32, -3.958287925e-5f32, 6.775619568e-6f32, 2.197765025e-5f32, 
    -1.420346416e-5f32, -3.287819345e-5f32, 3.394011219e-5f32, 2.478359238e-5f32, 1.598632298e-5f32, -1.275667910e-5f32, 
    -2.471666085e-5f32, -2.847558426e-5f32, -1.779783270e-5f32, 1.430779434e-6f32, 3.543426283e-5f32, 8.365664689e-5f32, 
    0.000000000e0f32, -2.566422882e-5f32, 8.266221994e-5f32, -5.632774628e-5f32, 3.683994873e-4f32, -6.660879590e-4f32, 
    -8.356828766e-5f32, 5.133050145e-4f32, -5.897869705e-4f32, 6.392401992e-4f32, -2.343104570e-4f32, 1.262022997e-4f32, 
    -1.382187766e-4f32, 5.318328840e-5f32, -1.821055339e-4f32, 4.106081615e-4f32, -2.668133238e-4f32, 5.253398922e-5f32, 
    -6.348027091e-5f32, 7.352134708e-5f32, -7.011153502e-5f32, 6.647958071e-5f32, -6.436883996e-5f32, 6.062118700e-5f32, 
    -5.818033242e-5f32, 5.610382868e-5f32, -5.482941197e-5f32, 5.199672887e-5f32, -5.007596337e-5f32, 4.858442480e-5f32, 
    -4.768868530e-5f32, 4.539834845e-5f32, -4.540544251e-5f32, 4.271611033e-5f32, -4.171338151e-5f32, 3.789961193e-5f32, 
    -3.653272870e-5f32, 4.038035331e-5f32, -4.586769501e-5f32, 3.960727554e-5f32, -1.026329392e-5f32, 4.361050378e-5f32, 
    -7.236587408e-5f32, 3.261413076e-5f32, -3.300197204e-5f32, 4.022487701e-5f32, -4.928738781e-5f32, 4.817149238e-5f32, 
    -2.247742668e-5f32, 3.969467434e-5f32, -5.261243132e-5f32, 3.839893179e-5f32, -3.569448018e-5f32, 2.879283238e-5f32, 
    -6.159810437e-5f32, 1.073584499e-4f32, -1.531498674e-5f32, -7.948496932e-5f32, 1.520381193e-5f32, 6.554173888e-5f32, 
    -3.202657535e-5f32, -1.886023165e-5f32, -2.861221401e-5f32, 5.944473014e-5f32, -1.425310347e-5f32, -3.526550927e-7f32, 
    -2.210812454e-5f32, 1.565372986e-5f32, -2.868090633e-6f32, -2.423618753e-6f32, -5.484180292e-5f32, 1.438076579e-4f32, 
    9.088413208e-5f32, -2.110313799e-4f32, -2.378405479e-4f32, 2.525206946e-4f32, 1.171191980e-4f32, 5.364806930e-5f32, 
    -1.750551601e-4f32, -3.547690721e-5f32, 4.366644862e-5f32, 3.722043766e-5f32, -5.320638229e-5f32, 3.956051296e-5f32, 
    -1.465760192e-6f32, 2.821646831e-6f32, -5.993662489e-5f32, 4.668066686e-5f32, -8.411542694e-6f32, 6.453706010e-5f32, 
    -2.173435678e-5f32, 2.884034984e-5f32, -5.670378232e-5f32, 4.481727842e-7f32, -6.397540710e-5f32, 6.410016067e-5f32,
    ];

    let enc_input11_data: Vec<f32> = vec![
    -1.242811792e-3f32, 2.873111516e-4f32, 1.098804642e-3f32, -4.802785115e-4f32, -9.320862773e-6f32, -6.291736499e-4f32, 
    2.022367204e-3f32, -2.187116072e-3f32, 1.247301698e-3f32, -1.209101523e-3f32, 6.613002624e-4f32, 4.919864587e-4f32, 
    -4.729395441e-4f32, 1.727683048e-5f32, -3.079771996e-4f32, -7.201943663e-5f32, 5.578460405e-4f32, 1.355503773e-4f32, 
    -3.006586048e-4f32, -1.704455353e-4f32, -1.382828195e-4f32, 5.879967648e-5f32, 4.309896758e-5f32, 1.275511841e-5f32, 
    -1.711776463e-6f32, -4.509761493e-5f32, -2.918403334e-5f32, 3.680794180e-5f32, 5.451320976e-5f32, -1.420933313e-5f32, 
    -1.203425127e-4f32, -3.627108163e-5f32, 9.969130770e-5f32, -5.709405377e-5f32, -4.968544818e-5f32, 1.667783508e-4f32, 
    -5.540413440e-6f32, -4.691503818e-6f32, 5.759509804e-6f32, 6.105523789e-5f32, -3.480919695e-4f32, 2.027342998e-4f32, 
    9.250095900e-5f32, -1.658081455e-4f32, 5.488322859e-5f32, -3.315210779e-5f32, -1.677512955e-5f32, 9.359429532e-5f32, 
    1.267079642e-4f32, -1.614986832e-4f32, 1.975881605e-4f32, -1.465995447e-4f32, 7.692698273e-5f32, -1.597273949e-4f32, 
    4.403711137e-5f32, 5.769311247e-5f32, 2.817357381e-5f32, -7.453450962e-5f32, -9.649347339e-5f32, 1.354537962e-4f32, 
    -6.035428305e-5f32, 8.885172974e-6f32, 1.427026000e-4f32, 7.015568553e-5f32, -1.712844096e-4f32, 4.759894728e-5f32, 
    5.362776210e-5f32, -8.419005462e-5f32, -2.849707962e-4f32, -1.611791595e-5f32, 4.124185070e-4f32, 3.570386907e-4f32, 
    -1.330415718e-3f32, 6.043982648e-5f32, 1.031964086e-3f32, -1.333554246e-4f32, 2.414407063e-4f32, -5.947620957e-4f32, 
    -7.407079829e-5f32, 5.877721705e-4f32, 2.307928517e-4f32, -3.039844742e-4f32, 1.308297360e-4f32, -1.269049390e-4f32, 
    1.197682559e-5f32, 3.345044388e-5f32, -1.260754070e-4f32, -1.371875405e-4f32, 8.592909580e-5f32, 1.344077755e-4f32, 
    2.475137262e-5f32, 1.262387086e-4f32, -2.919914259e-4f32, 5.354459863e-4f32, 4.720382567e-4f32, 1.631769352e-3f32, 
    0.000000000e0f32, -1.265940489e-3f32, 4.863258800e-4f32, 5.849405425e-4f32, -6.538273301e-4f32, 1.066096011e-3f32, 
    -4.485899990e-4f32, 7.670756895e-4f32, -7.103267708e-4f32, -7.150852616e-5f32, -5.810330622e-4f32, 7.126889541e-4f32, 
    1.815613796e-4f32, -4.078133497e-5f32, 4.746180493e-4f32, -1.159142237e-3f32, 7.270184869e-5f32, 6.512567634e-4f32, 
    1.421732450e-4f32, -1.582154982e-5f32, -1.331211097e-4f32, -1.168043309e-4f32, 3.898250270e-6f32, 8.619345863e-6f32, 
    2.670200956e-5f32, 1.200538009e-5f32, -5.106241224e-5f32, -3.838452903e-5f32, 2.577760097e-5f32, 8.813681779e-5f32, 
    1.040269126e-5f32, -1.232152426e-4f32, -1.569423875e-5f32, 6.412904622e-5f32, -1.275300310e-4f32, -4.475989408e-5f32, 
    1.396077569e-4f32, -4.179319876e-5f32, 4.865530354e-5f32, 6.010538709e-5f32, 9.891012451e-5f32, -3.401091381e-4f32, 
    1.788868394e-4f32, 7.821090549e-6f32, -3.997203021e-5f32, -4.242648720e-5f32, 2.181611490e-5f32, -2.615124104e-4f32, 
    3.445823095e-4f32, -1.716832194e-4f32, 8.248911035e-5f32, 3.893812391e-5f32, 3.838250632e-5f32, 8.653159057e-6f32, 
    -2.872986079e-5f32, -1.781007741e-4f32, 7.737104897e-5f32, 2.269319521e-4f32, -1.206114830e-4f32, -1.416972518e-4f32, 
    3.986655065e-5f32, -1.369180154e-6f32, -3.362797725e-5f32, 7.767003990e-5f32, 1.178117527e-4f32, -9.211959696e-5f32, 
    9.533400589e-5f32, 1.067626654e-4f32, 1.405559942e-5f32, -3.891087836e-4f32, -1.132358557e-5f32, 4.309979267e-4f32, 
    2.039512474e-4f32, -6.803821889e-4f32, -3.101754119e-4f32, -2.459127281e-4f32, 1.116472878e-3f32, -6.235179171e-5f32, 
    -9.881244041e-4f32, 2.088176698e-4f32, 4.563001567e-4f32, 4.442001591e-5f32, -3.352944987e-6f32, 4.288949276e-5f32, 
    4.831852129e-5f32, 4.191781045e-5f32, 1.960327791e-4f32, -4.063056258e-4f32, 2.852575271e-4f32, -4.048123374e-4f32, 
    3.127535747e-4f32, 7.392800762e-5f32, -2.220575407e-4f32, -6.679839862e-5f32, -6.856799246e-6f32, 1.523040934e-3f32,
    ];

    // let feat_spec =
    //     Array::from_shape_vec((1, 2, 1, 96), enc_input11_data)?;
    let feat_spec_arr_01: Array4<f32> = Array4::from_shape_vec((1, 2, 1, 96), enc_input01_data)?;
    let feat_spec_vec_0: Vec<f32> = feat_spec_arr_01.into_raw_vec();

    let feat_spec_arr_11: Array4<f32> = Array4::from_shape_vec((1, 2, 1, 96), enc_input11_data)?;
    let feat_spec_vec_1: Vec<f32> = feat_spec_arr_11.into_raw_vec();
    // let feat_spec: Tensor = Tensor::from_shape(&[1usize, 2usize, 1usize, 96usize], &feat_spec_vec)?;
    

    let feat_erb_sets: Vec<Vec<f32>> = vec![
        feat_erb_vec_0, // len = 32
        feat_erb_vec_1, // len = 32
    ];

    let feat_spec_sets: Vec<Vec<f32>> = vec![
        feat_spec_vec_0, // len = 192
        feat_spec_vec_1, // len = 192
    ];
    // Run inference
    
    // Load and prepare model
    // let runnable = tract_onnx::onnx()
    //     .model_for_path(model_path)?
    //     .into_optimized()?
    //     .into_runnable()?;

    let runnable = build_streaming_runnable(model_path)?;    
    let mut enc_state = SimpleState::new(runnable)?; // <-- DeepFilter-aligned

    // Run twice (match your C++ behavior)
    for run_idx in 0..2 {
        let inputs = make_enc_inputs(
            &feat_erb_sets[run_idx],
            &feat_spec_sets[run_idx],
        )?;
        let outputs = enc_state.run(inputs)?;

        println!("==============================");
        println!("RUN {} OUTPUTS:", run_idx);

        // Output order: e0,e1,e2,e3,emb,c0,lsnr
        let names = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"];

        for (i, tv) in outputs.into_iter().enumerate() {
            let name = if i < names.len() { names[i] } else { "out" };

            // In recent tract versions, TValue -> Tensor via into_tensor()
            let t: Tensor = tv.into_tensor();

            let vec = tensor_to_f32_vec(&t)?;
            print_cpp_vector(name, &vec);
        }
    }

    Ok(())
}
