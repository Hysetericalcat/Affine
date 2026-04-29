use hf_hub::api::sync::Api;
use candle_nn::VarBuilder;
use candle_core::DType;
mod utils {
    pub mod gpt2;
    pub mod transformerblock;
    pub mod multiheadattention;
    pub mod mlp;
    pub mod embeddings;
}

use utils::gpt2::GPT2;


fn main()-> Result<(), Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model("gpt2".to_string());
    let weights_path = repo.get("model.safetensors")?;
    let device = candle_core::Device::Cpu;
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? 
    };
    let model = GPT2::new(768, 12, 50257, vb)?;
    Ok(())
}
