use hf_hub::api::sync::Api;
use candle_nn::VarBuilder;
use candle_core::{DType, Tensor};
use tokenizers::Tokenizer;
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
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    let encoding = tokenizer.encode("The cat sat on the", false).unwrap();
     let ids: Vec<u32> = encoding.get_ids().to_vec();
     let seq_len = ids.len();
     let input = Tensor::from_vec(ids, (1, seq_len), &device)?;
     let (logits,residual_streams) = model.forward(&input, 768)?;
     println!("{:?}", logits.shape());
     println!("{:?}", residual_streams.len());
     Ok(())
}
