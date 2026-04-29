use candle_nn::linear_no_bias;
use candle_core::{Tensor, Result};
use super::transformerblock::TransformerBlock;
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};
use super::embeddings::Embeddings;


// Unembedding: projects residual stream [seq_len x 768] -> vocab scores [seq_len x 50257]
// Highest score at each position = most likely next token

pub struct GPT2 {
    pub embeddings: Embeddings,
    pub transformer_blocks: Vec<TransformerBlock>,
    pub ln_f: LayerNorm,
    pub lm_head: Linear,
}


impl GPT2 {
    pub fn new(d_model:usize,n_head:usize,vocab_size:usize,vb:VarBuilder) -> Result<Self>{
        let block_size = 1024;
        let embeddings = Embeddings::new(vocab_size, block_size, d_model, vb.pp("wte"), vb.pp("wpe"))?;
        let mut transformer_blocks = vec![];
        for i in 0..12 {
          let trasnsformerblock = TransformerBlock::new(d_model,n_head,vb.pp(format!("h.{}", i)))?;
          transformer_blocks.push(trasnsformerblock);
        }
        let lm_head_w = vb.get((50257, 768), "wte.weight")?.t()?.contiguous()?;
        let lm_head = candle_nn::Linear::new(lm_head_w, None);
        let ln_f = layer_norm(d_model, 1e-5, vb.pp("ln_f"))?;
        Ok(Self {transformer_blocks,lm_head,embeddings,ln_f})
    }

    pub fn forward(&self,token_ids:&Tensor,d_model:usize) -> Result<Tensor>{
        let mut x = self.embeddings.forward(token_ids)?; // accesible outside for loop too,SCOPE doesn't ends after the loop
        //nothing to do with ownership
        for block in &self.transformer_blocks{
            x = block.forward(&x,d_model)?;  //& -> do not provide ownership of local var.
            //each block handling its own residual stream
        }
        let x = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits)
    }
}
