use candle_nn::{embedding, Embedding, Module, VarBuilder};
use candle_core::{Tensor, Result};

pub struct Embeddings {
    pub token_emb: Embedding,
    pub pos_emb: Embedding,
}

impl Embeddings {
    pub fn new(vocab_size: usize, block_size: usize, d_model: usize, wte_vb: VarBuilder, wpe_vb: VarBuilder)-> Result<Self> {
        let token_emb = embedding(vocab_size, d_model, wte_vb)?; //loads wte.weight
        let pos_emb = embedding(block_size, d_model, wpe_vb)?;   //loads wpe.weight
        Ok(Self {token_emb,pos_emb})
    }
    
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let seq_len = token_ids.dim(1)?; //extract token ids from output of tokeniser.
        let positions = Tensor::arange(0u32, seq_len as u32, token_ids.device())?; //creates tensor of seq_len*token_ids
        let tok = self.token_emb.forward(token_ids)?;
        let pos = self.pos_emb.forward(&positions)?; //position passed as reference becoz Tensor reuqires it
        tok + pos
    }
}