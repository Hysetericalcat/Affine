use multiheadattention::MultiHeadAttention;
use mlp::MLP;
use candle_nn::{layer_norm, LayerNorm, VarBuilder};
use candle_core::{Tensor, Result};;

// Struct holds weights in memory once via new(); forward() reuses them — weights persist, functions don't.
//vb ensures weight are loaded not created

pub struct TransformerBlock {
     pub multihead :MultiHeadAttention,
     pub mlp : MLP,
     pub l1:LayerNorm,
     pub l2:LayerNorm
};

impl TransformerBlock {
    pub fn new(d_model:usize,n_head:usize,vb:VarBuilder)->Result<Self>{
        let multihead = MultiHeadAttention ::new(d_model,n_head,vb.pp("attn"));
        let mlp = MLP::new(d_model,vb.pp("mlp"))?;
        let ln1 = layer_norm(d_model, 1e-5, vb.pp("ln1"))?;
        let ln2 = layer_norm(d_model, 1e-5, vb.pp("ln2"))?;
        Ok(Self {multihead,mlp,ln1,ln2})
    }

    pub fn forward(&self, x: &Tensor,d_model:usize)-> Result<Tensor>{
    let x = (x + self.multihead.forward(&self.ln1.forward(x)?, d_model)?)?;
    let x = (x + self.mlp.forward(&self.ln2.forward(&x)?)?)?;
    Ok(x)
    }
}

