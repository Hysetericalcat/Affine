use super::multiheadattention::MultiHeadAttention;
use super::mlp::MLP;
use candle_nn::{layer_norm, LayerNorm, VarBuilder};
use candle_core::{Tensor, Result};
use candle_nn::Module; //import to come with forward function

//Tensor -> does not implement copy traits like numbers

// Struct holds weights in memory once via new(); forward() reuses them — weights persist, functions don't.
//vb ensures weight are loaded not created

pub struct TransformerBlock {
     pub multihead :MultiHeadAttention,
     pub mlp : MLP,
     pub l1:LayerNorm,
     pub l2:LayerNorm
}

impl TransformerBlock {
    pub fn new(d_model:usize,n_head:usize,vb:VarBuilder)->Result<Self>{
        let multihead = MultiHeadAttention ::new(d_model,n_head,vb.pp("attn"))?;
        let mlp = MLP::new(d_model,vb.pp("mlp"))?;
        let l1 = layer_norm(d_model, 1e-5, vb.pp("ln_1"))?;
        let l2 = layer_norm(d_model, 1e-5, vb.pp("ln_2"))?;
        Ok(Self {multihead,mlp,l1,l2})
    }

    pub fn forward(&self, x: &Tensor,d_model:usize)-> Result<Tensor>{
    //x + consumes x. Now x is gone. Then inside something you write &x — trying to borrow x that no longer exists. Nothing to reference. Compiler rejects it.
    //x comes in as a reference. You can't move a reference — you can only borrow it. So x + on line 26 never moves anything, it's already a reference. No conflict.
    let x = (x + self.multihead.forward(&self.l1.forward(&x)?, d_model)?)?;
      //+ -> operation consumes x
    let x = (&x + self.mlp.forward(&self.l2.forward(&x)?)?)?;
    // let x (shadowed) is consumed by x (in which forward pass was added).
    Ok(x)
    }
    //? -> unwraps the tensor
}
