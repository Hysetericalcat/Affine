use candle_nn::{embedding, Embedding, VarBuilder};
use candle_nn::{linear, Linear};
use candle_core::Result;
use candle_nn::ops::softmax;
use candle_core::Tensor;

struct CausalSelfAttention{
    wq: Linear,
    wk: Linear,
    wv: Linear
}

//usize -> flexible
//n_seq*dim = n_seq*dim.dim*dim 
//so dim*dim = w
impl CausalSelfAttention{
    fn new(d_model:usize,vb: VarBuilder)->Result<Self>{
        let wq = linear(d_model,d_model,vb)?;
        let wk = linear(d_model,d_model,vb)?;
        let wv = linear(d_model,d_model,vb)?;
        Ok(Self {wq,wk,wv})
    }

    fn forward(&self,x:&Tensor,d_head:usize)->Result<Tensor>{
       let Q = self.wq.forward(&x)?;
       let K = self.wk.forward(&x)?;
       let V = self.wv.forward(&x)?;
       
       let scale = (d_head as f64).sqrt();
       let scores = (Q.matmul(&K.t()?)? / scale)?;
       let mask = Tensor::tril(
        &Tensor::ones((seq_len, seq_len), candle_core::DType::F32, x.device())?,
        0
        )?;
       let mask = ((1. - mask)? * f32::NEG_INFINITY)?;
       let scores = (scores + mask)?; //causal masking -> blurring future positions
       let attn_weights = softmax(&scores, 1)?;
       let final_scores = attn_weights.matmul(&V)?;
       final_scores
    }
}