use candle_nn::{linear, Linear, Module};
use candle_core::Result;
use candle_nn::ops::softmax;
use candle_core::Tensor;
use candle_nn::VarBuilder;

pub struct CausalSelfAttention{
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear
}

//usize -> flexible
//n_seq*dim = n_seq*dim.dim*dim 
//so dim*dim = w
impl CausalSelfAttention{
    pub fn new(d_model:usize,vb: VarBuilder)->Result<Self>{
        let wq = linear(d_model,d_model,vb.pp("wq"))?;
        let wk = linear(d_model,d_model,vb.pp("wk"))?;
        let wv = linear(d_model,d_model,vb.pp("wv"))?;
        Ok(Self {wq,wk,wv})
    }

    pub fn forward(&self,x:&Tensor,d_head:usize)->Result<Tensor>{
       let q = self.wq.forward(x)?;
       let k = self.wk.forward(x)?;
       let v = self.wv.forward(x)?;
       
       let scale = (d_head as f64).sqrt();
       let scores = (q.matmul(&k.t()?)? / scale)?;
       let seq_len = x.dim(0)?;
       let mask: Vec<f32> = (0..seq_len)
       .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY }))
        .collect();
       let mask = Tensor::from_vec(mask, (seq_len, seq_len), x.device())?;
       let scores = (scores + mask)?; //causal masking -> blurring future positions
       let attn_weights = softmax(&scores, 1)?;
       let final_scores = attn_weights.matmul(&v)?;
       Ok(final_scores)
    }
}