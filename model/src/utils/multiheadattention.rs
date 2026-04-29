use candle_nn::{linear, Linear, Module, VarBuilder};
use candle_core::{Tensor, Result};
use candle_nn::ops::softmax;

// GPT-2 combines W_Q, W_K, W_V into a single c_attn matrix [d_model, 3*d_model]
// then splits the output into Q, K, V and reshapes into heads

pub struct MultiHeadAttention {
    pub c_attn: Linear,   // combined QKV projection [d_model -> 3*d_model]
    pub c_proj: Linear,   // output projection [d_model -> d_model]
    pub n_head: usize,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
    let c_attn_w = vb.get((768, 2304), "c_attn.weight")?.t()?.contiguous()?;
    let c_attn_b = vb.get(2304, "c_attn.bias")?;
    let c_attn = candle_nn::Linear::new(c_attn_w, Some(c_attn_b));

    let c_proj_w = vb.get((768, 768), "c_proj.weight")?.t()?.contiguous()?;
    let c_proj_b = vb.get(768, "c_proj.bias")?;
     let c_proj = candle_nn::Linear::new(c_proj_w, Some(c_proj_b));
        Ok(Self { c_attn, c_proj, n_head })
    }

    pub fn forward(&self, x: &Tensor, d_model: usize) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let d_head = d_model / self.n_head;

        // Combined QKV projection: [batch, seq_len, d_model] -> [batch, seq_len, 3*d_model]
        let qkv = self.c_attn.forward(x)?;

        // Split into Q, K, V along last dim: each [batch, seq_len, d_model]
        let q = qkv.narrow(2, 0, d_model)?;
        let k = qkv.narrow(2, d_model, d_model)?;
        let v = qkv.narrow(2, 2 * d_model, d_model)?;

        // Reshape to multi-head: [batch, seq_len, d_model] -> [batch, seq_len, n_head, d_head]
        //                        -> [batch, n_head, seq_len, d_head]
        let q = q.reshape((batch, seq_len, self.n_head, d_head))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((batch, seq_len, self.n_head, d_head))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((batch, seq_len, self.n_head, d_head))?.transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention
        let scale = (d_head as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? / scale)?; // [batch, n_head, seq_len, seq_len]

        // Causal mask
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0f32 } else { f32::NEG_INFINITY }))
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, seq_len, seq_len), x.device())?; // [1, 1, seq, seq] broadcasts over batch & heads

        let scores = scores.broadcast_add(&mask)?; // causal masking -> blurring future positions
        let attn_weights = softmax(&scores, 3)?; // softmax over last dim (key positions)

        // Weighted sum: [batch, n_head, seq_len, seq_len] @ [batch, n_head, seq_len, d_head]
        let out = attn_weights.matmul(&v)?;

        // Reshape back: [batch, n_head, seq_len, d_head] -> [batch, seq_len, n_head, d_head] -> [batch, seq_len, d_model]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((batch, seq_len, d_model))?;

        // Output projection
        self.c_proj.forward(&out) //to maintain the n_seq x dim accross residuals
    }
}