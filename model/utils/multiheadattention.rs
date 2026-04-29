use candle_nn::{linear, Linear, VarBuilder};
use candle_core::{Tensor, Result};
mod causal_self_attention;
use causal_self_attention::CausalSelfAttention;


struct MultiHeadAttention {
    heads: Vec<CausalSelfAttention>,
    wo: Linear,
}

impl MultiHeadAttention {
    fn new(d_model: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let d_head = d_model / n_head;
        let mut heads = vec![];
        for i in 0..n_head {
            let head = CausalSelfAttention::new(d_head, vb.pp(format!("head_{}", i)))?;
            heads.push(head);
        }
        let wo = linear(d_model, d_model, vb.pp("wo"))?;
        Ok(Self { heads, wo })
    }

    fn forward(&self, x: &Tensor, d_model: usize) -> Result<Tensor> {
        let d_head = d_model / self.heads.len();
        let mut scores = vec![];
        for head in &self.heads {
            let s = head.forward(x, d_head)?;
            scores.push(s);
        }
        let concat = Tensor::cat(&scores.iter().collect::<Vec<_>>(), 1)?;
        self.wo.forward(&concat)
    }
}