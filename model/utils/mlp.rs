use candle_nn::{linear, Linear};
use candle_core::{Tensor, Result};
use candle_nn::{embedding, Embedding, VarBuilder};
use candle_nn::ops::gelu;


struct MLP {
    pub fc1: Linear,
    pub fc2: Linear,
}

//Larger d_model means each token's vector has more dimensions — more room to encode distinct features simultaneously.
impl MLP {
    pub fn new(d_model: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(d_model, 4 * d_model, vb.pp("fc1"))?;
        let fc2 = linear(4 * d_model, d_model, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self,x:&Tensor)->Result<Tensor>{
        //let x = gelu(&x)?;
        let x = self.fc1.forward(x)?;
        let x = gelu(x)?;   
        let x = self.fc2.forward(x)?; //x defined again and again due to passed ownership
        x
    }
}