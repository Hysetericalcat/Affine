use candle_nn::{linear, Linear, Module};
use candle_core::{Tensor, Result};
use candle_nn::VarBuilder;
use candle_nn::Activation;


pub struct MLP {
    pub fc1: Linear,
    pub fc2: Linear,
}

//Larger d_model means each token's vector has more dimensions — more room to encode distinct features simultaneously.
impl MLP {
    pub fn new(d_model: usize, vb: VarBuilder) -> Result<Self> {
        //loads the weight in transpose
        let fc1_w = vb.get((768, 3072), "c_fc.weight")?.t()?.contiguous()?;
        let fc1_b = vb.get(3072, "c_fc.bias")?;
        let fc1 = candle_nn::Linear::new(fc1_w, Some(fc1_b));

        let fc2_w = vb.get((3072, 768), "c_proj.weight")?.t()?.contiguous()?;
        let fc2_b = vb.get(768, "c_proj.bias")?;
        let fc2 = candle_nn::Linear::new(fc2_w, Some(fc2_b));
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self,x:&Tensor)->Result<Tensor>{
        let x = self.fc1.forward(x)?;
        let x = x.apply(&Activation::Gelu)?;
        let x = self.fc2.forward(&x)?;
        Ok(x)
    }
}