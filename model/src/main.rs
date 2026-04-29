use hf_hub::api::sync::Api;
use candle_nn::VarBuilder;
use candle_core::{DType, Tensor};
use tokenizers::Tokenizer;
use candle_core::IndexOp;
mod utils {
    pub mod gpt2;
    pub mod transformerblock;
    pub mod multiheadattention;
    pub mod mlp;
    pub mod embeddings;
}

use utils::gpt2::GPT2;
fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32> {
    let dot = a.mul(b)?.sum_all()?.to_scalar::<f32>()?;
    let norm_a = a.mul(a)?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    let norm_b = b.mul(b)?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
    Ok(dot / (norm_a * norm_b))
}

fn concatenate_by_label(label:&String,all_vectors:&Vec<(Vec<Vec<Tensor>>, &str)>)->Result<?Vec<Vec<Tensor>>>{
     for vector in all_vectors{
        let mut vectors = vec![];
        if vector[1] == label{
            vectors.push(vector[0])
        }
     }
     Ok(?vectors)
}

fn main()-> Result<(), Box<dyn std::error::Error>> {
    let sentences: Vec<(&str, &str)> = vec![
    ("She felt terrified when the lights went out.", "fear"),
    ("He felt afraid when the footsteps grew closer.", "fear"),
    ("She felt panicked when the bridge began to shake.", "fear"),
    ("He felt horrified when the door slowly opened.", "fear"),
    ("She felt dread when the phone rang at midnight.", "fear"),
    ("He felt scared when the shadows moved outside.", "fear"),
    ("She felt anxious when the results came back.", "fear"),
    ("He felt uneasy when the stranger followed him.", "fear"),
    ("She felt alarmed when the alarm suddenly stopped.", "fear"),
    ("He felt frightened when the dog growled loudly.", "fear"),

    ("She felt elated when the letter finally arrived.", "joy"),
    ("He felt overjoyed when his name was called.", "joy"),
    ("She felt delighted when the surprise was revealed.", "joy"),
    ("He felt ecstatic when the crowd began cheering.", "joy"),
    ("She felt thrilled when the results came back.", "joy"),
    ("He felt jubilant when the final whistle blew.", "joy"),
    ("She felt grateful when the stranger helped her.", "joy"),
    ("He felt content when the evening grew quiet.", "joy"),
    ("She felt cheerful when the sun came out.", "joy"),
    ("He felt happy when the door slowly opened.", "joy"),

    ("She felt furious when the letter finally arrived.", "anger"),
    ("He felt enraged when his name was called.", "anger"),
    ("She felt livid when the surprise was revealed.", "anger"),
    ("He felt outraged when the crowd began cheering.", "anger"),
    ("She felt bitter when the results came back.", "anger"),
    ("He felt irritated when the final whistle blew.", "anger"),
    ("She felt resentful when the stranger helped her.", "anger"),
    ("He felt hostile when the evening grew quiet.", "anger"),
    ("She felt agitated when the lights went out.", "anger"),
    ("He felt infuriated when the door slowly opened.", "anger"),

    ("She felt devastated when the letter finally arrived.", "sadness"),
    ("He felt heartbroken when his name was called.", "sadness"),
    ("She felt grief when the surprise was revealed.", "sadness"),
    ("He felt sorrowful when the crowd began cheering.", "sadness"),
    ("She felt melancholy when the results came back.", "sadness"),
    ("He felt despondent when the final whistle blew.", "sadness"),
    ("She felt lonely when the stranger helped her.", "sadness"),
    ("He felt mournful when the evening grew quiet.", "sadness"),
    ("She felt hopeless when the lights went out.", "sadness"),
    ("He felt empty when the door slowly opened.", "sadness"),

    ("She felt nothing when the letter finally arrived.", "neutral"),
    ("He felt indifferent when his name was called.", "neutral"),
    ("She felt calm when the surprise was revealed.", "neutral"),
    ("He felt unbothered when the crowd began cheering.", "neutral"),
    ("She felt neutral when the results came back.", "neutral"),
    ("He felt detached when the final whistle blew.", "neutral"),
    ("She felt passive when the stranger helped her.", "neutral"),
    ("He felt unmoved when the evening grew quiet.", "neutral"),
    ("She felt blank when the lights went out.", "neutral"),
    ("He felt numb when the door slowly opened.", "neutral"),
];

    let api = Api::new()?;
    let repo = api.model("gpt2".to_string());
    let weights_path = repo.get("model.safetensors")?;
    let device = candle_core::Device::Cpu;
    let vb = unsafe { 
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? 
    };
    let model = GPT2::new(768, 12, 50257, vb)?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
    // 1. collect all vectors
let mut all_vectors: Vec<(Vec<Tensor>, &str)> = vec![];

for (sentence, label) in &sentences {
    let encoding = tokenizer.encode(*sentence, false).unwrap();
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let seq_len = ids.len();
    let input = Tensor::from_vec(ids, (1, seq_len), &device)?;
    let (_logits, residual_streams) = model.forward(&input, 768)?;

    let mut vectors = vec![];
    for stream in &residual_streams {
        let mean_vec = stream.i(0)?.mean(0)?;
        vectors.push(mean_vec);
    }
    all_vectors.push((vectors, label));
}

// 2. compute per-layer centroids per category
let labels = vec!["fear", "joy", "anger", "sadness", "neutral"];
let n_layers = 13;

for layer in 0..n_layers {
    println!("=== Layer {} ===", layer);
    
    // compute centroid per category at this layer
    let mut centroids: Vec<(&str, Tensor)> = vec![];
    for label in &labels {
        let vecs: Vec<&Tensor> = all_vectors.iter()
            .filter(|(_, l)| l == label)
            .map(|(v, _)| &v[layer])
            .collect();
        let stacked = Tensor::stack(&vecs, 0)?;
        let centroid = stacked.mean(0)?;
        centroids.push((label, centroid));
    }

    // pairwise cosine similarity between centroids
    for i in 0..centroids.len() {
        for j in i+1..centroids.len() {
            let sim = cosine_similarity(&centroids[i].1, &centroids[j].1)?;
            println!("{} vs {}: {:.4}", centroids[i].0, centroids[j].0, sim);
        }
    }
}
    println!("Processed {} sentences", all_vectors.len());
    Ok(())
}
//everything returns Results ,you need to unwrap it