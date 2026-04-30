# affine

> Probing the geometry of affect in transformer residual streams. Are emotions directions? This finds out.

**affine** — Linear probes for emotional representations in transformer residual streams. Mechanistic interpretability, from scratch, in Rust.

---

## The Question

Are emotional concepts (fear, joy, anger, sadness) represented as distinct directions in residual stream space, and do these directions interact via superposition the same way factual features do?

This experiment tests a core assumption of dual-stream transformer architectures — that emotional and rational representations are separable. The answer, it turns out, is more interesting than a yes or no.

---

## Architecture

Built from scratch in Rust using [candle](https://github.com/huggingface/candle) (Hugging Face's Rust ML framework). No borrowed implementations — every component written and understood from first principles.

```
affine/
├── model/
│   └── src/
│       ├── main.rs               # experiment runner
│       └── utils/
│           ├── gpt2.rs           # GPT-2 top-level model
│           ├── transformerblock.rs
│           ├── multiheadattention.rs
│           ├── mlp.rs
│           └── embeddings.rs
```

**Model:** GPT-2 Small (117M parameters, 12 layers, 12 heads, d_model=768)  
**Weights:** Loaded directly from HuggingFace Hub via `hf-hub`

---

## Methodology

### Dataset
50 sentences across 5 emotion categories (10 per class): `fear`, `joy`, `anger`, `sadness`, `neutral`.

Controlled syntax — same sentence template, only the emotional word varies:

```
"She felt [emotion] when [situation]."
```

Situations repeat across categories to isolate emotion signal from syntactic/situational signal.

### Activation Extraction
For each sentence, the residual stream is captured at every layer boundary — 13 snapshots total (1 after embedding + 12 after each transformer block), producing a `[50 × 13 × 768]` activation tensor.

Mean pooling over the sequence dimension collapses each `[n_seq × 768]` snapshot to a single `[768]` vector per layer per sentence.

### Analysis

**Cross-category geometry:** At each layer, emotion class centroids are computed by averaging the 10 sentence vectors per class. Pairwise cosine similarity between centroids reveals whether emotion directions are orthogonal (separate features) or aligned (superposition).

**Intra-layer divergence:** For each sentence, cosine similarity between consecutive layer representations (layer i vs layer i+1) reveals where in the network representations change most — identifying the depth at which emotional processing occurs.

---

## Results

### Cross-Category Similarity

All pairwise cosine similarities between emotion centroids fall between **0.997 and 0.9999** across all 13 layers.

| Layer | fear vs joy | fear vs anger | anger vs sadness |
|-------|-------------|---------------|-----------------|
| 0     | 0.9972      | 0.9979        | 0.9989          |
| 3     | 0.9998      | 0.9998        | 0.9999          |
| 7     | 0.9995      | 0.9997        | 0.9999          |
| 12    | 0.9977      | 0.9983        | 0.9993          |

**Key observations:**
- Emotions are massively superposed — they occupy nearly identical directions in 768-dim space
- Fear is consistently the most geometrically distinct category, showing the lowest similarity to all other classes at every layer
- Separation is weakest at layers 3-6 (similarity peaks near 0.9999) and strongest at layer 12
- Emotion differentiation increases in later layers — geometry emerges late in the network

### Intra-Layer Divergence

Consistent pattern observed across all 50 sentences:

| Transition | Avg Cosine Similarity | Interpretation |
|------------|----------------------|----------------|
| layer 0 → 1 | ~0.17 | Massive jump — embedding to first attention block |
| layer 1 → 2 | ~0.76 | Large transformation continues |
| layer 2 → 3 | ~0.63 | Peak divergence — most representational work |
| layer 3 → 11 | ~0.999 | Near-static — middle layers barely change representations |
| layer 11 → 12 | ~0.65-0.69 | Sharp final transformation |

**Key observations:**
- The model does almost all of its representational work in layers 0-3 and the final layer 11-12
- Middle layers 3-11 are nearly static — representations are carried forward with minimal modification
- This bimodal processing pattern (early formation + late refinement) is consistent across all emotion classes and all 50 sentences

---

## Interpretation

### On Superposition
The near-unity cosine similarities confirm that GPT-2 does not maintain separate, orthogonal directions for distinct emotions. Fear, joy, anger, and sadness are encoded in heavily overlapping subspaces. This is consistent with the superposition hypothesis — the model is compressing many features into fewer dimensions than would be needed for clean separation.

### On the Dual-Stream Hypothesis
The original motivation for this experiment was to test whether emotional and rational representations are separable (a core assumption of dual-stream transformer architectures). The results suggest they are not — at the residual stream level, emotional content does not occupy a cleanly separable subspace. Routing "emotional" vs "rational" streams would require direction-level separation, not stream-level separation.

### On Layer Depth
The sharp divergence at layers 0-3 and 11-12 with stability in between suggests that emotion representations form early and are refined only at the very end. This has architectural implications — if you wanted to intervene on emotional representations, layers 2-3 and layer 11 are where the action is.

### On Fear
Fear's consistent geometric distinctiveness across all layers is an unexplained finding. One hypothesis: fear-related tokens have more distinctive syntactic and semantic contexts in GPT-2's training data, producing more separable activations. This warrants follow-up.

---

## Limitations

- **Dataset size:** 50 sentences is small. Results should be validated on a larger, more diverse corpus
- **Mean pooling:** Averaging over sequence positions discards token-level structure. The emotion word itself may carry more directional signal than the sentence average suggests
- **Single model:** Results are specific to GPT-2 Small. Larger models with higher d_model may show cleaner separation (more room for orthogonal features before superposition is forced)
- **Centroid-based probing:** Linear probe training (finding discriminative directions via SGD) would give stronger evidence than centroid cosine similarity

---

## Stack

| Component | Technology |
|-----------|-----------|
| Language | Rust |
| ML framework | [candle](https://github.com/huggingface/candle) 0.10 |
| Model weights | HuggingFace Hub (`gpt2`) |
| Tokenizer | `tokenizers` crate |
| Model | GPT-2 Small (117M) |

---

## Running

```bash
git clone https://github.com/chromeblood/affine
cd affine
cargo run --release
```

Weights are downloaded automatically from HuggingFace Hub on first run (~500MB).

---

## Next Steps

- Train proper linear probes (SGD over residual stream vectors) to find discriminative emotion directions
- Test orthogonality of learned probe directions via SVD
- Ablate emotion directions mid-forward-pass and measure output shift
- Replicate on GPT-2 Medium/Large to test whether separation improves with model scale
- Per-token analysis — probe the emotion word position specifically rather than mean pooling

---

## Connection to Prior Work

This experiment was motivated by questions in mechanistic interpretability around the superposition hypothesis (Elman, Anthropic) and geometric structure of semantic features in transformer residual streams. The methodology follows the linear representation hypothesis — that features are encoded as directions in activation space — and tests whether affect is one such feature class.

