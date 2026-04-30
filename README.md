# affine

> Probing the geometry of affect in transformer residual streams.

A learning project exploring whether emotional concepts have geometric structure in GPT-2's residual stream. Built from scratch in Rust as an exercise in mechanistic interpretability tooling.

---

## The Question

Are emotional concepts (fear, joy, anger, sadness) represented as distinct directions in residual stream space?

This is a preliminary exploration, not a rigorous study. The methodology has known weaknesses and the results should be interpreted carefully.

---

## Architecture

GPT-2 Small implemented from scratch in Rust using [candle](https://github.com/huggingface/candle). The main motivation was learning — understanding the architecture by building it rather than borrowing an existing implementation.

```
affine/
├── model/
│   └── src/
│       ├── main.rs
│       └── utils/
│           ├── gpt2.rs
│           ├── transformerblock.rs
│           ├── multiheadattention.rs
│           ├── mlp.rs
│           └── embeddings.rs
```

**Model:** GPT-2 Small (117M parameters, 12 layers, 12 heads, d_model=768)  
**Weights:** Loaded from HuggingFace Hub via `hf-hub`

---

## Methodology

### Dataset
50 sentences across 5 emotion categories (10 per class): `fear`, `joy`, `anger`, `sadness`, `neutral`.

Same sentence template throughout:

```
"She/He felt [emotion word] when [situation]."
```

Situations repeat across categories. This controls for syntax to some degree but the dataset is far too small and homogeneous to draw strong conclusions from.

### Activation Extraction
The residual stream is captured at every layer boundary — 13 snapshots per sentence (1 after embedding + 12 after each transformer block).

Mean pooling over token positions collapses each `[n_seq × 768]` snapshot to `[768]`. This is a coarse approximation — it averages across all tokens including syntactic filler, potentially diluting the signal from the emotion word itself.

### Analysis

**Cross-category geometry:** Emotion class centroids computed per layer, pairwise cosine similarity between centroids measured across all 13 layers.

**Intra-layer divergence:** Cosine similarity between consecutive layer representations per sentence, to see where representations change most.

---

## Results

### Cross-Category Similarity

Pairwise cosine similarities between emotion centroids across all 13 layers:

| Layer | fear vs joy | fear vs anger | anger vs sadness |
|-------|-------------|---------------|-----------------|
| 0     | 0.9972      | 0.9979        | 0.9989          |
| 3     | 0.9998      | 0.9998        | 0.9999          |
| 7     | 0.9995      | 0.9997        | 0.9999          |
| 12    | 0.9977      | 0.9983        | 0.9993          |

All values fall between 0.997 and 0.9999. These are very high similarities — though it's worth noting that high cosine similarity between mean-pooled centroids in high-dimensional space is somewhat expected and doesn't strongly imply superposition on its own. Proper linear probe training would give more meaningful evidence.

Fear shows slightly lower similarity to other categories across all layers. The reason for this is unclear.

Similarity is weakest at layer 12, suggesting some differentiation emerges late — but the differences are small.

### Intra-Layer Divergence

Consistent pattern across all 50 sentences:

| Transition | Avg Cosine Similarity |
|------------|----------------------|
| layer 0 → 1 | ~0.17 |
| layer 1 → 2 | ~0.76 |
| layer 2 → 3 | ~0.63 |
| layer 3 → 11 | ~0.999 |
| layer 11 → 12 | ~0.65-0.69 |

Most representational change happens in early layers (0-3) and the final layer (11-12). Middle layers are nearly static. This pattern is consistent across all sentences but the interpretation is tentative.

---

## Limitations

These are significant, not minor:

- **Dataset is too small and too uniform.** 50 sentences from one template is not enough to make claims about emotion geometry. Results could reflect template recognition rather than emotion encoding.
- **Mean pooling discards token-level structure.** The emotion word is one token out of ~10. Averaging across all positions likely dilutes the signal. Probing the emotion token position specifically would be more informative.
- **Centroid cosine similarity is a weak probe.** It doesn't account for within-class variance and high similarity in high-dimensional space is easy to achieve. Trained linear probes with proper evaluation would be more rigorous.
- **Single model, single architecture.** Results are specific to GPT-2 Small and may not generalize.
- **No statistical testing.** No baselines, no significance testing, no comparison against random sentence pairs.

The intra-layer divergence results are more robust than the cross-category similarity results since they don't depend on the emotion labels at all.

---

## Stack

| Component | Technology |
|-----------|-----------|
| Language | Rust |
| ML framework | [candle](https://github.com/huggingface/candle) 0.10 |
| Model weights | HuggingFace Hub (`gpt2`) |
| Tokenizer | `tokenizers` crate |

---

## Running

```bash
git clone https://github.com/chromeblood/affine
cd affine
cargo run --release
```

Weights are downloaded automatically on first run (~500MB).

---

## What's Next

If this were to be extended into something more rigorous:

- Train linear probes via SGD rather than using centroid similarity
- Probe the emotion token position specifically rather than mean pooling
- Add baselines — random sentence pairs, shuffled labels
- Test on a larger, more diverse dataset
- Replicate on larger models to see if separation changes with scale
- Ablate emotion directions and measure output shift to test causal relevance

---

## Motivation

Built to learn mechanistic interpretability by doing rather than reading. The goal was to understand transformer internals deeply enough to build the tooling from scratch — the research question was secondary to that.

Related reading: Elhage et al. "A Mathematical Framework for Transformer Circuits", Anthropic's work on the superposition hypothesis.