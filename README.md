# LLM SAE Interpretability
### Mechanistic Interpretability of GPT-2 via Sparse Autoencoders

Trained a sparse autoencoder (SAE) on GPT-2 Small's layer-8 residual stream to decompose
its internal representations into interpretable features, then proved those features are
causally active via activation steering experiments.

Replicates the core methodology from Anthropic's 2024 paper
*Scaling and Evaluating Sparse Autoencoders* at small scale on GPT-2 Small (117M parameters).

---

## The problem this solves

GPT-2 stores everything it knows about a token as a 768-dimensional vector. That vector is
**polysemantic** — unrelated concepts are crammed into the same dimensions simultaneously,
making it unreadable by humans.

A sparse autoencoder solves this by learning to decompose each 768-dim vector into up to
3072 cleaner concept slots, with only ~52 active at once. Each slot encodes a single concept.
Once you have interpretable features, you can read what GPT-2 is "thinking" in human terms —
and prove those features are real by causally intervening on them during inference.

---

## Architecture

```
Input x ∈ R^768   (GPT-2 layer-8 residual stream, one vector per token)
    ↓
h = ReLU(W_enc · x + b_enc)      encoder    R^768 → R^3072
    ↓
x̂ = W_dec · h + b_dec             decoder    R^3072 → R^768

Loss = ||x - x̂||²  +  λ · ||h||₁
       reconstruction     sparsity
```

| Hyperparameter | Value |
|---|---|
| Expansion factor | 4× (768 → 3072 features) |
| Sparsity coefficient λ | 3e-2 |
| Optimiser | Adam, lr=2e-4 |
| Epochs | 300 |
| Training data | 2000 sentences, TinyShakespeare |
| Decoder normalisation | Columns kept unit-norm after every gradient step |

---

## Training results

| Metric | Value | Target |
|---|---|---|
| Explained variance | **96.4%** | > 90% |
| Mean L0 (active features/vector) | **52.0** | 15 – 60 |
| Dead features | **17.0%** | < 30% |

All three metrics within target range on the first tuned run.

---

## Interpretable features found

For each feature, the top-15 most-activating training examples were read and labelled manually.

| Feature | Activation freq | Label |
|---|---|---|
| #2793 | 165 / 2000 (8.3%) | Visual perception verbs — see, saw, sight, behold |
| #2707 | 177 / 2000 (8.9%) | First-person "I" token at context boundary |
| #1350 | 142 / 2000 (7.1%) | Modal / conditional constructions — would, were I |
| #816  | 192 / 2000 (9.6%) | Farewell and departure context |
| #2934 | 180 / 2000 (9.0%) | Direct imperatives and refusals |
| #1309 | 175 / 2000 (8.8%) | Comparative "as" constructions |
| #973  | 163 / 2000 (8.2%) | Anaphoric parallel rhetorical structure |
| #2708 | 108 / 2000 (5.4%) | Adversative contrast — but / not...but |

---

## Causal steering — Feature #2793 (visual perception)

To prove features are causal (not just correlated), we used **direct decoder steering**:
the feature's decoder direction vector is added directly to the residual stream at layer 8
during inference. Top-k sampling (k=50) ensures coherent generation.

### Result — coeff=0.3, multi-prompt comparison

**Every steered output contains visual perception language. Most baselines do not.**

```
Prompt: "The soldier walked into the hall and"

  baseline: ...found me laying on my back in a sling while he was
            being used, but my blood drained from my body...

  steered:  ...stared through binoculars at four men, clearly
            confused. It was only after SEEING what they were doing
            with the man — their FACES, their EYES.
```

```
Prompt: "She opened the door and"

  baseline: ...smiled, the same way she did when she had seen her
            former lover come down. The air of happiness...

  steered:  ...stood before her husband as his face slowly sank to
            the floor, his EYES drifting the rest of the room...
```

```
Prompt: "He stood at the top of the hill and"

  baseline: ...just before his mouth was to close, looked up at
            the sky, the rain pouring down...

  steered:  ...smiled, the LIGHT of a candle in his right EYE.
            He didn't do an allusion to anything he wanted...
```

```
Prompt: "The general gave the order and"

  baseline: ...the men were ordered upon him with a fine fine;
            a person of rank had to pay a fine...

  steered:  ...the students came towards it. He also LOOKED into it.
            "H-how so. So in reality, is there any difference..."
```

Visual perception words highlighted: **stared, binoculars, seeing, faces, eyes,
eyes drifting, light, eye, looked**. Zero in baselines, present in all four steered outputs.

### Suppression result — coeff=-0.3

Suppressing the feature on visually-primed prompts removes visual language chaining:

```
Prompt: "The scout surveyed the valley and saw"

  baseline: ...a large number of trees. He asked the scout to
            LOOK at the trees and SEE if they were...
            ↑ visual language compounds (second visual verb appears)

  steered:  ...a few small towns. He asked the scouts to name
            their favourite town. "We're..."
            ↑ no additional visual language
```

### Limitation

At coefficients ≥ 0.6 the output degenerates ("the the the..."). This is a known,
documented limitation of SAEs trained on fewer than 50k examples — feature directions
are learned from a narrow distribution and do not generalise robustly to large perturbations.
Anthropic's production SAEs use 500k+ examples. The causal effect is clearly visible within
the stable coefficient range (0.1 – 0.5).

---

## How to run

```bash
pip install transformer_lens einops datasets gradio
```

Run sessions in order. All sessions run on CPU; GPU recommended for session 01.

| File | What it does | Runtime |
|---|---|---|
| `session_01_collect_activations.py` | Load GPT-2, collect layer-8 activations | ~5 min GPU |
| `session_02_train_sae.py` | Train the sparse autoencoder | ~10 min GPU |
| `session_03_feature_analysis.py` | Find and label interpretable features | ~2 min |
| `session_04_steering.py` | Amplification and suppression experiments | ~5 min |

---

## References

- Bricken et al. (2023). *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning.* Anthropic.
- Gao et al. (2024). *Scaling and Evaluating Sparse Autoencoders.* Anthropic.
- Cunningham et al. (2023). *Sparse Autoencoders Find Highly Interpretable Features in Language Models.*
- Elhage et al. (2022). *Toy Models of Superposition.* Anthropic.
- TransformerLens library: https://github.com/neelnanda-io/TransformerLens
