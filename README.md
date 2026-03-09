# MambaInterp 🐍🔬

> Transposing Transcoders on Mamba SSM. You can read more [here](https://lorenzoruggerii.github.io/blog/2026/mambacoder/).

---

## Overview

MambaInterp explores tools to peer inside Mamba SSM (State Space Model) architectures and understand *how* they process information. Inspired by transformer circuit-tracing methods (e.g. [Transcoders's original paper](https://arxiv.org/abs/2406.11944), [Anthropic CLT](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)), this repo adapts those techniques to Mamba's unique selective-scan mechanism.

The core idea: train **MambaCoders** (transcoders) on each layer of a Mamba model, then use them to trace which features — across layers and token positions — are causally responsible for a given model output.

---

## Features

- 🔁 **Circuit Tracing** — Greedy and normalized path-finding through Mamba feature circuits, with support for novelty-penalized exploration
- 🧠 **MambaCoder Transcoders** — Sparse autoencoders that decompose Mamba hidden states into interpretable features
- 🏋️ **Fine-tuning** — Train Mamba models from scratch or fine-tune on custom datasets
- 📓 **Interpretability Notebook** — End-to-end demo extracting and visualizing feature circuits on TinyStories
- ⚙️ **Configurable** — Clean dataclass-based config for transcoders, with TopK sparsity and scheduler support

---

## Project Structure

```
src/
├── __init__.py
├── circuit_tracing.py     # Core circuit tracing logic (feature attribution, path search)
├── config.py              # TranscoderConfig and TopK activation
├── finetune.py            # Mamba fine-tuning script with CLI args
├── interpretability.ipynb # Demo notebook: TinyStories circuit analysis
├── mambacoder.py          # MambaCoder transcoder model
├── model.py               # Mamba model wrapper
└── utils.py               # Utilities (e.g. most_act_diff)
```

---

## Installation

```bash
git clone https://github.com/your-username/mamba-interp.git
cd mamba-interp
pip install -r requirements.txt
```

**Dependencies include:**
- `torch`
- `transformers`
- `datasets`
- `mamba_ssm` (or compatible Mamba implementation)
- `numpy`

---

## Quick Start

### 1. Fine-tune a Mamba model

```bash
python src/finetune.py \
  --lr 1e-4 \
  --bs 8 \
  --dataset roneneldan/TinyStories \
  --output_dir outputs/mamba-tinystories \
  --logging_dir logs/ \
  --epochs 3 \
  --layers 4
```

### 2. Train a MambaCoder transcoder

Configure via `TranscoderConfig` in `src/config.py`, then use `MambaCoder` from `src/mambacoder.py`.

### 3. Run circuit tracing

```python
from circuit_tracing import greedy_get_top_paths, FeatureVector, Component, ComponentType
from mambacoder import MambaCoder

# Load your transcoder and run a forward pass to populate the cache
# ...

# Define a starting feature vector
f = FeatureVector(
    component_path=[],
    vector=my_feature_vec,
    layer=3,
    sublayer="resid_pre",
    token=10
)

# Trace top causal paths
paths = greedy_get_top_paths(transcoder, cache, f, num_iters=3, num_branches=5)
```

### 4. Interactive exploration

Open `src/interpretability.ipynb` to explore feature circuits on TinyStories with visualizations.

---

## Key Concepts

### MambaCoder (Transcoder)

A transcoder is a layer-wise sparse autoencoder that maps Mamba hidden states into a higher-dimensional, interpretable feature space. Each layer has an encoder and decoder; features are activated sparsely via a **TopK** gating function.

### Circuit Tracing

Given a feature vector at layer `L` and token `t`, the tracing algorithm attributes its value to:

1. **MambaCoder features** from earlier layers — via encoder/decoder weight products
2. **Mamba SSM heads** — via an explicit attention matrix computed from the selective scan's `A`, `B`, and `C` matrices
3. **Embedding contributions** — from the token embedding at position `t`

The `greedy_get_top_paths_normalized` function adds a novelty penalty to encourage diverse, non-redundant explanations.

### Attention Matrix from SSM

Following [Ali et al., 2024](https://arxiv.org/abs/2403.01590) the SSM's recurrence is unrolled into an explicit `(B, ED, L, L)` attention matrix, enabling token-to-token attribution analogous to transformer attention.

---

## Configuration

```python
from config import TranscoderConfig

cfg = TranscoderConfig(
    weight_path="path/to/base/model",
    save_path="models/MambaCoders/mambacoder.pt",
    num_features=12_288,   # 16× hidden size
    topk_features=32,
    lr=1e-4,
    batch_size=4,
    num_epochs=2,
    num_train_prompts=250_000,
    tokenizer_path="state-spaces/mamba-130m-hf"
)
```

## License

MIT
