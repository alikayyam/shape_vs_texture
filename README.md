
# Shape vs. Texture Bias in Deep Neural Networks

A research project investigating whether neural networks learn shape-based or texture-based features for image classification, using controlled synthetic datasets designed to decouple these two visual cues.

![Shape vs Texture](https://raw.githubusercontent.com/alikayyam/shape_vs_texture/main/shapetexture.png)
**Figure: Three experimental paradigms for probing shape vs. texture bias. (A) Cue-conflict (Geirhos et al., 2018): shape and texture from different categories placed in conflict. (B) Cue-suppression (Burgert et al., 2025): residual cue leakage limits interpretability. (C) Baker et al. (2018) design adopted here: texture restricted to object region, background kept neutral.**


## Overview

Neural networks trained on natural images are known to exhibit biases toward either shape or texture features. This project isolates that question by generating synthetic images where:

- **Training**: images contain both a shape (circle or square) and a correlated texture (dots or lines)
- **Testing**: images are evaluated under 7 conditions including matched, cue-conflict, texture-only, and shape-only variants

The "cue conflict" condition — where the shape label and texture label are inverted — is the key diagnostic: a model relying on shape will succeed, while one relying on texture will fail.

## Experiments

| Notebook | Description |
|---|---|
| `shape_texture_synthetic.ipynb` | Core experiment: CNN, MLP, and ViT on synthetic shape+texture data |
| `shape_texture_synthetic_sparse.ipynb` | Sparse CNN (top-k activation) variant |
| `shape_texture_synthetic_sparse_capsule.ipynb` | Capsule Network variant |
| `shape_cross.ipynb` | Binary shape classification with associated textures |
| `shape_cross_sparse.ipynb` | Sparse CNN on cross experiment |
| `shape_random_texture.ipynb` | Synthetic shapes with randomly composed textures |
| `shape_random_texture_sparse.ipynb` | Sparse model on MNIST/FashionMNIST with synthetic textures |

## Test Conditions

Each experiment evaluates the trained model across 7 conditions:

| Condition | Description |
|---|---|
| A | Shape + matched texture (training distribution) |
| B | Cue conflict: shape label switched, opposite texture applied |
| C | Texture/background only (no shape) |
| D | Shape edges on white background (no texture) |
| E | Shape + random texture |
| F | Shape + dot texture |
| G | Shape + line texture |

## Models

- **SimpleCNN**: 2-layer convolutional network with max pooling
- **SimpleMLP**: Fully-connected baseline (784→128→64→2)
- **SimpleViT**: Vision Transformer with 4×4 patch embedding, 4 layers, 4 heads
- **SimpleTopKNet**: Sparse CNN applying top-k activation selection (default: top 10%)
- **SimpleCapsNet**: Capsule network with routing-by-agreement (3 iterations)

## Key Results

**Synthetic shape+texture (CNN, 10-run mean ± std):**
- Test A (matched): 93.1% ± 0.9%
- Test B (cue conflict): 19.9% ± 0.9% — models rely heavily on texture
- Test C (texture only): 81.4% ± 9.1%
- Test D (edges only): 62.5% ± 1.2%

**Capsule network differences:**
- Test D (edges): 91.2% ± 17.3% — substantially stronger shape understanding
- Test C (texture only): 100% — also captures texture; different failure modes than CNNs

**Sparsity effect:** Top-k sparse networks amplify the dominant feature bias compared to dense counterparts.

## Setup

```bash
pip install torch torchvision numpy opencv-python matplotlib
```

GPU recommended (experiments use CUDA). Datasets:
- **MNIST / FashionMNIST**: downloaded automatically via `torchvision`
- **Synthetic shapes**: generated on-the-fly by each notebook

## Texture Generation

Two synthetic texture types are used:

- **Dot texture**: single-pixel dots on a regular grid with random offset
- **Line/grating texture**: parallel horizontal lines rotated by a random angle (0–45°)

Textures are assigned per class label, so the network can learn either the shape boundary or the background pattern to solve the training task.

> **Note:** Accurate performance results for all experiments are reported in the associated paper (see citation below).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kayyam2026cnns,
  title     = {Position: {CNN}s Don't See Shape --- And That Won't Change Without New Architectures},
  author    = {Kayyam, Ali},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Related Work

- Geirhos et al. (2019) — *ImageNet-trained CNNs are biased towards texture*
- Sabour et al. (2017) — *Dynamic Routing Between Capsules*
- Brendel & Bethge (2019) — *Approximating CNNs with bag-of-local-features models*
