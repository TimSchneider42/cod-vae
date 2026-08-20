# cod-vae: PyTorch/JAX Reimplementation of COD-VAE

Unofficial reimplementation of **COD-VAE** from ["Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models"](https://arxiv.org/abs/2503.08737) by In Cho, Youngbeom Yoo, Subin Jeon, and Seon Joo Kim (ICCV 2025).
COD-VAE is a 3D shape VAE that compresses a shape into a small set of latent vectors (e.g. 32 x 32) and decodes them back into an occupancy field via a transformer decoder with uncertainty-based token pruning.

This package is **not** the official implementation; the original code by the authors can be found at [join16/COD-VAE](https://github.com/join16/COD-VAE).
Compared to the original, this package provides:

- **Both PyTorch and JAX backends** behind a common numpy/[trimesh](https://trimsh.org/) interface that automatically selects the best available backend.
- **No compiled dependencies**: the `pointops` CUDA extension of the original is replaced by a pure implementation with identical results.
- **Self-contained weight files** (npz) loadable without torch, plus loading from and pushing to the **Hugging Face Hub**.
- **Training** (both stages, from scratch or fine-tuning) on either backend with **multi-GPU support**, on arbitrary meshes (watertight or not), preprocessed with the original authors' recipe — on the fly, or ahead of time via the bundled `cod-vae-dataset` tool, which also merges Hugging Face mesh datasets and the original preprocessed ShapeNet data.

Both backends have been validated against the original implementation using the officially released weights: encoder latents match to float32 round-off (bit-exact for the torch backend), decoded occupancy fields agree in sign on 100.0000% of grid points, and reconstructed meshes deviate by less than 1e-4 in units of the model's [-1, 1] cube.

## Installation

```bash
pip install cod-vae[OPTIONS]
```

where `OPTIONS` can be any subset of the following:

- `torch`: Install the PyTorch backend.
- `jax`: Install the JAX backend with CUDA 12 support.
- `jax-cpu`: Install the JAX backend without GPU support.
- `train`: Install training dependencies (optax; only needed for training with the JAX backend).
- `hub`: Install Hugging Face Hub support.
- `convert`: Install dependencies for converting original COD-VAE checkpoints (torch, pyyaml).
- `preprocess`: Install mesh preprocessing dependencies (point-cloud-utils, datasets), required for training on meshes and for `cod-vae-dataset`.
- `all`: Install all of the above.

Note that either `torch`, `jax`, or `jax-cpu` has to be chosen as a backend for cod-vae to work.
Depending on the installed CUDA version, [PyTorch](https://pytorch.org/get-started/locally/) and [JAX](https://docs.jax.dev/en/latest/installation.html) might have to be installed manually.

## Usage

```python
import trimesh
from cod_vae import CODVAE

vae = CODVAE.from_pretrained("TimSchneider42/cod-vae")  # Hugging Face Hub repo

mesh = trimesh.load("bunny.obj", force="mesh")
latent, transform = vae.encode_mesh(mesh, return_transform=True)  # (32, 32) numpy array
reconstruction = vae.decode_mesh(latent, transform=transform)     # trimesh.Trimesh
```

`CODVAE(...)` and its loaders return a backend-specific implementation, preferring JAX if installed and falling back to PyTorch; pass `backend="torch"` or `backend="jax"` to choose explicitly, and `device=...` to select a device.
All inputs and outputs of the public interface are numpy arrays and trimesh meshes, regardless of the backend.

Since the cube normalization removes the mesh's position and scale, the plain latent only describes the mesh's shape and orientation; the returned `transform` (bounding box center and isotropic scale) is required to map decoded geometry back into the original frame.
The *full latent* interface packs both into a single flat vector `[flattened latent, center (3), size (1)]`, which contains everything needed to reconstruct the mesh: the mesh's bounding box center and maximum half-extent, normalized by a `frame_half_size` describing the half-extent of the world frame (i.e. vertex positions divided by `frame_half_size` lie in `[-1, 1]`; the default of 1 means the mesh coordinates are used as-is).
Decoding a full latent maps queries given in this normalized world frame into the model's cube *inside the backend*, so the operation is differentiable with respect to the full latent — including its center and size entries — in the backend-native variants (`cod_vae.torch.CODVAEModule.decode_full`, `cod_vae.jax.decode_full`); the numpy methods dispatch to the model's backend just like `encode`/`decode`:

```python
full_latent = vae.encode_mesh_full(mesh)        # (num_latents * latent_dim + 4,)
reconstruction = vae.decode_mesh_full(full_latent)  # trimesh.Trimesh in the original frame
logits = vae.decode_full(full_latent, queries)  # logits at normalized-world-frame queries
latent, transform = vae.unpack_full_latent(full_latent)  # split it up again
full_latent = vae.pack_full_latent(latent, transform)    # ... and re-assemble it
```

Besides the mesh interface, latents can be computed from raw surface point clouds and decoded at arbitrary query points or into dense grids (all functions accept batched and unbatched inputs):

```python
latents = vae.encode(points)                   # (N, 3) or (B, N, 3) in [-1, 1]^3
logits = vae.decode(latents, queries)          # occupancy logits, positive inside
volume = vae.decode_volume(latents, resolution=128)  # dense logit grid
```

Models can be loaded from a Hugging Face Hub repo id, a local npz file, or an official COD-VAE release directory (`config.yaml` + `*.pt`, requires the `convert` extra), and saved/uploaded via `vae.save(path)` and `vae.push_to_hub("user/repo")`.

To convert an official release into the self-contained npz format (optionally uploading it):

```bash
cod-vae-convert path/to/vae_m32 vae_m32.npz --push-to-hub user/cod-vae
```

Note that the original authors have not attached a license to their released weights, so make sure you have their permission before re-hosting converted weights publicly.

### Pretrained models

A grid of models trained with this package is available on the Hugging Face Hub as `TimSchneider42/cod-vae-<num_latents>x<latent_dim>`, one repository per model:

```python
vae = CODVAE.from_pretrained("TimSchneider42/cod-vae-32x32")   # 32 x 32 = 1024 numbers per shape
```

| #latents \ latent-dim | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| **4** | [cod-vae-4x4](https://huggingface.co/TimSchneider42/cod-vae-4x4) | [cod-vae-4x8](https://huggingface.co/TimSchneider42/cod-vae-4x8) | [cod-vae-4x16](https://huggingface.co/TimSchneider42/cod-vae-4x16) | [cod-vae-4x32](https://huggingface.co/TimSchneider42/cod-vae-4x32) |
| **8** | [cod-vae-8x4](https://huggingface.co/TimSchneider42/cod-vae-8x4) | [cod-vae-8x8](https://huggingface.co/TimSchneider42/cod-vae-8x8) | [cod-vae-8x16](https://huggingface.co/TimSchneider42/cod-vae-8x16) | [cod-vae-8x32](https://huggingface.co/TimSchneider42/cod-vae-8x32) |
| **16** | [cod-vae-16x4](https://huggingface.co/TimSchneider42/cod-vae-16x4) | [cod-vae-16x8](https://huggingface.co/TimSchneider42/cod-vae-16x8) | [cod-vae-16x16](https://huggingface.co/TimSchneider42/cod-vae-16x16) | [cod-vae-16x32](https://huggingface.co/TimSchneider42/cod-vae-16x32) |
| **32** | [cod-vae-32x4](https://huggingface.co/TimSchneider42/cod-vae-32x4) | [cod-vae-32x8](https://huggingface.co/TimSchneider42/cod-vae-32x8) | [cod-vae-32x16](https://huggingface.co/TimSchneider42/cod-vae-32x16) | [cod-vae-32x32](https://huggingface.co/TimSchneider42/cod-vae-32x32) |
| **64** | [cod-vae-64x4](https://huggingface.co/TimSchneider42/cod-vae-64x4) | [cod-vae-64x8](https://huggingface.co/TimSchneider42/cod-vae-64x8) | [cod-vae-64x16](https://huggingface.co/TimSchneider42/cod-vae-64x16) | [cod-vae-64x32](https://huggingface.co/TimSchneider42/cod-vae-64x32) |

Rows are `num_latents`, columns are `latent_dim`; a shape is compressed into `num_latents` x `latent_dim` numbers, so the grid spans 16 (4x4) to 2048 (64x32) numbers per shape. `32x32` and `64x32` correspond to the released `vae_m32` and `vae_m64` configurations.
The largest model is additionally published under the short name [`TimSchneider42/cod-vae`](https://huggingface.co/TimSchneider42/cod-vae), which is what the examples above load; it is a copy of `cod-vae-64x32`, not a link to it, since the Hub has no aliasing between repositories.
Unlike the original models, they were not trained on ShapeNet alone, but on 110,077 shapes: the 48,597 ShapeNet training shapes plus 50,000 CAD meshes from ABC and all 11,480 MNIST3D meshes, both from [Tactile MNIST](https://github.com/TimSchneider42/tactile-mnist).
Otherwise the recipe is the paper's — see [TRAINING.md](TRAINING.md#how-the-published-cod-vae-nxm-models-were-trained) for the exact commands.

#### Reconstruction quality on ABC

Measured on the ABC test split, which no model saw during training, from the checkpoint each repository holds:

| **#latents** \ **latent-dim** | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| **4** | 0.671 / 0.712 | 0.743 / 0.758 | 0.804 / 0.797 | 0.852 / 0.829 |
| **8** | 0.727 / 0.748 | 0.806 / 0.793 | 0.854 / 0.829 | 0.887 / 0.851 |
| **16** | 0.782 / 0.770 | 0.873 / 0.835 | 0.903 / 0.863 | 0.916 / 0.875 |
| **32** | 0.831 / 0.808 | 0.900 / 0.858 | 0.919 / 0.877 | 0.925 / 0.886 |
| **64** | 0.856 / 0.826 | 0.915 / 0.874 | 0.927 / 0.887 | 0.932 / 0.893 |

Each cell is **volume IoU / near-surface accuracy**, averaged over 128 held-out meshes.
Both are computed on the decoded occupancy field rather than on a meshed reconstruction, against the query points of the original recipe: IoU over points drawn uniformly from the cube, and accuracy over points drawn near the surface.
The near-surface number is the harder of the two and the one that tracks fine detail, since points far from the surface are easy to classify and dominate the uniform sample.

Quality rises monotonically along both axes, but the two axes are not interchangeable: at a fixed budget of numbers per shape, a balanced split beats a lopsided one.
Of the four ways to spend 256 numbers, `16x16` is best at 0.903 and `64x4` is worst at 0.856, with `32x8` (0.900) and `8x32` (0.887) in between; the same ordering holds at 128 numbers, where `16x8` reaches 0.873 and `32x4` only 0.831.
A `latent_dim` of 4 is the weakest use of any budget, and the optimum sits around 8 to 16 with the remaining capacity spent on latents.
Returns also flatten toward the top-right corner: at `num_latents=64` the last doubling of latent width buys 0.005 IoU, against 0.059 for the first.

ABC is the harder of the two evaluation sources at every size but the smallest, where the ordering reverses (`4x4` scores 0.671 on ABC against 0.630 on MNIST3D) — with 16 numbers per shape the reconstruction is too coarse for the digit geometry to survive at all.
The corresponding MNIST3D figures are on each model card.

#### Decode-optimized small models

For pipelines where decoding speed matters — especially ones that backpropagate through the frozen decoder — a `-small` variant of each model is published under the same name plus the `-small` suffix: ~39M parameters instead of 188M (decode path ~20M instead of 90M), roughly **8x faster forward+backward** (23.6k vs 2.9k shapes/s at batch 1024 x 2048 queries, H100, JAX float16), measured ~9x end-to-end in a downstream RL loop that trains through the decoder.

| #latents \ latent-dim | 4 | 8 | 16 |
|---|---|---|---|
| **4** | [cod-vae-4x4-small](https://huggingface.co/TimSchneider42/cod-vae-4x4-small) | [cod-vae-4x8-small](https://huggingface.co/TimSchneider42/cod-vae-4x8-small) | [cod-vae-4x16-small](https://huggingface.co/TimSchneider42/cod-vae-4x16-small) |
| **8** | [cod-vae-8x4-small](https://huggingface.co/TimSchneider42/cod-vae-8x4-small) | [cod-vae-8x8-small](https://huggingface.co/TimSchneider42/cod-vae-8x8-small) | [cod-vae-8x16-small](https://huggingface.co/TimSchneider42/cod-vae-8x16-small) |
| **16** | [cod-vae-16x4-small](https://huggingface.co/TimSchneider42/cod-vae-16x4-small) | [cod-vae-16x8-small](https://huggingface.co/TimSchneider42/cod-vae-16x8-small) | [cod-vae-16x16-small](https://huggingface.co/TimSchneider42/cod-vae-16x16-small) |

Reconstruction quality on ABC, measured exactly as for the full-size grid above (**volume IoU / near-surface accuracy**, 128 held-out meshes):

| **#latents** \ **latent-dim** | 4 | 8 | 16 |
|---|---|---|---|
| **4** | 0.650 / 0.698 | 0.733 / 0.743 | 0.794 / 0.782 |
| **8** | 0.724 / 0.736 | 0.792 / 0.779 | 0.837 / 0.808 |
| **16** | 0.762 / 0.746 | 0.842 / 0.804 | 0.872 / 0.830 |

The ~8x speedup costs between 0.003 and 0.03 IoU against the full-size cell, generally less at smaller latent budgets (0.031 at `16x16`, 0.009 at `4x16`, 0.003 at `8x4`).
Each `-small` model has the same latent shape as its full-size counterpart — but a **different latent space**: latents from one cannot be decoded with the other.
See [TRAINING.md](TRAINING.md#how-the-published-cod-vae-16xm-small-models-were-trained) for the architecture and exact training commands.

## Training

Both stages of the paper's recipe can be trained with this package, on either backend and on multiple GPUs, either directly on a directory of arbitrary (not necessarily watertight) meshes or on a dataset built ahead of time with `cod-vae-dataset`:

```bash
cod-vae-train path/to/meshes checkpoints/stage1 --stage 1 --backend torch
cod-vae-train path/to/meshes checkpoints/stage2 --stage 2 --init-from checkpoints/stage1/checkpoint_last.npz
```

See [TRAINING.md](TRAINING.md) for the full guide: data preparation, both stages and their hyperparameters, multi-GPU launching, the Python API, replicating the original ShapeNet training, and the commands behind the published models above.

## Relation to the original implementation

The model code is reimplemented from scratch (MIT licensed), but faithfully mirrors the reference: parameters are stored as a flat mapping using the original state-dict names, so converted checkpoints remain transparently comparable, and all architectural quirks of the reference are reproduced exactly (see the module docstrings for details).
One deliberate deviation: during stage-2 training the frozen autoencoder runs deterministically, whereas the reference keeps stochastic depth active in it.

If you use this package in your research, please cite the original paper:

```bibtex
@inproceedings{cho2025cod,
  author={Cho, In and Yoo, Youngbeom and Jeon, Subin and Kim, Seon Joo},
  title={Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
