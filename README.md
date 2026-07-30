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
The grid is still training — a repository appears once its run finishes, and each model card states the exact state of the checkpoint it holds.
Unlike the original models, they were not trained on ShapeNet alone, but on 110,077 shapes: the 48,597 ShapeNet training shapes plus 50,000 CAD meshes from ABC and all 11,480 MNIST3D meshes, both from [Tactile MNIST](https://github.com/TimSchneider42/tactile-mnist).
Otherwise the recipe is the paper's — see [TRAINING.md](TRAINING.md#how-the-published-cod-vae-nxm-models-were-trained) for the exact commands, and each model card for the model's held-out reconstruction quality.

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
