# cod-vae: PyTorch/JAX Reimplementation of COD-VAE

Unofficial reimplementation of **COD-VAE** from ["Representing 3D Shapes with 64 Latent Vectors for 3D Diffusion Models"](https://arxiv.org/abs/2503.08737) by In Cho, Youngbeom Yoo, Subin Jeon, and Seon Joo Kim (ICCV 2025).
COD-VAE is a 3D shape VAE that compresses a shape into a small set of latent vectors (e.g. 32 x 32) and decodes them back into an occupancy field via a transformer decoder with uncertainty-based token pruning.

This package is **not** the official implementation; the original code by the authors can be found at [join16/COD-VAE](https://github.com/join16/COD-VAE).
Compared to the original, this package provides:

- **Both PyTorch and JAX backends** behind a common numpy/[trimesh](https://trimsh.org/) interface that automatically selects the best available backend.
- **No compiled dependencies**: the `pointops` CUDA extension of the original is replaced by a pure implementation with identical results.
- **Self-contained weight files** (npz) loadable without torch, plus loading from and pushing to the **Hugging Face Hub**.
- **Training** (both stages, from scratch or fine-tuning) on either backend with **multi-GPU support**, directly on watertight meshes — no preprocessed dataset format required.

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

## Training

Training follows the two-stage recipe of the paper:

1. **Stage 1** trains the autoencoder (point cloud encoder, triplane decoder with uncertainty-based token pruning, occupancy head) with occupancy reconstruction losses on both the refined and the initial prediction, plus a supervision loss for the uncertainty head.
2. **Stage 2** freezes the autoencoder and trains the latent VAE modules (`latent_proj_in`/`latent_proj_out`/`latent_decoder`) with a feature matching loss, the reconstruction loss through the frozen decoder, and a KL term.

Training data is generated directly from **watertight meshes**: per mesh, pools of surface points, uniform volume queries, and near-surface queries with ground-truth occupancy labels are computed (and optionally cached), from which random subsamples are drawn each step with the reference's anisotropic scaling augmentation.

```python
from cod_vae import CODVAEConfig
from cod_vae.training import MeshOccupancyDataset, TrainingConfig
from cod_vae.torch.training import train  # or: from cod_vae.jax.training import train

config = CODVAEConfig()  # architecture of the released vae_m32
dataset = MeshOccupancyDataset(mesh_files, repeat=16, cache_dir="occupancy_cache")

params = train(config, TrainingConfig(stage=1), dataset, out_dir="checkpoints/stage1")
params = train(config, TrainingConfig(stage=2), dataset, params=params, out_dir="checkpoints/stage2")
```

The resulting parameters are a flat numpy dict compatible with both backends (`CODVAE(config, params)`, `save_npz`, `push_to_hub`).
A command line interface is available as well:

```bash
cod-vae-train path/to/meshes checkpoints --stage 1 --backend torch
```

### Multi-GPU training

- **PyTorch**: standard DistributedDataParallel; launch with `torchrun`, e.g. `torchrun --nproc_per_node=4 -m cod_vae.cli train path/to/meshes checkpoints --stage 1`.
- **JAX**: single-process data parallelism across all visible devices; just run the training script and it will shard batches over all GPUs automatically.

In both cases `batch_size` is per device and the learning rate is scaled by the effective batch size, following the reference implementation.

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
