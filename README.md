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

### Pretrained models

A grid of models trained with this package is available on the Hugging Face Hub as `TimSchneider42/cod-vae-<num_latents>x<latent_dim>`, one repository per model:

```python
vae = CODVAE.from_pretrained("TimSchneider42/cod-vae-32x32")   # 32 x 32 = 1024 numbers per shape
```

| | `latent_dim` 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| **`num_latents` 4** | [cod-vae-4x4](https://huggingface.co/TimSchneider42/cod-vae-4x4) | [4x8](https://huggingface.co/TimSchneider42/cod-vae-4x8) | [4x16](https://huggingface.co/TimSchneider42/cod-vae-4x16) | [4x32](https://huggingface.co/TimSchneider42/cod-vae-4x32) |
| **8** | [8x4](https://huggingface.co/TimSchneider42/cod-vae-8x4) | [8x8](https://huggingface.co/TimSchneider42/cod-vae-8x8) | [8x16](https://huggingface.co/TimSchneider42/cod-vae-8x16) | [8x32](https://huggingface.co/TimSchneider42/cod-vae-8x32) |
| **16** | [16x4](https://huggingface.co/TimSchneider42/cod-vae-16x4) | [16x8](https://huggingface.co/TimSchneider42/cod-vae-16x8) | [16x16](https://huggingface.co/TimSchneider42/cod-vae-16x16) | [16x32](https://huggingface.co/TimSchneider42/cod-vae-16x32) |
| **32** | [32x4](https://huggingface.co/TimSchneider42/cod-vae-32x4) | [32x8](https://huggingface.co/TimSchneider42/cod-vae-32x8) | [32x16](https://huggingface.co/TimSchneider42/cod-vae-32x16) | [32x32](https://huggingface.co/TimSchneider42/cod-vae-32x32) |
| **64** | [64x4](https://huggingface.co/TimSchneider42/cod-vae-64x4) | [64x8](https://huggingface.co/TimSchneider42/cod-vae-64x8) | [64x16](https://huggingface.co/TimSchneider42/cod-vae-64x16) | [64x32](https://huggingface.co/TimSchneider42/cod-vae-64x32) |

A shape is compressed into `num_latents` x `latent_dim` numbers, so the grid spans 16 (4x4) to 2048 (64x32) numbers per shape; `32x32` and `64x32` correspond to the released `vae_m32` and `vae_m64` configurations.
The grid is still training — a repository appears once its run finishes, and each model card states the exact state of the checkpoint it holds.
They were trained on ShapeNet plus [Tactile MNIST](https://github.com/TimSchneider42/tactile-mnist) meshes rather than on ShapeNet alone — see [TRAINING.md](TRAINING.md#how-the-published-cod-vae-nxm-models-were-trained) for the dataset, the recipe, and the exact commands, and each model card for its held-out reconstruction quality.
To convert an official release into the self-contained npz format (optionally uploading it):

```bash
cod-vae-convert path/to/vae_m32 vae_m32.npz --push-to-hub user/cod-vae
```

Note that the original authors have not attached a license to their released weights, so make sure you have their permission before re-hosting converted weights publicly.

## Training

Training follows the two-stage recipe of the paper:

1. **Stage 1** trains the autoencoder (point cloud encoder, triplane decoder with uncertainty-based token pruning, occupancy head) with occupancy reconstruction losses on both the refined and the initial prediction, plus a supervision loss for the uncertainty head.
2. **Stage 2** freezes the autoencoder and trains the latent VAE modules (`latent_proj_in`/`latent_proj_out`/`latent_decoder`) with a feature matching loss, the reconstruction loss through the frozen decoder, and a KL term.

Both stages consume the same kind of training data: per shape, pools of surface points, uniform volume queries, and near-surface queries with ground-truth **occupancy labels**, from which random subsamples are drawn each step with the reference's anisotropic scaling augmentation.
These pools are always produced by one and the same preprocessing — the recipe the original authors used to build their ShapeNet training data ([sdf_gen](https://github.com/1zb/sdf_gen)): watertighting via [point_cloud_utils](https://github.com/fwilliams/point-cloud-utils) (so your meshes do **not** need to be watertight), normalization into the [-1, 1] cube, and sampling of the query pools with occupancy labels. It requires the `preprocess` extra (`pip install cod-vae[preprocess]`) and can run in two ways:

1. **On the fly**: point `cod-vae-train` at a directory of meshes; the pools are computed lazily during the first epoch.
2. **Ahead of time**: `cod-vae-dataset` builds a dataset on disk — preprocess once, train many times, and merge multiple sources (mesh directories, Hugging Face mesh datasets, the original preprocessed ShapeNet data) with train/val/test splits.

### Option 1: training directly on meshes

```bash
cod-vae-train path/to/meshes checkpoints/stage1 --stage 1 --backend torch
cod-vae-train path/to/meshes checkpoints/stage2 --stage 2 --init-from checkpoints/stage1/checkpoint_last.npz
```

The occupancy pools are computed on the fly; pass `--cache-dir` to reuse them across runs.
Checkpoints are self-contained npz files loadable by both backends (and by `CODVAE.load`).

### Option 2: building a dataset with cod-vae-dataset

`cod-vae-dataset` builds a training dataset on disk by merging any number of sources:

- `--meshes [NAME=]DIR`: a directory of mesh files. Meshes in `train`/`val`/`test` subdirectories are assigned to the corresponding splits; otherwise everything becomes training data.
- `--hf [NAME=]DATASET`: a Hugging Face mesh dataset in the [Tactile MNIST](https://github.com/TimSchneider42/tactile-mnist) format (rows with `mesh.vertices`/`mesh.faces` columns), given as a Hub repository id or a local path. By default the `train`/`val`(`idation`)/`test` splits are used, as far as present; `--hf-split SRC[=DST]` selects and remaps splits explicitly (e.g. `--hf-split holdout=val`).
- `--vecset PATH`: an existing preprocessed root as distributed by the 3DShape2VecSet authors, merged as-is (symlinked by default; `--link hardlink|copy` to materialize).

Every source accepts an optional `:FRACTION` suffix (e.g. `--hf TimSchneider42/tactile-mnist-mnist3d:0.1`) to keep only a deterministic random subsample of each split; the selection is controlled by `--seed` (default 0), so the same command always yields the same subset.
Preprocessing is resumable (existing outputs are skipped unless `--overwrite` is given) and parallelizes with `--workers`, each of which gets its own slice of the available cores (the watertighting and winding-number code parallelizes internally over all cores, which otherwise oversubscribes the machine badly).
Large builds can additionally be spread over several machines with `--shard INDEX/COUNT`: every shard preprocesses its share of the meshes, and a final run without `--shard` links the `--vecset` sources and writes the `.lst` files.
A mesh that fails preprocessing aborts the build; pass `--skip-failed` to instead drop failing meshes with a warning (the behavior of the original sdf_gen script).
Watertighting a pathological mesh can also run for hours without ever raising, which stalls an otherwise finished build; `--timeout SECONDS` treats those as failures too.

```bash
cod-vae-dataset data/merged \
    --meshes path/to/my_meshes \
    --hf TimSchneider42/tactile-mnist-mnist3d \
    --vecset path/to/shapenet_vecset_root \
    --workers 16
```

Training then points at the built dataset instead of a mesh directory — everything else works exactly as in option 1:

```bash
cod-vae-train data/merged checkpoints/stage1 --stage 1 --backend torch
```

`cod-vae-train` detects the input type automatically: a directory containing `ShapeNetV2_point/` is treated as a built dataset, anything else as a directory of mesh files.

Training can also be driven from Python (both options); see [TRAINING.md](TRAINING.md), which also covers replicating the original models' training on ShapeNet.

### Multi-GPU training

- **PyTorch**: standard DistributedDataParallel; launch with `torchrun`, e.g. `torchrun --nproc_per_node=4 -m cod_vae.cli train path/to/data checkpoints --stage 1`.
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
