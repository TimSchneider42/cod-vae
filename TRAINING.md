# Training COD-VAE on ShapeNet

This guide replicates the training of the released COD-VAE checkpoints (vae_m32 / vae_m64) as closely as this package allows, following the recipe of the reference implementation.
See the last section for the known differences.

## 1. Data

The original models were trained on the preprocessed ShapeNet dataset of [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet): 55 categories, roughly 35k training shapes, each with pools of ~100k surface points and ~100k volume/near-surface query points with ground-truth occupancy labels.
Download it following the instructions in the 3DShape2VecSet repository and arrange it as (this is the layout the reference implementation expects as well):

```
{root_dir}/
    ShapeNetV2_point/
        {synset_id}/
            train.lst / val.lst / test.lst
            {object_id}.npz    # vol_points, vol_label, near_points, near_label
            {object_id}.npy    # surface normalization factor
    ShapeNetV2_surface/
        {synset_id}/4_pointcloud/{object_id}.npz   # surface points
```

`cod_vae.training.ShapeNetVecSetDataset` reads this layout directly — no HDF5 conversion step is needed.
Per training sample it draws 2048 surface points and 4096 volume + 4096 near-surface queries and applies the reference's anisotropic scaling augmentation, matching the original data pipeline.

> **Training on your own data instead:** if you do not need ShapeNet, `cod_vae.training.MeshOccupancyDataset` computes the same kind of occupancy pools on the fly from arbitrary meshes, using the same sdf_gen preprocessing (see section 5).

### Custom and mixed datasets

The `cod-vae-dataset` tool (see README, "Option 2: building a dataset with cod-vae-dataset") builds a dataset in exactly this layout from any mix of preprocessed 3DShape2VecSet roots, directories of (not necessarily watertight) mesh files, and Hugging Face mesh datasets in the [Tactile MNIST](https://github.com/TimSchneider42/tactile-mnist) format, applying the original authors' [sdf_gen](https://github.com/1zb/sdf_gen) preprocessing to the meshes:

```bash
cod-vae-dataset {root_dir} --vecset path/to/shapenet_vecset_root --meshes path/to/my_meshes --hf TimSchneider42/tactile-mnist-mnist3d --workers 16
```

The rest of this guide applies unchanged to such a merged `{root_dir}`.

## 2. Stage 1 — autoencoder

Stage 1 trains the point cloud encoder, the triplane decoder with uncertainty-based token pruning, and the occupancy head.
Reference hyperparameters (the defaults of `TrainingConfig(stage=1)` / `examples/train_shapenet.py`):

| Hyperparameter    | Value                                                                                              |
|-------------------|----------------------------------------------------------------------------------------------------|
| Epochs            | 100 (dataset repeated 16x per epoch)                                                               |
| Batch size        | 32 per GPU, 2x gradient accumulation                                                               |
| Optimizer         | AdamW (weight decay 0.01), constant LR                                                             |
| Learning rate     | 1e-4 x effective_batch_size / 256 (scaled automatically)                                           |
| Gradient clipping | 0.5 (global norm)                                                                                  |
| Losses            | occupancy BCE (volume 1.0, near 0.1) on refined + initial prediction, uncertainty MSE (coeff 0.01) |
| Stochastic depth  | 0.1                                                                                                |

On 16 GPUs:

```bash
torchrun --nproc_per_node=16 examples/train_shapenet.py {root_dir} checkpoints/stage1 \
    --stage 1 --num-latents 32   # 64 for the vae_m64 variant
```

The learning rate is scaled by the effective batch size following the reference's rule, so the GPU count does not need to match theirs.
With 16 GPUs the effective batch is 32 x 2 x 16 = 1024; pass `--accumulate 1` if you prefer to stay at an effective batch of 512.
Checkpoints (`checkpoint_epoch_*.npz`, `checkpoint_last.npz`) are written to the output directory after every epoch and are loadable by both backends.

Three flags matter for long runs on the torch backend:

- `--resume` continues in the output directory where the last completed epoch left off, restoring the optimizer and LR schedule from `train_state_last.pt` (written atomically after each epoch). A run that outlives a scheduler's time limit, or dies on a node failure, just gets resubmitted.
- `--tf32` allows TF32 matmuls and convolutions on Ampere+ GPUs: 1.65x faster than full float32 on an H100, and still more precise than the reference's 16-mixed setup.
- `--repeat` sets how often the dataset is repeated per epoch (default 16, the reference's value for ~35k ShapeNet shapes). On a substantially larger dataset, lowering it keeps an epoch — and therefore the 100-epoch schedule — comparable in size.

For the JAX backend, run the same script with `--backend jax` from a single process; it shards batches across all visible GPUs automatically.

## 3. Stage 2 — latent VAE

Stage 2 freezes the stage-1 autoencoder and trains only the latent compression modules (`latent_proj_in`, `latent_proj_out`, `latent_decoder`) with the feature matching loss (coeff 1.0), the occupancy reconstruction loss through the frozen decoder (coeff 1.0), and the KL term (effective coeff 1e-6; the reference config nominally says 1e-3 but applies it twice).

| Hyperparameter | Value                                                |
|----------------|------------------------------------------------------|
| Epochs         | 100                                                  |
| Batch size     | 128 per GPU, no accumulation                         |
| LR schedule    | 1e-4 (scaled as above), halved at epochs 60/70/80/90 |

```bash
torchrun --nproc_per_node=16 examples/train_shapenet.py {root_dir} checkpoints/stage2 \
    --stage 2 --init-from checkpoints/stage1/checkpoint_last.npz
```

Since the encoder pass runs without gradients and only the small latent modules are trained, stage 2 is considerably faster per epoch than stage 1.

### Latent size: what belongs to which stage

A shape is compressed into `num_latents` x `latent_dim` numbers, and the two factors are not interchangeable:

- **`num_latents`** is a token count. The encoder attends from that many farthest-point queries and the triplane decoder attends over them, so the autoencoder is trained *through* it — a different count needs its own stage-1 run. No parameter shape depends on it, and passing `--num-latents` together with a mismatching `--init-from` is refused rather than silently ignored.
- **`latent_dim`** is the width of each latent vector. It shapes only `latent_proj_in.1` and `latent_proj_out.0`, both of which stage 1 leaves at their initial values. So **one stage-1 checkpoint serves any number of latent widths**: pass `--latent-dim` to stage 2 and those two projections are re-initialized (`cod_vae.init.adapt_params`) while all 137M autoencoder parameters carry over.

```bash
# One autoencoder, four latent widths -- each a complete stage-2 run.
for dim in 4 8 16 32; do
    torchrun --nproc_per_node=4 examples/train_shapenet.py {root_dir} checkpoints/stage2_d$dim \
        --stage 2 --init-from checkpoints/stage1/checkpoint_last.npz --latent-dim $dim
done
```

## 4. Using and publishing the result

```python
from cod_vae import CODVAE

vae = CODVAE.load("checkpoints/stage2/checkpoint_last.npz")
vae.push_to_hub("you/cod-vae-m32")  # weights you trained yourself are yours to host
```

A quick qualitative check: encode and decode a few validation shapes (`ShapeNetVecSetDataset(root_dir, split="val")` provides surface points and labeled queries, so occupancy accuracy/IoU can be computed by comparing `vae.decode(latents, queries) > 0` against the labels).

## 5. Training from Python

The CLI is a thin wrapper around the Python API; both data options plug into the same `train` functions and use the same sdf_gen preprocessing.
Directly on meshes (watertight or not), with the occupancy pools computed on the fly (and optionally cached; pass `settings=SdfGenSettings(...)` to adjust the preprocessing):

```python
from cod_vae import CODVAEConfig
from cod_vae.training import MeshOccupancyDataset, TrainingConfig
from cod_vae.torch.training import train  # or: from cod_vae.jax.training import train

config = CODVAEConfig()  # architecture of the released vae_m32
dataset = MeshOccupancyDataset(mesh_files, repeat=16, cache_dir="occupancy_cache")

params = train(config, TrainingConfig(stage=1), dataset, out_dir="checkpoints/stage1")
params = train(config, TrainingConfig(stage=2), dataset, params=params, out_dir="checkpoints/stage2")
```

On preprocessed data — a dataset built with `cod-vae-dataset` or the original authors' ShapeNet root — only the dataset class changes; everything else stays the same:

```python
from cod_vae.training import ShapeNetVecSetDataset

dataset = ShapeNetVecSetDataset("data/merged", split="train", repeat=16)
```

Here no caching is needed (the pools are read straight from disk), `categories=[...]` restricts training to a subset of the merged sources, and `split="val"`/`"test"` instantiates held-out splits for evaluation.
In both cases the resulting `params` are a flat numpy dict compatible with both backends (`CODVAE(config, params)`, `save_npz`, `push_to_hub`).

## How the published cod-vae-NxM models were trained

The `TimSchneider42/cod-vae-<num_latents>x<latent_dim>` models on the Hugging Face Hub (see the README) were produced with the commands below.
They differ from the original models in their training data: ShapeNet plus two [Tactile MNIST](https://github.com/TimSchneider42/tactile-mnist) mesh datasets, 110,077 shapes in total.

**1. Build the merged dataset.**
The pool sizes are scaled per source, since a CAD assembly needs a finer occupancy sampling than an embossed digit, and ShapeNet's own pools hold 500k volume and 500k near-surface points:

```bash
# ShapeNet (55 synsets, 48,597 training shapes), linked in as-is
cod-vae-dataset data/merged --vecset path/to/shapenet_vecset_root

# 50,000 of the 204,617 ABC training meshes, at ShapeNet's pool sizes
cod-vae-dataset data/merged \
    --hf abc=TimSchneider42/tactile-mnist-abc-dataset-small:0.24435897 --hf-split train \
    --num-vol 500000 --num-surface 250000

# all 11,480 MNIST3D training meshes, at a fifth of that
cod-vae-dataset data/merged \
    --hf mnist3d=TimSchneider42/tactile-mnist-mnist3d --hf-split train \
    --num-vol 50000 --num-surface 25000
```

Only training splits are used; the test splits of both Hugging Face datasets are built into a separate root the same way (`--hf-split test`) and used exclusively for evaluation.
Add `--workers N` to parallelize and `--shard INDEX/COUNT` to spread the build over several machines; neither changes the result, since each object's seed comes from its row index rather than from the processing order.

**2. Stage 1, once per `num_latents`.**
`--repeat 8` rather than the reference's 16, because the merged dataset is three times the size of the ShapeNet set that default was chosen for; an epoch is 880,616 samples either way:

```bash
for m in 4 8 16 32 64; do
    torchrun --nproc_per_node=4 examples/train_shapenet.py data/merged runs/m$m/stage1 \
        --stage 1 --num-latents $m --repeat 8 --num-workers 10 --tf32 --resume
done
```

**3. Stage 2, four latent widths per autoencoder.**
`latent_dim` shapes only the latent VAE's projections, so all four widths reuse the same stage-1 checkpoint:

```bash
for m in 4 8 16 32 64; do
    for d in 4 8 16 32; do
        torchrun --nproc_per_node=4 examples/train_shapenet.py data/merged runs/m$m/stage2_d$d \
            --stage 2 --init-from runs/m$m/stage1/checkpoint_last.npz \
            --latent-dim $d --repeat 8 --num-workers 10 --tf32 --resume
    done
done
```

Everything else is the reference recipe: 100 epochs per stage, effective batch 256 (stage 1) and 512 (stage 2), learning rate 1e-4 scaled by the effective batch and halved at epochs 60/70/80/90 in stage 2, gradient clipping 0.5, seed 123456.
On four H100s a stage-1 run takes about 2.2 days and a stage-2 run about 18 hours.

**4. Publish.**

```python
from cod_vae import CODVAE

vae = CODVAE.load("runs/m32/stage2_d32/checkpoint_last.npz")
vae.push_to_hub("TimSchneider42/cod-vae-32x32")
```

## Known differences from the reference training

- **Precision**: the reference trains with 16-mixed precision; these trainers run in full float32. Expect roughly twice the per-step cost; on H100-class hardware a full stage-1 run remains a matter of days.
- **Stage-2 determinism**: the frozen autoencoder runs deterministically here, while the reference (a side effect of Lightning's train mode) keeps stochastic depth active inside it.
- **Non-bit-identical runs**: weight initialization mirrors torch's default schemes but uses a different RNG, and data ordering/DropPath draws differ from the Lightning pipeline. You will reproduce the recipe and expected quality, not the exact released checkpoint.
- **Query subsampling**: the reference subsamples query pools with a chunked two-stage scheme for HDF5 IO efficiency; this implementation samples uniformly without replacement, which is what that scheme approximates.
