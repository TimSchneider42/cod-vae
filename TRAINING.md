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

> **Training on your own data instead:** if you do not need ShapeNet, `cod_vae.training.MeshOccupancyDataset` computes the same kind of occupancy pools directly from arbitrary watertight meshes (see README).

## 2. Stage 1 — autoencoder

Stage 1 trains the point cloud encoder, the triplane decoder with uncertainty-based token pruning, and the occupancy head.
Reference hyperparameters (the defaults of `TrainingConfig(stage=1)` / `examples/train_shapenet.py`):

| Hyperparameter | Value |
|---|---|
| Epochs | 100 (dataset repeated 16x per epoch) |
| Batch size | 32 per GPU, 2x gradient accumulation |
| Optimizer | AdamW (weight decay 0.01), constant LR |
| Learning rate | 1e-4 x effective_batch_size / 256 (scaled automatically) |
| Gradient clipping | 0.5 (global norm) |
| Losses | occupancy BCE (volume 1.0, near 0.1) on refined + initial prediction, uncertainty MSE (coeff 0.01) |
| Stochastic depth | 0.1 |

On 16 GPUs:

```bash
torchrun --nproc_per_node=16 examples/train_shapenet.py {root_dir} checkpoints/stage1 \
    --stage 1 --num-latents 32   # 64 for the vae_m64 variant
```

The learning rate is scaled by the effective batch size following the reference's rule, so the GPU count does not need to match theirs.
With 16 GPUs the effective batch is 32 x 2 x 16 = 1024; pass `--accumulate 1` if you prefer to stay at an effective batch of 512.
Checkpoints (`checkpoint_epoch_*.npz`, `checkpoint_last.npz`) are written to the output directory after every epoch and are loadable by both backends.

For the JAX backend, run the same script with `--backend jax` from a single process; it shards batches across all visible GPUs automatically.

## 3. Stage 2 — latent VAE

Stage 2 freezes the stage-1 autoencoder and trains only the latent compression modules (`latent_proj_in`, `latent_proj_out`, `latent_decoder`) with the feature matching loss (coeff 1.0), the occupancy reconstruction loss through the frozen decoder (coeff 1.0), and the KL term (effective coeff 1e-6; the reference config nominally says 1e-3 but applies it twice).

| Hyperparameter | Value |
|---|---|
| Epochs | 100 |
| Batch size | 128 per GPU, no accumulation |
| LR schedule | 1e-4 (scaled as above), halved at epochs 60/70/80/90 |

```bash
torchrun --nproc_per_node=16 examples/train_shapenet.py {root_dir} checkpoints/stage2 \
    --stage 2 --init-from checkpoints/stage1/checkpoint_last.npz
```

Since the encoder pass runs without gradients and only the small latent modules are trained, stage 2 is considerably faster per epoch than stage 1.

## 4. Using and publishing the result

```python
from cod_vae import CODVAE

vae = CODVAE.load("checkpoints/stage2/checkpoint_last.npz")
vae.push_to_hub("you/cod-vae-m32")  # weights you trained yourself are yours to host
```

A quick qualitative check: encode and decode a few validation shapes (`ShapeNetVecSetDataset(root_dir, split="val")` provides surface points and labeled queries, so occupancy accuracy/IoU can be computed by comparing `vae.decode(latents, queries) > 0` against the labels).

## Known differences from the reference training

- **Precision**: the reference trains with 16-mixed precision; these trainers run in full float32. Expect roughly twice the per-step cost; on H100-class hardware a full stage-1 run remains a matter of days.
- **Stage-2 determinism**: the frozen autoencoder runs deterministically here, while the reference (a side effect of Lightning's train mode) keeps stochastic depth active inside it.
- **Non-bit-identical runs**: weight initialization mirrors torch's default schemes but uses a different RNG, and data ordering/DropPath draws differ from the Lightning pipeline. You will reproduce the recipe and expected quality, not the exact released checkpoint.
- **Query subsampling**: the reference subsamples query pools with a chunked two-stage scheme for HDF5 IO efficiency; this implementation samples uniformly without replacement, which is what that scheme approximates.
