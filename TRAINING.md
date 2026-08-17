# Training COD-VAE

This guide covers everything about training with this package: how the training data is produced, the two stages and their hyperparameters, multi-GPU launching, the Python API, and the exact commands behind the published `cod-vae-NxM` models.
The recipe follows the reference implementation, so it also replicates the training of the released checkpoints (vae_m32 / vae_m64) as closely as this package allows; see the last section for the known differences.

Training follows the two-stage recipe of the paper:

1. **Stage 1** trains the autoencoder (point cloud encoder, triplane decoder with uncertainty-based token pruning, occupancy head) with occupancy reconstruction losses on both the refined and the initial prediction, plus a supervision loss for the uncertainty head.
2. **Stage 2** freezes the autoencoder and trains the latent VAE modules (`latent_proj_in`/`latent_proj_out`/`latent_decoder`) with a feature matching loss, the reconstruction loss through the frozen decoder, and a KL term.

Two command line entry points appear below:

- `cod-vae-train` is the general trainer: it takes either a directory of meshes or a preprocessed dataset root and covers the common flags.
- `examples/train_shapenet.py` is the reference-recipe example for preprocessed roots; it additionally exposes `--accumulate`, `--latent-dim`, `--resume` and `--tf32`, and defaults to the reference's hyperparameters (seed 123456, per-stage batch sizes). All commands in this guide that replicate the paper use it.

Both are thin wrappers around the same Python API (section 6).

## 1. Training data

Both stages consume the same kind of training data: per shape, pools of surface points, uniform volume queries, and near-surface queries with ground-truth **occupancy labels**, from which random subsamples are drawn each step with the reference's anisotropic scaling augmentation.
Per training sample, 2048 surface points and 4096 volume + 4096 near-surface queries are drawn.

These pools are always produced by one and the same preprocessing — the recipe the original authors used to build their ShapeNet training data ([sdf_gen](https://github.com/1zb/sdf_gen)): watertighting via [point_cloud_utils](https://github.com/fwilliams/point-cloud-utils) where a mesh needs it (so your meshes do **not** need to be watertight), normalization into the [-1, 1] cube, and sampling of the query pools with occupancy labels.
It requires the `preprocess` extra (`pip install cod-vae[preprocess]`) and can run in two ways: on the fly during training, or ahead of time into a dataset on disk.

### Option 1: training directly on meshes

Point `cod-vae-train` at a directory of mesh files; the occupancy pools are computed lazily during the first epoch.

```bash
cod-vae-train path/to/meshes checkpoints/stage1 --stage 1 --backend torch
cod-vae-train path/to/meshes checkpoints/stage2 --stage 2 --init-from checkpoints/stage1/checkpoint_last.npz
```

Pass `--cache-dir` to reuse the computed pools across runs.
Checkpoints are self-contained npz files loadable by both backends (and by `CODVAE.load`).

### Option 2: building a dataset with cod-vae-dataset

`cod-vae-dataset` preprocesses ahead of time — preprocess once, train many times — and builds a dataset on disk by merging any number of sources:

- `--meshes [NAME=]DIR`: a directory of mesh files. Meshes in `train`/`val`/`test` subdirectories are assigned to the corresponding splits; otherwise everything becomes training data.
- `--hf [NAME=]DATASET`: a Hugging Face mesh dataset in the [Tactile MNIST](https://github.com/TimSchneider42/tactile-mnist) format (rows with `mesh.vertices`/`mesh.faces` columns), given as a Hub repository id or a local path. By default the `train`/`val`(`idation`)/`test` splits are used, as far as present; `--hf-split SRC[=DST]` selects and remaps splits explicitly (e.g. `--hf-split holdout=val`).
- `--vecset PATH`: an existing preprocessed root as distributed by the 3DShape2VecSet authors (section 2), merged as-is (symlinked by default; `--link hardlink|copy` to materialize).

Every source accepts an optional `:FRACTION` suffix (e.g. `--hf TimSchneider42/tactile-mnist-mnist3d:0.1`) to keep only a deterministic random subsample of each split; the selection is controlled by `--seed` (default 0), so the same command always yields the same subset.
Preprocessing is resumable (existing outputs are skipped unless `--overwrite` is given) and parallelizes with `--workers`, each of which gets its own slice of the available cores (the watertighting and winding-number code parallelizes internally over all cores, which otherwise oversubscribes the machine badly).
Large builds can additionally be spread over several machines with `--shard INDEX/COUNT`: every shard preprocesses its share of the meshes, and a final run without `--shard` links the `--vecset` sources and writes the `.lst` files.
Watertighting is applied only to meshes that need it — one that already bounds a volume is left as it is, since the repair would only resample its surface onto an octree (and on some inputs never terminates); `--watertight-closed-meshes` runs it unconditionally, as the reference script does.
A mesh that fails preprocessing aborts the build; pass `--skip-failed` to instead drop failing meshes with a warning (the behavior of the original sdf_gen script), and `--timeout SECONDS` to treat one that never returns as a failure too.

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

## 2. The original ShapeNet data

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
`cod-vae-dataset` writes exactly this layout too, so a merged root and the authors' original root are interchangeable everywhere below.

## 3. Stage 1 — autoencoder

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

## 4. Stage 2 — latent VAE

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

## 5. Multi-GPU training

- **PyTorch**: standard DistributedDataParallel; launch with `torchrun`, e.g. `torchrun --nproc_per_node=4 -m cod_vae.cli train path/to/data checkpoints --stage 1`, or `torchrun --nproc_per_node=4 examples/train_shapenet.py ...` as above.
- **JAX**: single-process data parallelism across all visible devices; just run the training script and it will shard batches over all GPUs automatically.

In both cases `batch_size` is per device and the learning rate is scaled by the effective batch size, following the reference implementation.

## 6. Training from Python

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

## 7. Using and publishing the result

```python
from cod_vae import CODVAE

vae = CODVAE.load("checkpoints/stage2/checkpoint_last.npz")
vae.push_to_hub("you/cod-vae-m32")  # weights you trained yourself are yours to host
```

A quick qualitative check: encode and decode a few validation shapes (`ShapeNetVecSetDataset(root_dir, split="val")` provides surface points and labeled queries, so occupancy accuracy/IoU can be computed by comparing `vae.decode(latents, queries) > 0` against the labels).

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

## How the published cod-vae-16xM-small models were trained

The `TimSchneider42/cod-vae-16x<latent_dim>-small` models are decode-optimized variants: ~39M parameters instead of 188M, with a ~20M decode path instead of 90M, selected in an ablation campaign whose target was decode-path throughput *including the backward pass* (for pipelines that train through the frozen decoder) under a hard quality floor of 0.83 held-out ABC IoU for the 16x8 configuration.
The winning architecture halves the width, trims the encoder, quarters the refinement-decoder tokens with 16-px triplane patches, halves the query-plane channels, and doubles the latent decoder back to 12 layers to buy quality where it is cheap:

```bash
SMALL_ARCH="--arch embed_dim=256 --arch num_heads=4 --arch encoder_num_blocks=3 \
    --arch decoder_num_layers=6 --arch decoder_output_patch_size=16 --arch query_dim=16"
```

Same merged dataset as above. Stage 1 runs **200 epochs** rather than the reference 100 — measured worth +0.009 trunk IoU (~+0.003 after stage 2) — and the small models are dataloader-bound at 4 GPUs, so 2 GPUs at twice the per-GPU batch give the same wall-clock at half the allocation:

```bash
# Stage 1, once for the 16-latent trunk (2 GPUs, effective batch 256, 200 epochs)
torchrun --nproc_per_node=2 examples/train_shapenet.py data/merged runs/small-m16/stage1 \
    --stage 1 --num-latents 16 --epochs 200 --batch-size 128 \
    --repeat 8 --num-workers 10 --tf32 --resume $SMALL_ARCH

# Stage 2, one run per latent width (2 GPUs, effective batch 512, 100 epochs)
for d in 4 8 16; do
    torchrun --nproc_per_node=2 examples/train_shapenet.py data/merged runs/small-m16/stage2_d$d \
        --stage 2 --init-from runs/small-m16/stage1/checkpoint_last.npz \
        --latent-dim $d --epochs 100 --batch-size 256 \
        --repeat 8 --num-workers 10 --tf32 --resume --arch num_latent_layers=12
done
```

Note the two `--arch` sets: the stage-1 flags define the autoencoder and are inherited by stage 2 from the checkpoint; `num_latent_layers=12` belongs to stage 2 (the latent decoder only trains there).
Everything else is the reference recipe (learning rate, clipping, seed, LR milestones — stage 2's absolute 60/70/80/90 schedule is unaffected by the longer stage 1).
Published checkpoints additionally pin `attention_implementation="default"` in their config: on this architecture's short decode sequences the XLA attention path is ~1.3x faster than the cuDNN kernel that "auto" selects at half precision.

The choices behind each knob were measured one ablation at a time (width 512/384/256/128, patches 8/16/32, decoder depth 4/6/8, latent decoder 4/6/12, encoder blocks, keep ratio 0.5–0.10, query_dim 32/16), each candidate judged by its own stage-2 IoU — stage-1 gaps repeatedly failed to predict stage-2 verdicts across architecture changes.

## Known differences from the reference training

- **Precision**: the reference trains with 16-mixed precision; these trainers run in full float32. Expect roughly twice the per-step cost.
- **Stage-2 determinism**: the frozen autoencoder runs deterministically here, while the reference (a side effect of Lightning's train mode) keeps stochastic depth active inside it.
- **Non-bit-identical runs**: weight initialization mirrors torch's default schemes but uses a different RNG, and data ordering/DropPath draws differ from the Lightning pipeline. You will reproduce the recipe and expected quality, not the exact released checkpoint.
- **Query subsampling**: the reference subsamples query pools with a chunked two-stage scheme for HDF5 IO efficiency; this implementation samples uniformly without replacement, which is what that scheme approximates.
