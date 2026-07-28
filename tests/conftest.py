import numpy as np
import pytest
import trimesh

from cod_vae import CODVAEConfig, init_params

# Keep JAX from grabbing all GPU memory when both backends run in one process.
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


@pytest.fixture(scope="session")
def tiny_config() -> CODVAEConfig:
    return CODVAEConfig(
        latent_dim=8,
        num_latents=8,
        embed_dim=64,
        query_dim=8,
        num_heads=4,
        num_latent_layers=2,
        encoder_num_patches=32,
        encoder_num_blocks=1,
        encoder_num_layers_per_block=1,
        decoder_output_resolution=32,
        decoder_output_patch_size=8,
        decoder_num_layers=2,
        decoder_num_init_layers=1,
        decoder_num_merged_tokens=4,
    )


@pytest.fixture(scope="session")
def tiny_params(tiny_config):
    return init_params(tiny_config, seed=0)


@pytest.fixture(scope="session")
def tiny_deterministic_config(tiny_config) -> CODVAEConfig:
    """Tiny config without stochastic depth, for cross-backend loss comparisons."""
    import dataclasses

    return dataclasses.replace(tiny_config, droppath_rate=0.0)


@pytest.fixture(scope="session")
def point_batch() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(-0.9, 0.9, (2, 512, 3)).astype(np.float32)


@pytest.fixture(scope="session")
def meshes() -> list[trimesh.Trimesh]:
    return [
        trimesh.creation.box(extents=[1.0, 0.6, 0.4]),
        trimesh.creation.icosphere(subdivisions=2, radius=0.5),
    ]
