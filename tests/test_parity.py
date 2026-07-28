"""
Cross-backend parity tests: the torch and jax implementations must agree on inference
outputs and training losses given identical parameters and inputs.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

from cod_vae import CODVAE


@pytest.fixture(scope="module", autouse=True)
def _full_precision():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    jax.config.update("jax_default_matmul_precision", "highest")


@pytest.fixture(scope="module")
def models(tiny_config, tiny_params):
    return (
        CODVAE(tiny_config, tiny_params, backend="torch"),
        CODVAE(tiny_config, tiny_params, backend="jax"),
    )


def test_params_roundtrip(models, tiny_params):
    for model in models:
        params = model.get_params()
        assert set(params) == set(tiny_params)
        for key in params:
            np.testing.assert_array_equal(params[key], tiny_params[key])


def test_encode_parity(models, point_batch):
    latents_torch = models[0].encode(point_batch)
    latents_jax = models[1].encode(point_batch)
    assert latents_torch.shape == latents_jax.shape
    np.testing.assert_allclose(latents_torch, latents_jax, atol=1e-4)


def test_decode_parity(models, point_batch):
    latents = models[0].encode(point_batch)
    rng = np.random.default_rng(1)
    queries = rng.uniform(-1, 1, (300, 3)).astype(np.float32)
    logits_torch = models[0].decode(latents, queries)
    logits_jax = models[1].decode(latents, queries)
    np.testing.assert_allclose(logits_torch, logits_jax, atol=1e-4)


def test_decode_volume_parity(models, point_batch):
    latents = models[0].encode(point_batch)
    volume_torch = models[0].decode_volume(latents, resolution=16)
    volume_jax = models[1].decode_volume(latents, resolution=16)
    np.testing.assert_allclose(volume_torch, volume_jax, atol=1e-4)


def test_unbatched_matches_batched(models, point_batch):
    for model in models:
        batched = model.encode(point_batch)
        single = model.encode(point_batch[0])
        assert single.shape == batched.shape[1:]
        np.testing.assert_allclose(single, batched[0], atol=1e-5)


def _loss_batch(num_vol=64, num_near=64, batch_size=2, num_points=256):
    rng = np.random.default_rng(2)
    return {
        "surface": rng.uniform(-0.9, 0.9, (batch_size, num_points, 3)).astype(
            np.float32
        ),
        "queries": rng.uniform(-1, 1, (batch_size, num_vol + num_near, 3)).astype(
            np.float32
        ),
        "labels": (rng.random((batch_size, num_vol + num_near)) < 0.5).astype(
            np.float32
        ),
    }, num_vol


def test_stage1_loss_parity(tiny_deterministic_config, tiny_params):
    from cod_vae.jax.training import _stage1_loss
    from cod_vae.torch.model import CODVAEModule
    from cod_vae.torch.training import _LossModule
    from cod_vae.training import TrainingConfig

    batch, num_vol = _loss_batch()
    train_config = TrainingConfig(stage=1)

    module = CODVAEModule(tiny_deterministic_config)
    module.load_state_dict(
        {k: torch.from_numpy(v.copy()) for k, v in tiny_params.items()}
    )
    module.eval()
    loss_module = _LossModule(module, train_config, num_vol)
    with torch.no_grad():
        out_torch = loss_module(
            *(torch.from_numpy(batch[k]) for k in ("surface", "queries", "labels"))
        )

    import jax.numpy as jnp

    trainable = {k: jnp.asarray(v) for k, v in tiny_params.items()}
    loss_jax, metrics_jax = _stage1_loss(
        trainable,
        {},
        {k: jnp.asarray(v) for k, v in batch.items()},
        jax.random.key(0),
        config=tiny_deterministic_config,
        train_config=train_config,
        num_vol=num_vol,
    )
    assert float(out_torch["loss"]) == pytest.approx(float(loss_jax), abs=1e-4)
    for key in ("recon_loss", "init_loss", "uncertainty_loss"):
        assert float(out_torch[key]) == pytest.approx(float(metrics_jax[key]), abs=1e-4)


def test_stage2_loss_parity(tiny_deterministic_config, tiny_params, monkeypatch):
    from cod_vae.jax.training import _stage2_loss
    from cod_vae.torch.model import CODVAEModule
    from cod_vae.torch.training import _LossModule
    from cod_vae.training import TrainingConfig

    batch, num_vol = _loss_batch()
    train_config = TrainingConfig(stage=2)

    # Zero the posterior noise in both backends so the losses are deterministic.
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.zeros_like(x))
    monkeypatch.setattr(
        jax.random,
        "normal",
        lambda key, shape, dtype=None: jax.numpy.zeros(shape, dtype),
    )

    module = CODVAEModule(tiny_deterministic_config)
    module.load_state_dict(
        {k: torch.from_numpy(v.copy()) for k, v in tiny_params.items()}
    )
    module.eval()
    loss_module = _LossModule(module, train_config, num_vol)
    with torch.no_grad():
        out_torch = loss_module(
            *(torch.from_numpy(batch[k]) for k in ("surface", "queries", "labels"))
        )

    import jax.numpy as jnp

    params_jax = {k: jnp.asarray(v) for k, v in tiny_params.items()}
    loss_jax, metrics_jax = _stage2_loss(
        params_jax,
        {},
        {k: jnp.asarray(v) for k, v in batch.items()},
        jax.random.key(0),
        config=tiny_deterministic_config,
        train_config=train_config,
        num_vol=num_vol,
    )
    assert float(out_torch["loss"]) == pytest.approx(float(loss_jax), abs=1e-4)
    for key in ("feat_loss", "recon_loss", "kl_loss"):
        assert float(out_torch[key]) == pytest.approx(float(metrics_jax[key]), abs=1e-4)
