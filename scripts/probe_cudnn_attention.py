"""Diagnostic: which cuDNN fused-attention patterns work on THIS jax/GPU?

Run on a GPU node. Prints one line per pattern. The pattern that matters for training is
"vmap(grad), params mapped / qkv shared": that is what a critic ensemble differentiating
through the decoder produces, and it is the one that fails on jax 0.10.2 inside
_dot_product_attention_bwd_batcher, which sizes the cotangent reshape from the query alone.
"""

import jax
import jax.numpy as jnp
import numpy as np

B, T, N, H, E = 4, 218, 8, 64, 2
DTYPE = jnp.float16

rng = np.random.default_rng(0)
q, k, v = (jnp.asarray(rng.normal(size=(B, T, N, H)) * 0.1, DTYPE) for _ in range(3))
w = jnp.asarray(rng.normal(size=(E, H, H)) * 0.1, DTYPE)
w1 = jnp.asarray(rng.normal(size=(H, H)) * 0.1, DTYPE)


def report(label, fn):
    try:
        fn()
        print(f"  OK    {label}")
    except Exception as e:
        print(
            f"  FAIL  {label}\n          {type(e).__name__}: {str(e).splitlines()[-1][:150]}"
        )


print(
    f"jax {jax.__version__} on {jax.devices()[0].platform} "
    f"({jax.devices()[0].device_kind})"
)

for impl in ("xla", "cudnn"):
    print(f"\nimplementation={impl}")

    # impl is closed over, not passed: a string argument to a jitted function would be
    # rejected as non-static and every case would "fail" for the wrong reason.
    def loss(w_i, q, k, v, _impl=impl):
        out = jax.nn.dot_product_attention(q, k, v, implementation=_impl)
        return jnp.sum((out @ w_i).astype(jnp.float32))

    report(
        "forward only",
        lambda impl=impl: jax.jit(
            lambda q, k, v: jax.nn.dot_product_attention(q, k, v, implementation=impl)
        )
        .lower(q, k, v)
        .compile(),
    )

    report(
        "grad, no vmap",
        lambda loss=loss: jax.jit(jax.grad(loss, argnums=0))
        .lower(w1, q, k, v)
        .compile(),
    )

    report(
        "vmap(grad), params mapped / qkv shared  <-- the training shape",
        lambda loss=loss: jax.jit(
            jax.vmap(jax.grad(loss, argnums=0), in_axes=(0, None, None, None))
        )
        .lower(w, q, k, v)
        .compile(),
    )

    report(
        "vmap(grad), params AND qkv mapped",
        lambda loss=loss: jax.jit(
            jax.vmap(jax.grad(loss, argnums=0), in_axes=(0, 0, 0, 0))
        )
        .lower(
            w,
            jnp.broadcast_to(q, (E,) + q.shape),
            jnp.broadcast_to(k, (E,) + k.shape),
            jnp.broadcast_to(v, (E,) + v.shape),
        )
        .compile(),
    )

print("\nAlso: what cod_vae's own auto-selection decides here")
from cod_vae.jax.model import _resolve_attention  # noqa: E402

for dt in (jnp.float16, jnp.float32):
    print(
        f"  dtype={jnp.dtype(dt).name} -> {_resolve_attention('auto', jax.devices()[0], dt)}"
    )
