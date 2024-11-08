import jax.numpy as jnp
import flax.linen as nn


class RoPE(nn.Module):
    """https://arxiv.org/pdf/2403.13298v1"""

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        seq_len, dims = x.shape[-2:]
        assert dims & ~1, "Number of input dimensions must be even."

        thetas = 10 ** -jnp.arange(0, dims // 2, dtype=jnp.float32)
        angles = jnp.outer(jnp.arange(1, seq_len + 1), thetas)
        angles = jnp.e ** (1j * angles)

        x_complex = x[..., ::2] + 1j * x[..., 1::2]
        x_rotated = x_complex * angles

        x_out = jnp.empty_like(x)
        x_out = x_out.at[..., ::2].set(x_rotated.real)
        x_out = x_out.at[..., 1::2].set(x_rotated.imag)

        return x_out


class NDRoPE(nn.Module):
    """https://arxiv.org/pdf/2403.13298v1"""

    @nn.compact
    def __call__(self, x: jnp.ndarray, coordinates: jnp.ndarray):
        dims = x.shape[-1]
        assert dims & ~1, "Number of input dimensions must be even."

        angles = nn.Dense(dims // 2, use_bias=False)(coordinates)
        angles = jnp.e ** (1j * angles)

        x_complex = x[..., ::2] + 1j * x[..., 1::2]
        x_rotated = x_complex * angles

        x_out = jnp.empty_like(x)
        x_out = x_out.at[..., ::2].set(x_rotated.real)
        x_out = x_out.at[..., 1::2].set(x_rotated.imag)

        return x_out
