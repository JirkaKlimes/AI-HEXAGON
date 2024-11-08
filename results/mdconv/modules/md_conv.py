import jax.numpy as jnp
from flax.typing import Array
import flax.linen as nn


class MDConv(nn.Module):
    separable: bool = True

    @nn.compact
    def __call__(self, x: Array) -> Array:
        xf = jnp.fft.rfftn(x, axes=(-2, -1))
        xf = nn.LayerNorm()(xf)

        if self.separable:
            xf = nn.Conv(xf.shape[-1], (3, 3), feature_group_count=xf.shape[-1])(xf)
            xf = nn.Dense(xf.shape[-1])(xf)
        else:
            xf = nn.Conv(xf.shape[-1], (3, 3))(xf)

        xt = jnp.fft.irfftn(xf, axes=(-2, -1))

        if self.separable:
            xt = nn.Conv(xt.shape[-1], (3, 3), feature_group_count=xt.shape[-1])(xt)
            xt = nn.Dense(xt.shape[-1])(xt)
        else:
            xt = nn.Conv(xt.shape[-1], (3, 3))(xt)

        xt = nn.LayerNorm()(xt)
        xt = nn.silu(xt)
        return xt
