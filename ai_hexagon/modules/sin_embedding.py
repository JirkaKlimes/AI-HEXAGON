import flax.linen as nn
import jax.numpy as jnp
from flax.typing import Array


class SinEmbedding(nn.Module):
    base_freq: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        positions = jnp.arange(x.shape[-2])
        div_term = jnp.linspace(0, 1, num=x.shape[-1])[None]
        div_term = self.base_freq**div_term

        emb = positions / div_term
        emb = jnp.concatenate(
            [jnp.sin(emb[:, ::2])[..., None], jnp.cos(emb[:, 1::2])[..., None]], axis=-1
        )
        emb = emb.reshape((*emb.shape[:-2], -1))
        return emb
