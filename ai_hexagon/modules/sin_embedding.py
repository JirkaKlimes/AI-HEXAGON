import flax.linen as nn
import jax.numpy as jnp
from flax.typing import Array
from einops import rearrange


class SinEmbedding(nn.Module):
    base_freq: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        positions = jnp.arange(x.shape[-2])
        div_term = jnp.linspace(0, 1, num=x.shape[-1])[None]
        div_term = self.base_freq**div_term

        emb = positions[..., None] / div_term
        sim_emb = jnp.sin(emb[:, ::2])[..., None]
        cos_emb = jnp.cos(emb[:, 1::2])[..., None]
        emb = jnp.concatenate([sim_emb, cos_emb], axis=-1)
        emb = rearrange(emb, "... i j -> ... (i j)")
        return emb
