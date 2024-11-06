import flax.linen as nn
import jax.numpy as jnp
from flax.typing import Array


class GroupedQueryAttention(nn.Module):
    dims: int
    q_heads: int
    k_heads: int

    @nn.compact
    def __call__(self, q: Array, k: Array, v: Array, mask: Array = None) -> Array:
        assert (
            self.q_heads % self.k_heads == 0
        ), "num of kv heads must by divisible by num of q heads"
        assert self.dims % self.q_heads == 0, "dims must be devisible by num of q heads"

        in_dim = q.shape[-1]

        head_dim = self.dims // self.q_heads

        qx = nn.Dense(self.q_heads * head_dim)(q)
        qx = qx.reshape((*qx.shape[:-1], self.q_heads, head_dim))

        kx = nn.Dense(self.k_heads * head_dim)(k)
        kx = kx.reshape((*kx.shape[:-1], self.k_heads, head_dim))

        vx = nn.Dense(self.k_heads * head_dim)(v)
        vx = vx.reshape((*vx.shape[:-1], self.k_heads, head_dim))

        kx = kx.repeat(self.q_heads // self.k_heads, axis=-2)
        vx = vx.repeat(self.q_heads // self.k_heads, axis=-2)

        qx = jnp.einsum("...ijk->...jik", qx)
        kx = jnp.einsum("...ijk->...jik", kx)
        vx = jnp.einsum("...ijk->...jik", vx)
        scores = jnp.einsum("...jk,...ik->...ji", qx, kx)
        scores /= jnp.sqrt(head_dim)
        if mask is not None:
            scores -= 1 / mask + 1
        scores = nn.softmax(scores)

        out = jnp.einsum("...jk,...ki->...ji", scores, vx)
        out = jnp.einsum("...ijk->...jik", out)
        out = out.reshape((*out.shape[:-2], -1))
        out = nn.Dense(in_dim)(out)

        return out
