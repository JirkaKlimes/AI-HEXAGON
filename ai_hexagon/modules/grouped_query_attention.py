from typing import Optional
import jax.numpy as jnp
from einops import rearrange, einsum, repeat
import flax.linen as nn
from flax.typing import Array


class GroupedQueryAttention(nn.Module):
    dims: int
    q_heads: int
    k_heads: int

    @nn.compact
    def __call__(self, q: Array, kv: Array, mask: Optional[Array] = None) -> Array:
        assert (
            self.q_heads % self.k_heads == 0
        ), "Number of q_heads must be divisible by k_heads"
        assert self.dims % self.q_heads == 0, "dims must be divisible by num of q_heads"

        in_dim = q.shape[-1]
        head_dim = self.dims // self.q_heads
        repeat_factor = self.q_heads // self.k_heads

        q = nn.Dense(features=self.q_heads * head_dim)(q)
        q = rearrange(q, "... s (h d) -> ... h s d", h=self.q_heads)

        k = nn.Dense(features=self.k_heads * head_dim)(kv)
        k = rearrange(k, "... seq (h d) -> ... h s d", h=self.k_heads)
        v = nn.Dense(features=self.k_heads * head_dim)(kv)
        v = rearrange(v, "... s (h d) -> ... h s d", h=self.k_heads)

        k = repeat(k, "... h s d -> ... (h r) s d", r=repeat_factor)
        v = repeat(v, "... h s d -> ... (h r) s d", r=repeat_factor)

        scores = einsum(q, k, "... h q d, ... h k d -> ... h q k")
        scores = scores / head_dim**0.5

        if mask is not None:
            mask = repeat(mask, "... q k -> ... h q k", h=self.q_heads)
            scores = jnp.where(mask, scores, -jnp.inf)

        attn_weights = nn.softmax(scores, axis=-1)

        out = einsum(attn_weights, v, "... h q k, ... h k d -> ... h q d")

        out = rearrange(out, "... h s d -> ... s (h d)")
        out = nn.Dense(features=in_dim)(out)

        return out
