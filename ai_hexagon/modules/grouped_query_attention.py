from typing import Optional
import jax.numpy as jnp
from einops import rearrange, einsum, repeat
import flax.linen as nn
from flax.typing import Array


class GroupedQueryAttention(nn.Module):
    """https://arxiv.org/pdf/2305.13245"""

    dims: int
    q_heads: int
    kv_heads: int

    @nn.compact
    def __call__(self, q: Array, kv: Array, mask: Optional[Array] = None):
        assert (
            self.q_heads >= self.kv_heads
        ), "Number of Q heads must be greater or equal to number of KV heads."
        assert (
            self.q_heads % self.kv_heads == 0
        ), "Number of KV heads must be divisible by number of Q heads."
        assert (
            self.dims % self.q_heads == 0
        ), "Number of dimensions must be divisible by number of Q heads."

        in_dim = q.shape[-1]
        head_dims = self.dims // self.q_heads
        groups = self.q_heads // self.kv_heads

        q = nn.DenseGeneral((self.q_heads, head_dims), use_bias=False)(q)
        k = nn.DenseGeneral((self.kv_heads, head_dims), use_bias=False)(kv)
        v = nn.DenseGeneral((self.kv_heads, head_dims), use_bias=False)(kv)

        q = rearrange(q, "... s (h g) d -> ... g h s d", g=groups)
        k = rearrange(k, "... s h d -> ... h s d")
        v = rearrange(v, "... s h d -> ... h s d")

        scores = einsum(q, k, "... g h s d, ... h a d -> ... h s a")
        if mask is not None:
            mask = repeat(mask, "... q k -> ... h q k", h=self.q_heads)
            scores = jnp.where(mask, scores, -jnp.inf)

        scale = head_dims**0.5
        attention = nn.softmax(scores / scale)

        out = einsum(attention, v, "... h s a, ... h a d -> ... h s d")
        out = rearrange(out, "... h s d -> ... s (h d)")
        out = nn.Dense(in_dim, use_bias=False)(out)
        return out


# class GroupedQueryAttention(nn.Module):
#     dims: int
#     q_heads: int
#     kv_heads: int

#     @nn.compact
#     def __call__(self, q: Array, kv: Array, mask: Optional[Array] = None) -> Array:
#         assert (
#             self.q_heads % self.kv_heads == 0
#         ), "Number of q_heads must be divisible by k_heads"
#         assert self.dims % self.q_heads == 0, "dims must be divisible by num of q_heads"

#         in_dim = q.shape[-1]
#         head_dim = self.dims // self.q_heads
#         repeat_factor = self.q_heads // self.kv_heads

#         q = nn.Dense(features=self.q_heads * head_dim)(q)
#         q = rearrange(q, "... s (h d) -> ... h s d", h=self.q_heads)

#         k = nn.Dense(features=self.kv_heads * head_dim)(kv)
#         k = rearrange(k, "... seq (h d) -> ... h s d", h=self.kv_heads)
#         v = nn.Dense(features=self.kv_heads * head_dim)(kv)
#         v = rearrange(v, "... s (h d) -> ... h s d", h=self.kv_heads)

#         k = repeat(k, "... h s d -> ... (h r) s d", r=repeat_factor)
#         v = repeat(v, "... h s d -> ... (h r) s d", r=repeat_factor)

#         scores = einsum(q, k, "... h q d, ... h k d -> ... h q k")
#         scores = scores / head_dim**0.5

#         if mask is not None:
#             mask = repeat(mask, "... q k -> ... h q k", h=self.q_heads)
#             scores = jnp.where(mask, scores, -jnp.inf)

#         attn_weights = nn.softmax(scores, axis=-1)

#         out = einsum(attn_weights, v, "... h q k, ... h k d -> ... h q d")

#         out = rearrange(out, "... h s d -> ... s (h d)")
#         out = nn.Dense(features=in_dim)(out)

#         return out
