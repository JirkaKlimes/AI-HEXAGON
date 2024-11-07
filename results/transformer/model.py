import flax.linen as nn
from flax.typing import Array

from ai_hexagon.model import Model
from ai_hexagon.modules.grouped_query_attention import GroupedQueryAttention
from ai_hexagon.modules.mlp import MLP


class Transformer(Model):
    """Decoder stack from 'Attention Is All You Need' paper."""

    dims: int = 32
    q_heads: int = 8
    kv_heads: int = 8
    blocks: int = 4
    base_freq: int = 10000

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Embed(self.vocab_size, self.dims)(x)

        # TODO: fix embedding
        # x += SinEmbedding(self.base_freq)(x)

        for _ in range(self.blocks):
            x = x + GroupedQueryAttention(self.dims, self.q_heads, self.kv_heads)(
                x, x, x
            )
            x = nn.LayerNorm()(x)

            x = x + MLP(4 * self.dims)(x)
            x = nn.LayerNorm()(x)

        x = nn.Dense(self.vocab_size)(x)
        return x
