import flax.linen as nn
from flax.typing import Array

from ai_hexagon.model import Model
from ai_hexagon.modules.grouped_query_attention import GroupedQueryAttention
from ai_hexagon.modules.mlp import MLP
from ai_hexagon.modules.sin_embedding import SinEmbedding


class Transformer(Model):
    """Decoder stack from 'Attention Is All You Need'."""

    __authors__ = [
        "Ashish Vaswani",
        "Noam Shazeer",
        "Niki Parmar",
        "Jakob Uszkoreit",
        "Llion Jones",
        "Aidan N. Gomez",
        "Lukasz Kaiser",
        "Illia Polosukhin",
    ]
    __paper__ = "https://arxiv.org/abs/1706.03762"

    __variations__ = {
        "deep": {
            "dims": 48,
            "q_heads": 8,
            "kv_heads": 8,
            "blocks": 32,
            "base_freq": 10000,
        }
    }

    dims: int = 96
    q_heads: int = 8
    kv_heads: int = 8
    blocks: int = 6
    base_freq: int = 10000

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Embed(self.vocab_size, self.dims)(x)

        x += SinEmbedding(self.base_freq)(x)

        for _ in range(self.blocks):
            x = x + GroupedQueryAttention(self.dims, self.q_heads, self.kv_heads)(x, x)
            x = nn.LayerNorm()(x)

            x = x + MLP(4 * self.dims)(x)
            x = nn.LayerNorm()(x)

        x = nn.Dense(self.vocab_size)(x)
        return x
