from jax import Array
import flax.linen as nn

from ai_hexagon.model import Model

from results.mdconv.modules.md_conv import MDConv


class MultiDomainCNN(Model):
    """Multi-Domain CNN model performs convolutions in both time and frequency domains."""

    __title__ = "Multi-Domain CNN"
    __authors__ = ["Jiří Klimeš"]

    dims: int = 48
    blocks: int = 48
    separable_conv: bool = True

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Embed(self.vocab_size, self.dims)(x)

        for _ in range(self.blocks):
            x = x + MDConv(self.separable_conv)(x)

        x = nn.Dense(self.vocab_size)(x)
        return x
