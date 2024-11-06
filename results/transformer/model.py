import flax.linen as nn
import jax.numpy as jnp
from flax.typing import Array
from ai_hexagon.modules.attention import GroupedQuaryAttention
from ai_hexagon.modules.swiglu import Swiglu
from ai_hexagon.modules.time_embeding import SinCosEmbeding

class Transformer(nn.Module):
    n_embedings:int
    dims:int
    n_qheads:int
    n_kvheads:int
    n_blocks:int
    base_freq:int = 10000

    @nn.compact
    def __call__(self, inp:Array) -> Array:
        embed = self.param("embed", nn.linear.default_embed_init, (self.n_embedings, self.dims))
        x = jnp.take(embed, inp, axis=0)

        x += SinCosEmbeding(self.base_freq)(x)

        for _ in range(self.n_blocks):
            skip = x
            x = GroupedQuaryAttention(self.dims * 4, self.n_qheads, self.n_kvheads)(x, x, x)
            x += skip
            x = nn.LayerNorm()(x)

            skip = x
            x = Swiglu(self.dims * 3 // 2)(x)
            x += skip
            x = nn.LayerNorm()(x)

        x = nn.Dense(self.n_embedings)(x)
        return x