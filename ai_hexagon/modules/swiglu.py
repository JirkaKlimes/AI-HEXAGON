import flax.linen as nn
from flax.typing import Array

class Swiglu(nn.Module):
    dims:int
    @nn.compact
    def __call__(self, x:Array) -> Array:
        x_dims = x.shape[-1]
        x1 = nn.Dense(self.dims)(x)
        x2 = nn.Dense(self.dims)(x)
        return nn.Dense(x_dims)(nn.silu(x1) * x2)