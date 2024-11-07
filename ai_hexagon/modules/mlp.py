import flax.linen as nn


class MLP(nn.Module):
    dims: int
    layers: int = 1

    @nn.compact
    def __call__(self, x):
        in_dims = x.shape[-1]
        for _ in range(self.layers):
            x = nn.Dense(self.dims)(x)
            x = nn.silu(x)
        x = nn.Dense(in_dims)(x)
        return x
