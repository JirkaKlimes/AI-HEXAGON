import flax.linen as nn
import jax.numpy as jnp
from flax.typing import Array, Initializer

class GroupedQuaryAttention(nn.Module):
    dim:int
    num_qheads:int
    num_kvheads:int
    use_bias:bool=True
    kernel_init:Initializer=nn.initializers.lecun_normal()
    bias_init:Initializer=nn.initializers.zeros_init()
    use_kvcache:bool=True

    @nn.compact
    def __call__(self, q:Array, k:Array, v:Array, mask:Array=None) -> Array:
        assert self.num_qheads % self.num_kvheads == 0, "num of kv heads must by divisible by num of q heads"
        assert self.dim % self.num_qheads == 0, "dims must be devisible by num of q heads"

        in_dim = q.shape[-1]

        head_dim = self.dim // self.num_qheads

        qx = nn.Dense(self.num_qheads * head_dim, use_bias=self.use_bias, kernel_init=self.kernel_init, bias_init=self.bias_init)(q)
        qx = qx.reshape((*qx.shape[:-1], self.num_qheads, head_dim))

        kx = nn.Dense(self.num_kvheads * head_dim, use_bias=self.use_bias, kernel_init=self.kernel_init, bias_init=self.bias_init)(k)
        kx = kx.reshape((*kx.shape[:-1], self.num_kvheads, head_dim))

        vx = nn.Dense(self.num_kvheads * head_dim, use_bias=self.use_bias, kernel_init=self.kernel_init, bias_init=self.bias_init)(v)
        vx = vx.reshape((*vx.shape[:-1], self.num_kvheads, head_dim))

        kx = kx.repeat(self.num_qheads // self.num_kvheads, axis=-2)
        vx = vx.repeat(self.num_qheads // self.num_kvheads, axis=-2)

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
        out = nn.Dense(in_dim, use_bias=self.use_bias, kernel_init=self.kernel_init, bias_init=self.bias_init)(out)

        return out