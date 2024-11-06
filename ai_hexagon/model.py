import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax


class Model(nn.Module):
    def init_train_state(
        self, x: jnp.ndarray, y: jnp.ndarray, key: jax.random.PRNGKey
    ) -> TrainState:
        variables = self.init(key, x, y)
        state = TrainState.create(
            apply_fn=self.apply,
            params=variables["params"],
            tx=optax.adamw(3e-4),
        )
        return state

    def train_step(self, x: jnp.ndarray, y: jnp.ndarray, state: TrainState):
        def loss_fn(params: flax.core.FrozenDict):
            y_pred = state.apply_fn({"params": params}, x)
            loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, y)
            return loss

        grads = jax.grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state
