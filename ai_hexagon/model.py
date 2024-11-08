from typing import Any, ClassVar, Dict, List, Optional
import inflection
import jax
import flax.linen as nn
from flax.typing import Array, FrozenVariableDict
from flax.training.train_state import TrainState
import optax


class Model(nn.Module):
    vocab_size: int

    __title__: ClassVar[Optional[str]] = None
    __variations__: ClassVar[Dict[str, Dict[str, Any]]] = {}
    __authors__: ClassVar[Optional[List[str]]] = None
    __paper__: ClassVar[Optional[str]] = None

    @classmethod
    def get_model_title(cls) -> str:
        return cls.__title__ or inflection.titleize(cls.__name__)

    @classmethod
    def get_default_arguments(cls) -> Dict[str, Any]:
        blacklist = {"parent", "name", "scope"}
        arguments = {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("_") and k not in blacklist
        }
        return arguments

    @classmethod
    def get_variations(cls) -> Dict[str, Dict[str, Any]]:
        return {"default": cls.get_default_arguments(), **cls.__variations__}

    def init_train_state(self, x: Array, key: Array) -> TrainState:
        variables = self.init(key, x)
        state = TrainState.create(
            apply_fn=self.apply,
            params=variables["params"],
            tx=optax.adamw(3e-4),
        )
        return state

    def train_step(self, x: Array, y: Array, state: TrainState):
        def loss_fn(params: FrozenVariableDict):
            y_pred = state.apply_fn({"params": params}, x)
            loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, y).mean()
            return loss

        grads = jax.grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state
