import operator
from typing import Any, ClassVar, Dict, List, Optional
import inflection
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.typing import Array, FrozenVariableDict
from flax.training.train_state import TrainState
import optax
from pydantic import BaseModel


class ModelStats(BaseModel):
    size: int
    size_doubling_rate: float
    size_big_o: str
    flops: int
    flops_doubling_rate: float
    flops_big_o: str


def _compute_doubling_rate(x: Dict):
    n = jnp.array(list(x.keys()))
    fn = jnp.array(list(x.values()))
    slope = jnp.polyfit(jnp.log2(n), jnp.log2(fn), 1)[0]
    if abs(slope) < 1e-3:
        return 0.0
    return float(slope.round(2))


def _fit_big_o(x: Dict) -> str:
    n = jnp.array(list(x.keys()))  # noqa F841
    fn = jnp.array(list(x.values()))  # noqa F841
    # TODO
    return "???"


class Model(nn.Module):
    vocab_size: int

    __title__: ClassVar[Optional[str]] = None
    __variation__: ClassVar[Optional[str]] = None
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
    def compute_stats(
        cls, vocab_size: int, sequence_length: int, sequence_lengths: List[int]
    ) -> ModelStats:
        lenghts = set(sequence_lengths) | {sequence_length}
        sizes: Dict[int, int] = {}
        flops: Dict[float, float] = {}
        key = jax.random.PRNGKey(0)
        x = jnp.zeros((1, sequence_length), dtype=jnp.uint32)
        print(cls(vocab_size=vocab_size).tabulate(key, x, depth=1))
        for length in lenghts:
            model = cls(vocab_size=vocab_size)
            x = jnp.zeros((1, length), dtype=jnp.uint32)
            variables = model.init(key, x)
            params = variables["params"]
            sizes[length] = jax.tree.reduce(
                operator.add, jax.tree.map(lambda x: x.nbytes, params)
            )
            compiled = jax.jit(model.apply).lower(variables, x).compile()
            flops[length] = float(compiled.cost_analysis()[0]["flops"])

        return ModelStats(
            size=sizes[sequence_length],
            size_doubling_rate=_compute_doubling_rate(sizes),
            size_big_o=_fit_big_o(sizes),
            flops=flops[sequence_length],
            flops_doubling_rate=_compute_doubling_rate(flops),
            flops_big_o=_fit_big_o(sizes),
        )

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
