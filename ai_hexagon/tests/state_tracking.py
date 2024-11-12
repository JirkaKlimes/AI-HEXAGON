from typing import Type, Literal
from flax.typing import Array
import jax.numpy as jnp
import jax
import numpy as np
from einops import rearrange, repeat
from queue import Queue
from threading import Thread, Event
from flax.training.train_state import TrainState
import optax
from tqdm import tqdm

from ai_hexagon.model import Model
from ai_hexagon.test import BaseTest


class KeyGenerator:
    def __init__(self, key):
        self.key = key

    def __call__(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey


class StateTracking(BaseTest):
    __title__ = "State Tracking"
    __description__ = "Tests model ability to manipulate and track state"

    name: Literal["state_tracking"] = "state_tracking"

    num_steps: int = 256
    state_size: int = 10

    batch_size: int = 64
    num_train_steps: int = 10000
    steps_group_size: int = 100
    min_divergence: float = 0.001

    def evalulate(self, model_class: Type[Model]) -> float:
        assert not (
            self.num_train_steps % self.steps_group_size
        ), f"num_train_steps ({self.num_train_steps}) must be divisible by steps_group_size ({self.steps_group_size})."

        keygen = KeyGenerator(jax.device_put(self.key, jax.devices("cpu")[0]))

        def get_batch():
            swaps = jax.random.randint(
                keygen(),
                (self.batch_size, self.num_steps, 2),
                minval=0,
                maxval=self.state_size - 1,
            )
            swaps = np.array(swaps)

            x = rearrange(swaps, "... i j -> ... (i j)")

            state = np.arange(self.state_size, dtype=np.uint8)
            state = repeat(state, "i -> j i", j=self.batch_size)

            y = np.empty((self.batch_size, self.num_steps), dtype=np.uint8)
            for idx, sw in enumerate(rearrange(swaps, "i j ... -> j i ...")):
                of, to = sw[..., 0], sw[..., 1]
                state[np.arange(self.batch_size), to] = state[
                    np.arange(self.batch_size), of
                ]
                y[np.arange(self.batch_size), idx] = state[
                    np.arange(self.batch_size), to
                ]

            return x, y

        def worker():
            while True:
                if queue.qsize() >= 2:
                    print("ready")
                    batch_group_take.clear()
                    batch_group_take.wait()
                    continue
                print("do......")
                batch_group_x = [None] * self.steps_group_size
                batch_group_y = [None] * self.steps_group_size
                for idx in range(self.steps_group_size):
                    x, y = get_batch()
                    batch_group_x[idx] = x
                    batch_group_y[idx] = y
                batch_group_x = np.stack(batch_group_x)
                batch_group_y = np.stack(batch_group_y)

                queue.put((batch_group_x, batch_group_y))
                print("do")

        def train(state: TrainState, batch_group: Array):
            def loss_fn(y_pred, y):
                y_pred = y_pred[:, 1::2]
                loss = optax.softmax_cross_entropy_with_integer_labels(y_pred, y).mean()
                return loss

            def scan_fn(state, batch):
                x, y = batch
                state = model.train_step(x, y, state, loss_fn)
                return state, None

            state, _ = jax.lax.scan(scan_fn, state, batch_group)
            return state

        queue: Queue = Queue()
        batch_group_take = Event()

        train = jax.jit(train)

        Thread(target=worker, daemon=True).start()

        model = model_class(self.state_size)
        state = model.init_train_state(get_batch()[0], self.key)

        for _ in tqdm(range(self.num_train_steps // self.steps_group_size)):
            batch_group = queue.get()
            batch_group_take.set()
            state = train(state, batch_group)

        last_accuracy_sum = 0.0
        accuracy_sum = 0.0
        divergence = 1.0
        ref_idx = 0
        while divergence > self.min_divergence:
            ref_idx += 1
            x, y = get_batch()
            y_pred = model.apply_seq(state.params, state, x)
            y_pred = y_pred[:, 1::2]
            y_pred = jnp.argmax(y_pred, axis=-1)
            sim = y_pred == y
            sim = jnp.pad(sim, ((0, 0), (0, 1)))
            acc = jnp.argmin(sim, axis=-1)
            accuracy_sum += float(jnp.sum(acc))

            divergence = (
                divergence * 0.9
                + abs((accuracy_sum - last_accuracy_sum))
                / (self.num_steps * self.batch_size * ref_idx)
                * 0.1
            )
            last_accuracy_sum = accuracy_sum

        accuracy = accuracy_sum / (self.num_steps * self.batch_size * ref_idx)
        return accuracy
