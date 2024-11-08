import random
from typing import Tuple, Type, Literal

from ai_hexagon.model import Model
from ai_hexagon.test import BaseTest


class StateTracking(BaseTest):
    __title__ = "State Tracking"
    __description__ = "Tests model ability to manipulate and track state"

    name: Literal["state_tracking"] = "state_tracking"

    num_steps_range: Tuple[int, int] = (2, 128)
    state_size: int = 16

    def evalulate(self, model_class: Type[Model]) -> float:
        # TODO: implement the testing
        return random.random()
