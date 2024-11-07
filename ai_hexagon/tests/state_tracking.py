import random
from typing import Tuple, Type

from ai_hexagon.model import Model
from ai_hexagon.test import Test


class StateTracking(Test):
    __test_title__ = "State Tracking"
    __test_description__ = "Tests model ability to manipulate and track state"

    num_steps: Tuple[int, int] = (2, 128)
    state_size: int = 16

    def evalulate(self, model: Type[Model]) -> float:
        # TODO: implement the testing
        return random.random()
