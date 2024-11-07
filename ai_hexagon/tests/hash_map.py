import random
from typing import Tuple, Type

from ai_hexagon.test import Test
from ai_hexagon.model import Model


class HashMap(Test):
    __test_title__ = "Hash Map"
    __test_description__ = (
        "Tests the model's capacity to memorize key-value pairs from the training data."
    )

    key_length: int = 8
    value_length: int = 64
    num_pairs: Tuple[int, int] = (32, 65536)
    vocab_size: int = 1024

    def evalulate(self, model: Type[Model]) -> float:
        # TODO: implement the testing
        return random.random()
