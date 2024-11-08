import random
from typing import Literal, Tuple, Type

from ai_hexagon.test import BaseTest
from ai_hexagon.model import Model


class HashMap(BaseTest):
    __title__ = "Hash Map"
    __description__ = (
        "Tests the model's capacity to memorize key-value pairs from the training data."
    )
    name: Literal["hash_map"] = "hash_map"

    key_length: int = 8
    value_length: int = 64
    num_pairs_range: Tuple[int, int] = (32, 65536)
    vocab_size: int = 1024

    def evalulate(self, model_class: Type[Model]) -> float:
        # TODO: implement the testing
        return random.random()
