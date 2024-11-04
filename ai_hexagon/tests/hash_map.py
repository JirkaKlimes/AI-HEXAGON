from typing import Tuple
import flax.linen as nn
from ai_hexagon.test import Test


class HashMap(Test):
    __test_name__ = "Hash Map"
    __test_description__ = (
        "Tests the model capacity to memorize key value pairs from the training data."
    )
    key_value_ratio: float = 1 / 128
    pair_limit_range: Tuple[int, int] = (1, 1000000)

    def setup(self):
        self._key_length = int(self.sequence_length * self.key_value_ratio)
        self._max_pairs = min(self.pair_limit, self.vocab_size**self._key_length)

    @property
    def pair_limit(self):
        return (
            self.pair_limit_range[0] * (1 - self.diffuculty)
            + self.pair_limit_range[1] * self.diffuculty
        )

    def evaluate_diffuculty(self, model: nn.Module) -> bool: ...


if __name__ == "__main__":
    test = HashMap()
    print(test.__test_name__)
    print(test.__test_description__)
