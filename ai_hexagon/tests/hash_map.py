from ai_hexagon.test import Test


class HashMap(Test):
    __test_title__ = "Hash Map"
    __test_description__ = (
        "Tests the model capacity to memorize key value pairs from the training data."
    )
    key_value_ratio: float = 1 / 128
    num_pairs: int = 1000000
