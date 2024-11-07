from typing import Any, Dict, List
from pydantic import BaseModel


class Test(BaseModel):
    name: str
    weight: float
    parameters: Dict[str, Any]


class Metric(BaseModel):
    name: str
    description: str
    tests: List[Test]


class TestSuite(BaseModel):
    name: str
    description: str
    metrics: List[Metric]


if __name__ == "__main__":
    memory_capacity = Metric(
        name="Memory Capacity",
        description="The ability of the model to store and recall information from the training data.",
        tests=[
            Test(
                name="hash_map",
                weight=1.0,
                parameters={
                    "key_length": 8,
                    "value_length": 64,
                    "num_pairs": (32, 65536),
                    "vocab_size": 1024,
                },
            )
        ],
    )
    state_management = Metric(
        name="State Management",
        description="The ability to maintain and manipulate an internal hidden state across a sequence of operations.",
        tests=[
            Test(
                name="state_tracking",
                weight=1.0,
                parameters={"num_steps": (2, 128), "state_size": 16},
            )
        ],
    )
    suite = TestSuite(
        name="General 1M",
        description="General test of model architecture performance",
        metrics=[memory_capacity, state_management],
    )
    print(suite.model_dump_json(indent=4))
