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
                parameters={"num_pairs": 100000, "key_value_ratio": 1 / 128},
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
                parameters={"num_steps": 128, "state_size": 8},
            )
        ],
    )
    suite = TestSuite(
        name="General 1M",
        description="General test of model architecture performance",
        metrics=[memory_capacity, state_management],
    )
    print(suite.model_dump_json(indent=4))
