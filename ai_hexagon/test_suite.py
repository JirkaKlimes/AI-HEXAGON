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
    metrics: List[Metric]


if __name__ == "__main__":
    test = Test(
        name="hash_map",
        weight=1.0,
        parameters={"num_pairs": 100000, "key_value_ratio": 1 / 128},
    )
    metric = Metric(
        name="Memory", description="Ability to retrieve training data", tests=[test]
    )
    suite = TestSuite(name="General 1M", metrics=[metric])
    print(suite.model_dump_json(indent=4))
