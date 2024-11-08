from typing import Dict, List, Type
from pydantic import BaseModel

from ai_hexagon.model import Model
from ai_hexagon.tests import Test, HashMap, StateTracking


class WeightedTest(BaseModel):
    weight: float
    test: Test


class Metric(BaseModel):
    name: str
    description: str
    tests: List[WeightedTest]


class Results(BaseModel):
    metrics: Dict[str, float] = {}


class TestSuite(BaseModel):
    name: str
    description: str
    metrics: List[Metric]

    def evaluate(self, model: Type[Model]) -> Results:
        weighted_tests = sum([m.tests for m in self.metrics], [])
        tests = {wt.test for wt in weighted_tests}
        test_results = {}
        for t in tests:
            test_results[t] = t.evalulate(model)
        results = Results()
        for m in self.metrics:
            results.metrics[m.name] = sum(
                [wt.weight * test_results[wt.test] for wt in m.tests]
            )
        return results


if __name__ == "__main__":
    memory_capacity = Metric(
        name="Memory Capacity",
        description="The ability of the model to store and recall information from the training data.",
        tests=[
            WeightedTest(
                weight=1.0,
                test=HashMap(
                    key_length=8,
                    value_length=64,
                    num_pairs_range=(32, 65536),
                    vocab_size=1024,
                ),
            )
        ],
    )
    state_management = Metric(
        name="State Management",
        description="The ability to maintain and manipulate an internal hidden state across a sequence of operations.",
        tests=[
            WeightedTest(
                weight=1.0,
                test=StateTracking(num_steps=(2, 128), state_size=16),
            ),
        ],
    )
    suite = TestSuite(
        name="General 1M",
        description="General test of model architecture performance",
        metrics=[memory_capacity, state_management],
    )
    print(suite.model_dump_json(indent=4))
