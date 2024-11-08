from typing import Dict, List, Optional, Type
from pydantic import BaseModel

from ai_hexagon.model import Model, ModelStats
from ai_hexagon.tests import Test, HashMap, StateTracking


class WeightedTest(BaseModel):
    weight: float
    test: Test


class Metric(BaseModel):
    name: str
    description: str
    tests: List[WeightedTest]


class Results(BaseModel):
    title: str
    description: str
    authors: Optional[List[str]]
    paper: Optional[str]
    metrics: Dict[str, float]
    model_stats: ModelStats

    model_config = {"protected_namespaces": ()}


class TestSuite(BaseModel):
    name: str
    description: str
    vocab_size: int
    sequence_length: int
    sequence_lengths: List[int]
    metrics: List[Metric]

    def evaluate(self, model: Type[Model]) -> Results:
        weighted_tests = sum([m.tests for m in self.metrics], [])
        tests = {wt.test for wt in weighted_tests}
        test_results = {}
        for t in tests:
            test_results[t] = t.evalulate(model)
        metrics = {}
        for m in self.metrics:
            metrics[m.name] = sum([wt.weight * test_results[wt.test] for wt in m.tests])
        model_stats = model.compute_stats(
            self.vocab_size, self.sequence_length, self.sequence_lengths
        )
        return Results(
            title=model.get_model_title(),
            description=model.__doc__,
            authors=model.__authors__,
            paper=model.__paper__,
            metrics=metrics,
            model_stats=model_stats,
        )


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
        vocab_size=16000,
        sequence_length=8192,
        sequence_lengths=[32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        metrics=[memory_capacity, state_management],
    )
    print(suite.model_dump_json(indent=4))
