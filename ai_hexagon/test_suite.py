from typing import Dict, List, Optional, Type
from pydantic import BaseModel

from ai_hexagon.model import Model, ModelStats
from ai_hexagon.tests import Test


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
