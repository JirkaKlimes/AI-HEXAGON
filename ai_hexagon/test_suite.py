import functools
import operator
from typing import Any, Dict, List, Optional, Type, cast
import jax
import jax.numpy as jnp
from pydantic import BaseModel

from ai_hexagon.model import Model
from ai_hexagon.tests import Test


class WeightedTest(BaseModel):
    weight: float
    test: Test


class Metric(BaseModel):
    name: str
    description: str
    tests: List[WeightedTest]


class ModelStats(BaseModel):
    size: int
    size_doubling_rate: float
    size_big_o: str
    flops: int
    flops_doubling_rate: float
    flops_big_o: str


class VariationResults(BaseModel):
    arguments: Dict[str, Any]
    metrics: Dict[str, float]
    model_stats: ModelStats

    model_config = {"protected_namespaces": ()}


class Results(BaseModel):
    name: str
    title: str
    description: str
    authors: Optional[List[str]]
    paper: Optional[str]
    variations: Dict[str, VariationResults]


def _compute_doubling_rate(x: Dict):
    n = jnp.array(list(x.keys()))
    fn = jnp.array(list(x.values()))
    slope = jnp.polyfit(jnp.log2(n), jnp.log2(fn), 1)[0]
    if abs(slope) < 1e-3:
        return 0.0
    return float(slope.round(2))


def _fit_big_o(x: Dict) -> str:
    n = jnp.array(list(x.keys()))  # noqa F841
    fn = jnp.array(list(x.values()))  # noqa F841
    # TODO
    return "???"


class TestSuite(BaseModel):
    name: str
    description: str
    vocab_size: int
    sequence_length: int
    sequence_lengths: List[int]
    metrics: List[Metric]

    def compute_stats(
        self,
        model_class: Type[Model],
        vocab_size: int,
        sequence_length: int,
        sequence_lengths: List[int],
    ) -> ModelStats:
        lenghts = set(sequence_lengths) | {sequence_length}
        sizes: Dict[int, int] = {}
        flops: Dict[float, float] = {}
        key = jax.random.PRNGKey(0)
        x = jnp.zeros((1, sequence_length), dtype=jnp.uint32)
        print(model_class(vocab_size=vocab_size).tabulate(key, x, depth=1))
        for length in lenghts:
            model = model_class(vocab_size=vocab_size)
            x = jnp.zeros((1, length), dtype=jnp.uint32)
            variables = model.init(key, x)
            params = variables["params"]
            sizes[length] = jax.tree.reduce(
                operator.add, jax.tree.map(lambda x: x.nbytes, params)
            )
            compiled = jax.jit(model.apply).lower(variables, x).compile()
            flops[length] = float(compiled.cost_analysis()[0]["flops"])

        return ModelStats(
            size=sizes[sequence_length],
            size_doubling_rate=_compute_doubling_rate(sizes),
            size_big_o=_fit_big_o(sizes),
            flops=flops[sequence_length],
            flops_doubling_rate=_compute_doubling_rate(flops),
            flops_big_o=_fit_big_o(sizes),
        )

    def evaluate(self, model_class: Type[Model]) -> Results:
        weighted_tests = sum([m.tests for m in self.metrics], [])
        tests = {wt.test for wt in weighted_tests}
        variation_results = {}
        for v in model_class.get_variations():
            print(model_class.get_variations()[v])
            cls = cast(
                Type[Model],
                functools.partial(model_class, **model_class.get_variations()[v]),
            )
            model_stats = self.compute_stats(
                cls, self.vocab_size, self.sequence_length, self.sequence_lengths
            )
            test_results = {}
            for t in tests:
                test_results[t] = t.evalulate(model_class)
            metrics = {}
            for m in self.metrics:
                metrics[m.name] = sum(
                    [wt.weight * test_results[wt.test] for wt in m.tests]
                )
            variation_results[v] = VariationResults(
                arguments=model_class.get_variations()[v],
                metrics=metrics,
                model_stats=model_stats,
            )
        return Results(
            name=model_class.__name__,
            title=model_class.get_model_title(),
            description=model_class.__doc__,
            authors=model_class.__authors__,
            paper=model_class.__paper__,
            variations=variation_results,
        )
