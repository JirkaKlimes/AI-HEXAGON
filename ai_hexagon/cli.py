import json
from pathlib import Path
from typing import get_args
import click
from tabulate import tabulate
import importlib.util
import inspect

from ai_hexagon.test_suite import TestSuite
from ai_hexagon.tests import Test
from ai_hexagon.model import Model


@click.group()
def cli():
    pass


@cli.group()
def tests():
    pass


@tests.command()
@click.argument("test_name")
def show(test_name: str):
    tests = {t.get_test_name(): t for t in get_args(Test)}
    test = tests[test_name]
    print(f"Title: {test.__title__}")
    print(f"Description: {test.__description__}")
    print()
    print("Schema:")
    print(json.dumps(test.model_json_schema(), indent=4))


@tests.command()
def list():
    headers = ["Test Name", "Test Title", "Description"]
    table_data = []

    for test in get_args(Test):
        name = test.get_test_name()
        table_data.append([name, test.__title__, test.__description__])

    print(tabulate(table_data, headers=headers, tablefmt="simple_grid"))


@cli.group()
def suite():
    pass


@suite.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--suite_path", type=click.Path(exists=True), default=Path("./results/suite.json")
)
@click.option("--save", is_flag=True)
def run(model_path: Path, suite_path: Path, save: bool):
    suite = TestSuite(**json.load(open(suite_path)))
    print(suite.model_dump_json(indent=4))

    spec = importlib.util.spec_from_file_location("model_module", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {model_path}")

    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    classes = [
        cls
        for name, cls in inspect.getmembers(model_module, inspect.isclass)
        if issubclass(cls, Model) and cls is not Model
    ]

    if not classes:
        print(f"No subclass of Model found in {model_path}")
        return

    if len(classes) > 1:
        print(
            f"Multiple subclasses of Model found in {model_path}: {[cls.__name__ for cls in classes]}"
        )
        return

    model_class = classes[0]
    result = suite.evaluate(model_class)
    json_str = json.dumps(result.model_dump(), indent=4, ensure_ascii=False)
    print(json_str)
    if save:
        with open(Path(model_path).with_suffix(".result.json"), "w") as f:
            f.write(json_str)


if __name__ == "__main__":
    cli()
