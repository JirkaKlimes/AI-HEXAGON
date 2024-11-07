import json
from pathlib import Path
import click
from tabulate import tabulate
import importlib.util
import inspect

from ai_hexagon import Test
from ai_hexagon.test_suite import TestSuite
from ai_hexagon.tests import *  # noqa: F403
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
    test = Test.__tests__[test_name]
    print(f"Title: {test.__test_title__}")
    print(f"Description: {test.__test_description__}")
    print()
    print("Schema:")
    print(json.dumps(test.model_json_schema(), indent=4))


@tests.command()
def list():
    headers = ["Test Name", "Test Title", "Description"]
    table_data = []

    for test in Test.__tests__.values():
        name = test.__test_name__
        title = test.__test_title__
        description = getattr(test, "__test_description__", "")
        table_data.append([name, title, description])

    print(tabulate(table_data, headers=headers, tablefmt="simple_grid"))


@cli.group()
def suite():
    pass


@suite.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option(
    "--suite_path", type=click.Path(exists=True), default="./results/suite.json"
)
def run(model_path: Path, suite_path: Path):
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
    print(f"Model: {model_class.__name__}")
    print(f"Description: {model_class.__doc__}")


if __name__ == "__main__":
    cli()
