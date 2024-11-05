import json
from typing import Optional
import click
from tabulate import tabulate

from ai_hexagon import Test
from ai_hexagon.tests import *  # noqa: F403


@click.group()
def cli():
    pass


@cli.command()
@click.argument("test_name", required=False)
def tests(test_name: Optional[str]):
    if test_name:
        test = Test.__tests__[test_name]
        print(f"Title: {test.__test_title__}")
        print(f"Description: {test.__test_description__}")
        print()
        print("Schema:")
        print(json.dumps(test.model_json_schema(), indent=4))
        return

    headers = ["Test Name", "Test Title", "Description"]
    table_data = []

    for test in Test.__tests__.values():
        name = test.__test_name__
        title = test.__test_title__
        description = getattr(test, "__test_description__", "")
        table_data.append([name, title, description])

    print(tabulate(table_data, headers=headers, tablefmt="simple_grid"))


if __name__ == "__main__":
    cli()
