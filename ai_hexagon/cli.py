import click
from tabulate import tabulate

from ai_hexagon import Test
from ai_hexagon.tests import *  # noqa: F403


@click.group()
def cli():
    pass


@cli.command()
def tests():
    headers = ["#", "Test Name", "Description"]
    table_data = []
    schema_data = {}

    for i, test in enumerate(Test.__tests__.values()):
        name = test.__test_name__
        description = getattr(test, "__test_description__", "")
        table_data.append([i, name, description])
        schema_data[name] = test.model_json_schema()

    print(tabulate(table_data, headers=headers, tablefmt="simple_grid"))


if __name__ == "__main__":
    cli()
