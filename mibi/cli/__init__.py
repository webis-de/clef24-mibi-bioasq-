from typing import Any

from click import group, Context, Parameter, echo, option
from dotenv import load_dotenv, find_dotenv

from mibi import __version__ as app_version
from mibi.cli.run import run
from mibi.cli.index import index
from mibi.cli.parse import parse
from mibi.cli.utils import utils
from mibi.cli.compile import compile


def echo_version(
    context: Context,
    _parameter: Parameter,
    value: Any,
) -> None:
    if not value or context.resilient_parsing:
        return
    echo(app_version)
    context.exit()


@group()
@option(
    "-V",
    "--version",
    is_flag=True,
    callback=echo_version,
    expose_value=False,
    is_eager=True,
)
def cli() -> None:
    if find_dotenv():
        print("Loading configuration from .env file.")
        load_dotenv(override=True)


cli.add_command(run)
cli.add_command(index)
cli.add_command(parse)
cli.add_command(utils)
cli.add_command(compile)
