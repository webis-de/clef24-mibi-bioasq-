from typing import Any

from click import group, Context, Parameter, echo, option
from dotenv import load_dotenv, find_dotenv

from mibi import __version__ as app_version
from mibi.cli.run import run


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
        load_dotenv()


cli.add_command(run)
