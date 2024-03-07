from pathlib import Path
from typing import Any, Literal

from click import Choice, group, Context, Parameter, echo, option, Path as PathType, argument
from dotenv import load_dotenv, find_dotenv
from dspy import OpenAI as DSPyOpenAI, settings as dspy_settings

from mibi import __version__ as app_version
from mibi.model import QuestionData
from mibi.modules import AnswerModule, DocumentsModule, ExactAnswerModule, IdealAnswerModule, SnippetsModule
from mibi.modules.incremental import IncrementalAnswerModule
from mibi.modules.mock import MockDocumentsModule, MockExactAnswerModule, MockIdealAnswerModule, MockSnippetsModule


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


@cli.command()
@argument(
    "input_path",
    # help="The BioASQ questions file.",
    type=PathType(
        path_type=Path,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        writable=False,
        resolve_path=True,
        allow_dash=False
    ),
)
@argument(
    "output_path",
    type=PathType(
        path_type=Path,
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=False,
        writable=True,
        resolve_path=True,
        allow_dash=False
    ),
)
@option(
    "-d", "--documents-module", "documents_module_type",
    type=Choice([
        "mock"
    ]),
    default="mock",
)
@option(
    "-s", "--snippets-module", "snippets_module_type",
    type=Choice([
        "mock"
    ]),
    default="mock",
)
@option(
    "-e", "--exact-answer-module", "exact_answer_module_type",
    type=Choice([
        "mock"
    ]),
    default="mock",
)
@option(
    "-i", "--ideal-answer-module", "ideal_answer_module_type",
    type=Choice([
        "mock"
    ]),
    default="mock",
)
@option(
    "-a", "--answer-module", "answer_module_type",
    type=Choice([
        "incremental"
    ]),
    default="incremental",
)
def run(
    input_path: Path,
    output_path: Path,
    documents_module_type: Literal[
        "mock"
    ],
    snippets_module_type: Literal[
        "mock"
    ],
    exact_answer_module_type: Literal[
        "mock"
    ],
    ideal_answer_module_type: Literal[
        "mock"
    ],
    answer_module_type: Literal[
        "incremental"
    ],
) -> None:
    with input_path.open("r") as input_file:
        data = QuestionData.model_validate_json(input_file.read())
    echo(f"Found {len(data.questions)} questions.")

    turbo = DSPyOpenAI(model="gpt-3.5-turbo")
    dspy_settings.configure(lm=turbo)

    documents_module: DocumentsModule
    if documents_module_type == "mock":
        documents_module = MockDocumentsModule()
    else:
        raise ValueError("Unknown documents module type.")
    snippets_module: SnippetsModule
    if snippets_module_type == "mock":
        snippets_module = MockSnippetsModule()
    else:
        raise ValueError("Unknown documents module type.")
    exact_answer_module: ExactAnswerModule
    if exact_answer_module_type == "mock":
        exact_answer_module = MockExactAnswerModule()
    else:
        raise ValueError("Unknown documents module type.")
    ideal_answer_module: IdealAnswerModule
    if ideal_answer_module_type == "mock":
        ideal_answer_module = MockIdealAnswerModule()
    else:
        raise ValueError("Unknown documents module type.")
    answer_module: AnswerModule
    if answer_module_type == "incremental":
        answer_module = IncrementalAnswerModule(
            documents_module=documents_module,
            snippets_module=snippets_module,
            exact_answer_module=exact_answer_module,
            ideal_answer_module=ideal_answer_module,
        )
    else:
        raise ValueError("Unknown documents module type.")

    print(answer_module.forward(data.questions[0]))
