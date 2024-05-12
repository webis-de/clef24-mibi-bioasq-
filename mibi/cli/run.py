from pathlib import Path
from typing import Literal

from click import Choice, IntRange, echo, option, Path as PathType, argument, command


@command()
@argument(
    "input_path",
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
        "mock",
        "pyterrier",
    ]),
    default="pyterrier",
)
@option(
    "-s", "--snippets-module", "snippets_module_type",
    type=Choice([
        "mock",
        "pyterrier",
    ]),
    default="pyterrier",
)
@option(
    "-e", "--exact-answer-module", "exact_answer_module_type",
    type=Choice([
        "llm",
        "mock",
    ]),
    default="llm",
)
@option(
    "-i", "--ideal-answer-module", "ideal_answer_module_type",
    type=Choice([
        "llm",
        "mock",
    ]),
    default="llm",
)
@option(
    "-a", "--answer-module", "answer_module_type",
    type=Choice([
        "retrieve-then-generate", "rtg",
        "generate-then-retrieve", "gtr",
        "retrieve-then-generate-then-retrieve", "rtgtr",
        "generate-retrieve-then-generate", "gtrtg",
        "incremental",
        "independent",
    ]),
    default="retrieve-then-generate",
)
@option(
    "-l", "--llm", "--language-model-name", "language_model_name",
    type=Choice([
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "text-davinci-003",
        "Mixtral-8x7B-Instruct-v0.1",
        "Mistral-7B-Instruct-v0.2",
    ]),
    default="gpt-3.5-turbo",
)
@option(
    "-n", "--first", "--first-questions", "first_questions",
    type=IntRange(min=0),
)
@option(
    "--elasticsearch-url",
    type=str,
    envvar="ELASTICSEARCH_URL",
)
@option(
    "--elasticsearch-username",
    type=str,
    envvar="ELASTICSEARCH_USERNAME",
)
@option(
    "--elasticsearch-password",
    type=str,
    envvar="ELASTICSEARCH_PASSWORD",
)
@option(
    "--elasticsearch-index",
    type=str,
    envvar="ELASTICSEARCH_INDEX_PUBMED",
)
@option(
    "-m", "--model-path", "model_path",
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
def run(
    input_path: Path,
    output_path: Path,
    documents_module_type: Literal[
        "mock",
        "pyterrier",
    ],
    snippets_module_type: Literal[
        "mock",
        "pyterrier",
    ],
    exact_answer_module_type: Literal[
        "llm",
        "mock",
    ],
    ideal_answer_module_type: Literal[
        "llm",
        "mock",
    ],
    answer_module_type: Literal[
        "retrieve-then-generate", "rtg",
        "generate-then-retrieve", "gtr",
        "retrieve-then-generate-then-retrieve", "rtgtr",
        "generate-retrieve-then-generate", "gtrtg",
        "incremental",
        "independent",
    ],
    language_model_name: str,
    first_questions: int | None,
    elasticsearch_url: str | None,
    elasticsearch_username: str | None,
    elasticsearch_password: str | None,
    elasticsearch_index: str | None,
    model_path: Path | None
) -> None:
    from typing import Iterable
    from mibi.model import AnsweredQuestion, AnsweredQuestionData, PartiallyAnsweredQuestionData, PartiallyAnsweredQuestion, Answer
    from mibi.modules import JsonAnswerModule
    from mibi.modules.build import build_answer_module

    with input_path.open("rb") as input_file:
        data = PartiallyAnsweredQuestionData.model_validate_json(
            input_file.read())
    echo(f"Found {len(data.questions)} questions.")

    answer_module = build_answer_module(
        documents_module_type=documents_module_type,
        snippets_module_type=snippets_module_type,
        exact_answer_module_type=exact_answer_module_type,
        ideal_answer_module_type=ideal_answer_module_type,
        answer_module_type=answer_module_type,
        language_model_name=language_model_name,
        elasticsearch_url=elasticsearch_url,
        elasticsearch_username=elasticsearch_username,
        elasticsearch_password=elasticsearch_password,
        elasticsearch_index=elasticsearch_index,
    )

    questions = data.questions
    if first_questions is not None:
        questions = questions[:first_questions]

    question_answer_pairs: Iterable[tuple[PartiallyAnsweredQuestion, Answer]]

    if model_path is not None:
        json_answer_module = JsonAnswerModule(answer_module)
        print(f"Loading LLM programm parameters from: {model_path}")
        json_answer_module.load(model_path)
        question_answer_pairs = (
            (question, json_answer_module.forward(
                question.model_dump(mode="json")
            ))
            for question in questions
        )
    else:
        question_answer_pairs = (
            (question, answer_module.forward(question))
            for question in questions
        )

    answered_questions = [
        AnsweredQuestion(
            id=question.id,
            type=question.type,
            body=question.body,
            documents=answer.documents,
            snippets=answer.snippets,
            ideal_answer=answer.ideal_answer,
            exact_answer=answer.exact_answer,
        )
        for question, answer in question_answer_pairs
    ]

    answered_data = AnsweredQuestionData(
        questions=answered_questions,
    )

    with output_path.open("wt") as output_file:
        output_file.write(answered_data.model_dump_json(
            indent=2,
            by_alias=True,
        ))
    echo(f"Answered {len(answered_data.questions)} questions.")
