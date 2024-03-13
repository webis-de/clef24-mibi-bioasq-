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
    ]),
    default="mock",
)
@option(
    "-s", "--snippets-module", "snippets_module_type",
    type=Choice([
        "mock",
    ]),
    default="mock",
)
@option(
    "-e", "--exact-answer-module", "exact_answer_module_type",
    type=Choice([
        "mock",
    ]),
    default="mock",
)
@option(
    "-i", "--ideal-answer-module", "ideal_answer_module_type",
    type=Choice([
        "mock",
    ]),
    default="mock",
)
@option(
    "-a", "--answer-module", "answer_module_type",
    type=Choice([
        "standard",
        "incremental",
        "independent",
    ]),
    default="standard",
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
    default="gpt-3.5-turbo-0125",
)
@option(
    "-n", "--first", "--first-questions", "first_questions",
    type=IntRange(min=0),
)
def run(
    input_path: Path,
    output_path: Path,
    documents_module_type: Literal[
        "mock",
    ],
    snippets_module_type: Literal[
        "mock",
    ],
    exact_answer_module_type: Literal[
        "mock",
    ],
    ideal_answer_module_type: Literal[
        "mock",
    ],
    answer_module_type: Literal[
        "standard",
        "incremental",
        "independent",
    ],
    language_model_name: Literal[
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "text-davinci-003",
        "Mixtral-8x7B-Instruct-v0.1",
        "Mistral-7B-Instruct-v0.2",
    ],
    first_questions: int | None,
) -> None:
    from mibi.model import AnsweredQuestion, AnsweredQuestionData, QuestionData
    from mibi.modules import AnswerModule, DocumentsModule, ExactAnswerModule, IdealAnswerModule, SnippetsModule
    from mibi.utils.language_models import init_language_model_clients

    with input_path.open("rb") as input_file:
        data = QuestionData.model_validate_json(input_file.read())
    echo(f"Found {len(data.questions)} questions.")

    init_language_model_clients(language_model_name)

    documents_module: DocumentsModule
    if documents_module_type == "mock":
        from mibi.modules.mock import MockDocumentsModule
        documents_module = MockDocumentsModule()
    else:
        raise ValueError("Unknown documents module type.")
    snippets_module: SnippetsModule
    if snippets_module_type == "mock":
        from mibi.modules.mock import MockSnippetsModule
        snippets_module = MockSnippetsModule()
    else:
        raise ValueError("Unknown documents module type.")
    exact_answer_module: ExactAnswerModule
    if exact_answer_module_type == "mock":
        from mibi.modules.mock import MockExactAnswerModule
        exact_answer_module = MockExactAnswerModule()
    else:
        raise ValueError("Unknown documents module type.")
    ideal_answer_module: IdealAnswerModule
    if ideal_answer_module_type == "mock":
        from mibi.modules.mock import MockIdealAnswerModule
        ideal_answer_module = MockIdealAnswerModule()
    else:
        raise ValueError("Unknown documents module type.")
    answer_module: AnswerModule
    if answer_module_type == "standard":
        from mibi.modules.standard import StandardAnswerModule
        answer_module = StandardAnswerModule(
            documents_module=documents_module,
            snippets_module=snippets_module,
            exact_answer_module=exact_answer_module,
            ideal_answer_module=ideal_answer_module,
        )
    elif answer_module_type == "incremental":
        from mibi.modules.incremental import IncrementalAnswerModule
        answer_module = IncrementalAnswerModule(
            documents_module=documents_module,
            snippets_module=snippets_module,
            exact_answer_module=exact_answer_module,
            ideal_answer_module=ideal_answer_module,
        )
    elif answer_module_type == "independent":
        from mibi.modules.independent import IndependentAnswerModule
        answer_module = IndependentAnswerModule(
            documents_module=documents_module,
            snippets_module=snippets_module,
            exact_answer_module=exact_answer_module,
            ideal_answer_module=ideal_answer_module,
        )
    else:
        raise ValueError("Unknown documents module type.")

    questions = data.questions
    if first_questions is not None:
        questions = questions[:first_questions]

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
        ))
    echo(f"Answered {len(answered_data.questions)} questions.")
