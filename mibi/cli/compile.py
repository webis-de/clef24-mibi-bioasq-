from pathlib import Path
from typing import Literal, cast

from click import Choice, IntRange, echo, option, Path as PathType, argument, command

from mibi.metrics import DefaultMeasure
from mibi.model import Question


@command()
@argument(
    "training_data_path",
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
    "model_path",
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
    default="gpt-3.5-turbo-0125",
)
@option(
    "-n", "--first", "--first-questions", "first_questions",
    type=IntRange(min=0),
)
@option(
    "-o", "--optimizer", "optimizer_type",
    type=Choice([
        "bootstrap-few-shot", "bfs",
        "bootstrap-few-shot-with-random-search", "bfsrs",
        "mipro",
        "copro",
    ]),
    required=True,
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
def compile(
    training_data_path: Path,
    model_path: Path,
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
    optimizer_type: Literal[
        "bootstrap-few-shot", "bfs",
        "bootstrap-few-shot-with-random-search", "bfsrs",
        "copro",
        "mipro",
    ],
    elasticsearch_url: str | None,
    elasticsearch_username: str | None,
    elasticsearch_password: str | None,
    elasticsearch_index: str | None,
) -> None:
    from dspy import Module, Example
    from dspy.teleprompt import Teleprompter, BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPRO, COPRO
    from mibi.model import PartiallyAnsweredQuestionData, Answer
    from mibi.modules import JsonAnswerModule
    from mibi.modules.build import build_answer_module

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

    wrapped_answer_module = JsonAnswerModule(answer_module)

    with training_data_path.open("rb") as input_file:
        data = PartiallyAnsweredQuestionData.model_validate_json(
            input_file.read())
    echo(f"Found {len(data.questions)} training questions.")
    training_data: list[Example] = [
        Example({
            "question": Question(
                id=question.id,
                type=question.type,
                body=question.body,
            ).model_dump(mode="json"),
            "answer": Answer(
                documents=question.documents,
                snippets=question.snippets,
                exact_answer=question.exact_answer,
                ideal_answer=question.ideal_answer,
            ).model_dump(mode="json"),
        }).with_inputs("question")
        for question in data.questions
        if (
            question.documents is not None and
            question.snippets is not None and
            question.exact_answer is not None and
            question.ideal_answer is not None
        )
    ]
    echo(f"Found {len(training_data)} valid training examples.")

    if first_questions is not None:
        training_data = training_data[:first_questions]

    measure = DefaultMeasure()

    def metric(example: Example, predicted_answer: Answer, _trace=None) -> float:
        question: Question = Question.model_validate(example["question"])
        ground_truth_answer: Answer = Answer.model_validate(example["answer"])
        return measure.metric(
            question=question,
            ground_truth_answer=ground_truth_answer,
            predicted_answer=predicted_answer,
        )

    print("Create optimizer.")
    optimizer: Teleprompter
    if optimizer_type in ("bootstrap-few-shot", "bfs"):
        optimizer = BootstrapFewShot(
            metric=metric,
            metric_threshold=0.5,
            max_bootstrapped_demos=4,
            max_labeled_demos=16,
            max_rounds=1,
            max_errors=5,
        )
    elif optimizer_type in ("bootstrap-few-shot-with-random-search", "bfsrs"):
        optimizer = BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=16,
            max_rounds=1,
            num_candidate_programs=16,
            num_threads=1,
            max_errors=10,
        )
    elif optimizer_type == "mipro":
        optimizer = MIPRO(
            metric=metric,
            num_candidates=10,
            init_temperature=1.0,
            track_stats=True,
            verbose=True,
            view_data_batch_size=3,
        )
    elif optimizer_type == "copro":
        optimizer = COPRO(
            metric=metric,
            breadth=10,
            depth=3,
            init_temperature=1.0,
            track_stats=True,
            verbose=True,
        )
    else:
        raise ValueError("Unknown optimizer type.")

    print("Compile LLM program.")
    optimized_answer_module: Module
    if isinstance(optimizer, MIPRO):
        optimized_answer_module = cast(Module, optimizer.compile(
            student=wrapped_answer_module,
            trainset=training_data,
            num_trials=3,
            max_bootstrapped_demos=3,
            max_labeled_demos=3,
            eval_kwargs=dict(
                display_progress=True,
                display_table=True,
                display=True,
            ),
        ))
        if optimized_answer_module is None:
            raise RuntimeError("Could not opitimze answer module.")
    elif isinstance(optimizer, COPRO):
        optimized_answer_module = cast(Module, optimizer.compile(
            student=wrapped_answer_module,
            trainset=training_data,
            eval_kwargs=dict(
                display_progress=True,
                display_table=True,
                display=True,
            ),
        ))
    else:
        optimized_answer_module = optimizer.compile(
            student=wrapped_answer_module,
            trainset=training_data,
        )

    print(f"Saving LLM programm parameters to: {model_path}")
    optimized_answer_module.save(model_path)
