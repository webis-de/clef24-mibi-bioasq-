from math import nan
from pathlib import Path
from typing import Collection, Literal

from click import Choice, IntRange, echo, option, Path as PathType, argument, command

from mibi.model import NOT_AVAILABLE, Question


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
    elasticsearch_url: str | None,
    elasticsearch_username: str | None,
    elasticsearch_password: str | None,
    elasticsearch_index: str | None,
) -> None:
    from statistics import mean, harmonic_mean
    from dspy import Module, Example
    from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPRO
    from rouge_score.rouge_scorer import RougeScorer
    from rouge_score.scoring import Score
    from mibi.model import PartiallyAnsweredQuestionData, Answer
    from mibi.modules.build import build_answer_module

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
            ),
            "answer": Answer(
                documents=question.documents,
                snippets=question.snippets,
                exact_answer=question.exact_answer,
                ideal_answer=question.ideal_answer,
            ),
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
    if not isinstance(answer_module, Module):
        raise RuntimeError("Can only optimize DSPy modules.")

    rouge_scorer = RougeScorer(
        rouge_types=["rouge1", "rougeL"],
        use_stemmer=True,
        tokenizer=None,  # TODO: Default?
    )

    def metric(example: Example, predicted_answer: Answer, _trace=None) -> float:
        question: Question = example["question"]
        ground_truth_answer: Answer = example["answer"]

        exact_answer_score: float
        if question.type == "yesno":
            if ground_truth_answer.exact_answer not in ("yes", "no") or predicted_answer.exact_answer not in ("yes", "no"):
                raise RuntimeError(
                    "Expected exact answer to be either 'yes' or 'no'.")
            exact_answer_score = 1 if ground_truth_answer.exact_answer == predicted_answer.exact_answer else 0
        elif question.type == "factoid":
            if not isinstance(ground_truth_answer.exact_answer, str) or not isinstance(predicted_answer.exact_answer, str):
                raise RuntimeError("Expected exact answer to be a string.")
            # TODO: Stemming?
            exact_answer_score = 1 if ground_truth_answer.exact_answer == predicted_answer.exact_answer else 0
        elif question.type == "list":
            if not isinstance(ground_truth_answer.exact_answer, Collection) or not isinstance(predicted_answer.exact_answer, Collection):
                raise RuntimeError("Expected exact answer to be a collection.")
            # TODO: Stemming?
            predicted_exact_answer_set: set[str] = set(
                predicted_answer.exact_answer)
            ground_truth_exact_answer_set: set[str] = set(
                ground_truth_answer.exact_answer)
            correct_exact_answer_set = predicted_exact_answer_set & ground_truth_exact_answer_set
            exact_answer_precision = (
                (
                    len(correct_exact_answer_set) /
                    len(predicted_exact_answer_set)
                )
                if len(predicted_exact_answer_set) > 0 else 0
            )
            exact_answer_recall = (
                (
                    len(correct_exact_answer_set) /
                    len(ground_truth_exact_answer_set)
                )
                if len(ground_truth_exact_answer_set) > 0 else 0
            )
            exact_answer_f1 = (
                (
                    2 * (exact_answer_precision * exact_answer_recall) /
                    (exact_answer_precision + exact_answer_recall)
                )
                if exact_answer_precision + exact_answer_recall > 0 else 0
            )
            exact_answer_score = exact_answer_f1
        elif question.type == "summary":
            if ground_truth_answer.exact_answer != NOT_AVAILABLE or predicted_answer.exact_answer != NOT_AVAILABLE:
                raise RuntimeError("Expected exact answer to be empty.")
            exact_answer_score = nan
        else:
            raise RuntimeError("Unknown exact answer type.")
        print(f"Exact answer score ({question.type}): {exact_answer_score:.2f} (GT: {ground_truth_answer.exact_answer}, P: {predicted_answer.exact_answer})")

        ideal_answer_rouge_scores = rouge_scorer.score(
            target=ground_truth_answer.ideal_answer,
            prediction=predicted_answer.ideal_answer,
        )
        ideal_answer_rouge_1: Score = ideal_answer_rouge_scores["rouge1"]
        ideal_answer_rouge_l: Score = ideal_answer_rouge_scores["rougeL"]
        ideal_answer_score = harmonic_mean((
            ideal_answer_rouge_1.fmeasure,
            ideal_answer_rouge_l.fmeasure
        ))
        print(f"Ideal answer score: {ideal_answer_score:.2f} (GT: {ground_truth_answer.ideal_answer}, P: {predicted_answer.ideal_answer})")

        answer_score: float
        if question.type == "summary":
            answer_score = ideal_answer_score
        else:
            answer_score = mean((
                exact_answer_score,
                ideal_answer_score,
            ))
        print(f"Answer score: {answer_score:.2f}")

        return answer_score

    print("Create optimizer.")
    optimizer = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        max_errors=0,  # For debugging.
    )
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=metric,
        max_bootstrapped_demos=3,
        max_labeled_demos=3,
        num_candidate_programs=10,
        num_threads=1,
    )
    # optimizer = MIPRO(
    #     metric=metric,
    #     num_candidates=10,
    #     init_temperature=1.0,
    #     track_stats=True,
    #     verbose=True,
    #     view_data_batch_size=10,
    # )

    print("Compile LLM program.")
    optimized_answer_module: Module = optimizer.compile(
        student=answer_module,
        trainset=training_data,
        # num_trials=3,  # TODO
        # max_bootstrapped_demos=3,  # TODO
        # max_labeled_demos=3,  # TODO
        # eval_kwargs=dict(
        #     display_progress=True,
        #     display_table=True,
        #     display=True,
        #     max_errors=1,
        # ),
    )  # type: ignore

    print(f"Saving LLM programm parameters to: {model_path}")
    optimized_answer_module.save(model_path)
