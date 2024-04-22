from pathlib import Path

from click import UsageError, echo, Path as PathType, argument, group


@group()
def utils() -> None:
    pass


@utils.command()
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
    "results_paths",
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
    nargs=-1,
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
def merge(input_path: Path, results_paths: list[Path], output_path: Path) -> None:
    from mibi.model import QuestionData, PartiallyAnsweredQuestionData, QuestionId, PartiallyAnsweredQuestion

    with input_path.open("rb") as input_file:
        data = QuestionData.model_validate_json(input_file.read())
    questions: dict[QuestionId, PartiallyAnsweredQuestion] = {
        question.id: PartiallyAnsweredQuestion.from_question(question)
        for question in data.questions
    }
    question_ids = set(questions.keys())
    echo(f"Found {len(questions)} questions.")

    for results_path in results_paths:
        with results_path.open("rb") as input_file:
            results_data = PartiallyAnsweredQuestionData.model_validate_json(
                input_file.read())
        results_questions = {
            question.id: question
            for question in results_data.questions
        }
        results_question_ids = set(results_questions.keys())
        missing_question_ids = question_ids - results_question_ids
        additional_question_ids = results_question_ids - question_ids
        if len(missing_question_ids):
            raise UsageError(
                f"Results file at {results_path} is missing questions from the input file at {input_path}. Missing IDs: {', '.join(missing_question_ids)}")
        if len(additional_question_ids):
            raise UsageError(
                f"Results file at {results_path} contains additional questions not found in the input file at {input_path}. Additional IDs: {', '.join(additional_question_ids)}")

        questions = {
            question_id: question.merge(results_questions[question_id])
            for question_id, question in questions.items()
        }
        echo(f"Merged {len(results_questions)} questions.")

    output_data = PartiallyAnsweredQuestionData(
        questions=list(questions.values()),
    )
    with output_path.open("wt") as output_file:
        output_file.write(output_data.model_dump_json(
            indent=2,
            by_alias=True,
        ))
    echo(f"Answered {len(output_data.questions)} questions.")
