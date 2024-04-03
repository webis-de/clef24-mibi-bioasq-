from pathlib import Path

from click import echo, Path as PathType, argument, command


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
def parse(input_path: Path) -> None:
    from mibi.model import QuestionData

    with input_path.open("rb") as input_file:
        data = QuestionData.model_validate_json(input_file.read())
    echo(f"Found {len(data.questions)} questions.")
