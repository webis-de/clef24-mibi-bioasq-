from pydantic import TypeAdapter
from mibi.modules.incremental import NextTaskInput, NextTaskOutput


def test_next_task_input_json_validation_schema() -> None:
    schema = TypeAdapter(NextTaskInput).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_next_task_input_json_serialization_schema() -> None:
    schema = TypeAdapter(NextTaskInput).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_next_task_output_json_validation_schema() -> None:
    schema = TypeAdapter(NextTaskOutput).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_next_task_output_json_serialization_schema() -> None:
    schema = TypeAdapter(NextTaskOutput).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"

