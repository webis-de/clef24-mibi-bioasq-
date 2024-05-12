from pydantic import TypeAdapter
from mibi.modules.ideal_answer.llm import IdealInput, IdealOutput


def test_ideal_input_json_validation_schema() -> None:
    schema = TypeAdapter(IdealInput).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_ideal_input_json_serialization_schema() -> None:
    schema = TypeAdapter(IdealInput).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_ideal_output_json_validation_schema() -> None:
    schema = TypeAdapter(IdealOutput).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_ideal_output_json_serialization_schema() -> None:
    schema = TypeAdapter(IdealOutput).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"

