from pydantic import TypeAdapter
from mibi.modules.exact_answer.llm import FactoidInput, FactoidOutput, ListInput, ListOutput, YesNoInput, YesNoOutput


def test_yes_no_input_json_validation_schema() -> None:
    schema = TypeAdapter(YesNoInput).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_yes_no_input_json_serialization_schema() -> None:
    schema = TypeAdapter(YesNoInput).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_yes_no_output_json_validation_schema() -> None:
    schema = TypeAdapter(YesNoOutput).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_yes_no_output_json_serialization_schema() -> None:
    schema = TypeAdapter(YesNoOutput).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_factoid_input_json_validation_schema() -> None:
    schema = TypeAdapter(FactoidInput).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_factoid_input_json_serialization_schema() -> None:
    schema = TypeAdapter(FactoidInput).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_factoid_output_json_validation_schema() -> None:
    schema = TypeAdapter(FactoidOutput).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_factoid_output_json_serialization_schema() -> None:
    schema = TypeAdapter(FactoidOutput).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_list_input_json_validation_schema() -> None:
    schema = TypeAdapter(ListInput).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_list_input_json_serialization_schema() -> None:
    schema = TypeAdapter(ListInput).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_list_output_json_validation_schema() -> None:
    schema = TypeAdapter(ListOutput).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_list_output_json_serialization_schema() -> None:
    schema = TypeAdapter(ListOutput).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"
