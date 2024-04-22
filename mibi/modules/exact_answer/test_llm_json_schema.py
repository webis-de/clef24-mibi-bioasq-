from mibi.modules.exact_answer.llm import FactoidInput, FactoidOutput, ListInput, ListOutput, YesNoInput, YesNoOutput


def test_yes_no_input_json_validation_schema() -> None:
    schema = YesNoInput.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_yes_no_input_json_serialization_schema() -> None:
    schema = YesNoInput.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_yes_no_output_json_validation_schema() -> None:
    schema = YesNoOutput.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_yes_no_output_json_serialization_schema() -> None:
    schema = YesNoOutput.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_factoid_input_json_validation_schema() -> None:
    schema = FactoidInput.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_factoid_input_json_serialization_schema() -> None:
    schema = FactoidInput.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_factoid_output_json_validation_schema() -> None:
    schema = FactoidOutput.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_factoid_output_json_serialization_schema() -> None:
    schema = FactoidOutput.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_list_input_json_validation_schema() -> None:
    schema = ListInput.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_list_input_json_serialization_schema() -> None:
    schema = ListInput.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_list_output_json_validation_schema() -> None:
    schema = ListOutput.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_list_output_json_serialization_schema() -> None:
    schema = ListOutput.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"

