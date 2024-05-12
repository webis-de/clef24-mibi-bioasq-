from pydantic import TypeAdapter
from mibi.model import Answer, AnsweredQuestion, AnsweredQuestionData, PartialAnswer, PartiallyAnsweredQuestion, PartiallyAnsweredQuestionData, Question, QuestionData, Snippet


def test_question_json_validation_schema() -> None:
    schema = TypeAdapter(Question).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_question_json_serialization_schema() -> None:
    schema = TypeAdapter(Question).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_question_data_json_validation_schema() -> None:
    schema = TypeAdapter(QuestionData).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_question_data_json_serialization_schema() -> None:
    schema = TypeAdapter(QuestionData).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_snippet_json_validation_schema() -> None:
    schema = TypeAdapter(Snippet).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_snippet_json_serialization_schema() -> None:
    schema = TypeAdapter(Snippet).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_answer_json_validation_schema() -> None:
    schema = TypeAdapter(Answer).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_answer_json_serialization_schema() -> None:
    schema = TypeAdapter(Answer).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_partial_answer_json_validation_schema() -> None:
    schema = TypeAdapter(PartialAnswer).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_partial_answer_json_serialization_schema() -> None:
    schema = TypeAdapter(PartialAnswer).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_answered_question_json_validation_schema() -> None:
    schema = TypeAdapter(AnsweredQuestion).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_answered_question_json_serialization_schema() -> None:
    schema = TypeAdapter(AnsweredQuestion).json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_answered_question_data_json_validation_schema() -> None:
    schema = TypeAdapter(AnsweredQuestionData).json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_answered_question_data_json_serialization_schema() -> None:
    schema = TypeAdapter(AnsweredQuestionData).json_schema(
        mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_partially_answered_question_json_validation_schema() -> None:
    schema = TypeAdapter(PartiallyAnsweredQuestion).json_schema(
        mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_partially_answered_question_json_serialization_schema() -> None:
    schema = TypeAdapter(PartiallyAnsweredQuestion).json_schema(
        mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_partially_answered_question_data_json_validation_schema() -> None:
    schema = TypeAdapter(PartiallyAnsweredQuestionData).json_schema(
        mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_partially_answered_question_data_json_serialization_schema() -> None:
    schema = TypeAdapter(PartiallyAnsweredQuestionData).json_schema(
        mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"
