from mibi.model import Answer, AnsweredQuestion, AnsweredQuestionData, PartialAnswer, PartiallyAnsweredQuestion, PartiallyAnsweredQuestionData, Question, QuestionData, Snippet


def test_question_json_validation_schema() -> None:
    schema = Question.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_question_json_serialization_schema() -> None:
    schema = Question.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_question_data_json_validation_schema() -> None:
    schema = QuestionData.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_question_data_json_serialization_schema() -> None:
    schema = QuestionData.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_snippet_json_validation_schema() -> None:
    schema = Snippet.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_snippet_json_serialization_schema() -> None:
    schema = Snippet.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_answer_json_validation_schema() -> None:
    schema = Answer.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_answer_json_serialization_schema() -> None:
    schema = Answer.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_partial_answer_json_validation_schema() -> None:
    schema = PartialAnswer.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_partial_answer_json_serialization_schema() -> None:
    schema = PartialAnswer.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_answered_question_json_validation_schema() -> None:
    schema = AnsweredQuestion.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_answered_question_json_serialization_schema() -> None:
    schema = AnsweredQuestion.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_answered_question_data_json_validation_schema() -> None:
    schema = AnsweredQuestionData.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_answered_question_data_json_serialization_schema() -> None:
    schema = AnsweredQuestionData.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_partially_answered_question_json_validation_schema() -> None:
    schema = PartiallyAnsweredQuestion.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_partially_answered_question_json_serialization_schema() -> None:
    schema = PartiallyAnsweredQuestion.model_json_schema(mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"


def test_partially_answered_question_data_json_validation_schema() -> None:
    schema = PartiallyAnsweredQuestionData.model_json_schema(mode="validation")
    assert "type" in schema
    assert schema["type"] == "object"


def test_partially_answered_question_data_json_serialization_schema() -> None:
    schema = PartiallyAnsweredQuestionData.model_json_schema(
        mode="serialization")
    assert "type" in schema
    assert schema["type"] == "object"
