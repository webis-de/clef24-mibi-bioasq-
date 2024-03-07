
from mibi.builder import AnswerBuilder
from mibi.model import Question
from mibi.modules.mock import MockDocumentsMaker, MockSnippetsMaker, MockExactAnswerMaker, MockIdealAnswerMaker

def test_answer_builder() -> None:
    question = Question(
        id="6415c252690f196b51000011",
        type="factoid",
        body="Which cancer is the BCG vaccine used for?",
    )
    builder = AnswerBuilder(
        question=question,
        documents_maker=MockDocumentsMaker(),
        snippets_maker=MockSnippetsMaker(),
        exact_answer_maker=MockExactAnswerMaker(),
        ideal_answer_maker=MockIdealAnswerMaker(),
    )
    assert builder.partial_answer.documents is None
    assert builder.partial_answer.snippets is None
    assert builder.partial_answer.exact_answer is None
    assert builder.partial_answer.ideal_answer is None
    assert not builder.is_ready
    builder.make_documents()
    assert builder.partial_answer.documents is not None
    assert builder.partial_answer.snippets is None
    assert builder.partial_answer.exact_answer is None
    assert builder.partial_answer.ideal_answer is None
    assert not builder.is_ready
    builder.make_snippets()
    assert builder.partial_answer.documents is not None
    assert builder.partial_answer.snippets is not None
    assert builder.partial_answer.exact_answer is None
    assert builder.partial_answer.ideal_answer is None
    assert not builder.is_ready
    builder.make_exact_answer()
    assert builder.partial_answer.documents is not None
    assert builder.partial_answer.snippets is not None
    assert builder.partial_answer.exact_answer is not None
    assert builder.partial_answer.ideal_answer is None
    assert not builder.is_ready
    builder.make_ideal_answer()
    assert builder.partial_answer.documents is not None
    assert builder.partial_answer.snippets is not None
    assert builder.partial_answer.exact_answer is not None
    assert builder.partial_answer.ideal_answer is not None
    assert builder.is_ready
    builder.make_documents()
    assert builder.partial_answer.documents is not None
    assert builder.partial_answer.snippets is not None
    assert builder.partial_answer.exact_answer is not None
    assert builder.partial_answer.ideal_answer is not None
    assert builder.is_ready
    builder.make_snippets()
    assert builder.partial_answer.documents is not None
    assert builder.partial_answer.snippets is not None
    assert builder.partial_answer.exact_answer is not None
    assert builder.partial_answer.ideal_answer is not None
    assert builder.is_ready
    builder.make_exact_answer()
    assert builder.partial_answer.documents is not None
    assert builder.partial_answer.snippets is not None
    assert builder.partial_answer.exact_answer is not None
    assert builder.partial_answer.ideal_answer is not None
    assert builder.is_ready
    builder.make_ideal_answer()
    assert builder.partial_answer.documents is not None
    assert builder.partial_answer.snippets is not None
    assert builder.partial_answer.exact_answer is not None
    assert builder.partial_answer.ideal_answer is not None
    assert builder.is_ready
