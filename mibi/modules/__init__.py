from typing import Protocol
from mibi.model import Question, PartialAnswer, Documents, Snippets, ExactAnswer, IdealAnswer


class DocumentsMaker(Protocol):
    def make_documents(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> Documents:
        raise NotImplementedError()


class SnippetsMaker(Protocol):
    def make_snippets(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> Snippets:
        raise NotImplementedError()


class ExactAnswerMaker(Protocol):
    def make_exact_answer(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> ExactAnswer:
        raise NotImplementedError()


class IdealAnswerMaker(Protocol):
    def make_ideal_answer(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> IdealAnswer:
        raise NotImplementedError()
