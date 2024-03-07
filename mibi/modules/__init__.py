from typing import Protocol
from mibi.model import Question, PartialAnswer, Documents, Snippets, ExactAnswer, IdealAnswer, YesNoExactAnswer, FactoidExactAnswer, ListExactAnswer, SummaryExactAnswer, NOT_AVAILABLE


class DocumentsMaker(Protocol):
    def make_documents(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> Documents:
        raise NotImplementedError()

    def __call__(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> Documents:
        return self.make_documents(question, partial_answer)


class SnippetsMaker(Protocol):
    def make_snippets(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> Snippets:
        raise NotImplementedError()

    def __call__(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> Snippets:
        return self.make_snippets(question, partial_answer)


class ExactAnswerMaker(Protocol):
    def make_exact_answer(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> ExactAnswer:
        raise NotImplementedError()

    def __call__(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> ExactAnswer:
        return self.make_exact_answer(question, partial_answer)


class SwitchExactAnswerMaker(ExactAnswerMaker, Protocol):
    def make_exact_answer(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> ExactAnswer:
        if question.type == "yesno":
            return self.make_yes_no_exact_answer(question, partial_answer)
        elif question.type == "factoid":
            return self.make_factoid_exact_answer(question, partial_answer)
        elif question.type == "list":
            return self.make_list_exact_answer(question, partial_answer)
        elif question.type == "summary":
            return self.make_summary_exact_answer(question, partial_answer)
        else:
            raise ValueError(f"Unknown question type: {question.type}")

    def make_yes_no_exact_answer(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> YesNoExactAnswer:
        raise NotImplementedError()

    def make_factoid_exact_answer(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> FactoidExactAnswer:
        raise NotImplementedError()

    def make_list_exact_answer(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> ListExactAnswer:
        raise NotImplementedError()

    def make_summary_exact_answer(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> SummaryExactAnswer:
        return NOT_AVAILABLE


class IdealAnswerMaker(Protocol):
    def make_ideal_answer(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> IdealAnswer:
        raise NotImplementedError()

    def __call__(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> IdealAnswer:
        return self.make_ideal_answer(question, partial_answer)
