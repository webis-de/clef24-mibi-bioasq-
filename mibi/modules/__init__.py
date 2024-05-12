from abc import ABC, ABCMeta, abstractmethod

from dspy import Module, ProgramMeta
from pydantic import JsonValue

from mibi.model import Question, PartialAnswer, Documents, Snippets, ExactAnswer, IdealAnswer, Answer


class _ABCProgramMeta(ABCMeta, ProgramMeta):
    pass


class ABCModule(ABC, Module, metaclass=_ABCProgramMeta):
    pass


class DocumentsModule(ABCModule):

    @abstractmethod
    def forward(
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
        return self.forward(question, partial_answer)


class SnippetsModule(ABCModule):

    @abstractmethod
    def forward(
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
        return self.forward(question, partial_answer)


class ExactAnswerModule(ABCModule):

    @abstractmethod
    def forward(
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
        return self.forward(question, partial_answer)


class IdealAnswerModule(ABCModule):

    @abstractmethod
    def forward(
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
        return self.forward(question, partial_answer)


class AnswerModule(ABCModule):

    @abstractmethod
    def forward(
        self,
        question: Question,
    ) -> Answer:
        raise NotImplementedError()

    def __call__(
        self,
        question: Question,
    ) -> Answer:
        return self.forward(question)


class JsonAnswerModule(Module):
    answer_module: AnswerModule

    def __init__(self, answer_module: AnswerModule) -> None:
        self.answer_module = answer_module

    def forward(
        self,
        question: JsonValue,
    ) -> Answer:
        return self.answer_module.forward(Question.model_validate(question))
