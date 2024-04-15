from abc import ABC, ABCMeta, abstractmethod

from dspy import Module, ProgramMeta

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
