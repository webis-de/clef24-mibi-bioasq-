from typing import Protocol
from mibi.model import Question, PartialAnswer, Documents, Snippets, ExactAnswer, IdealAnswer, YesNoExactAnswer, Answer


class DocumentsModule(Protocol):
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


class SnippetsModule(Protocol):
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


class YesNoExactAnswerModule(Protocol):
    def forward(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> YesNoExactAnswer:
        raise NotImplementedError()

    def __call__(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> YesNoExactAnswer:
        return self.forward(question, partial_answer)


class ExactAnswerModule(Protocol):
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


class IdealAnswerModule(Protocol):
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


class AnswerModule(Protocol):
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
