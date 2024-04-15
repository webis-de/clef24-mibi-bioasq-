from typing import TypeAlias, cast

from dspy import Module, Signature, Prediction, InputField, OutputField, TypedPredictor
from pydantic import BaseModel, Field

from mibi.model import Question, PartialAnswer, YesNoExactAnswer, FactoidExactAnswer, ListExactAnswer
from mibi.modules.helpers import AutoExactAnswerModule


Context: TypeAlias = list[str]


class YesNoInput(BaseModel):
    question: str = Field(
        description="The yes-no question that should be answered.")
    context: Context = Field(
        description="Context that should be used to answer the question.")


class YesNoOutput(BaseModel):
    answer: YesNoExactAnswer = Field(
        description="The yes-no answer to the given question.")


class YesNoPredict(Signature):
    """Answer the medical yes-no question based on the given context (from a relevant medical abstract), basic medical knowledge, and current standard practices from medical guidelines. The answer should be based mostly on the given context if it is factually correct."""
    input: YesNoInput = InputField()
    output: YesNoOutput = OutputField()


class FactoidInput(BaseModel):
    question: str = Field(
        description="The factoid question that should be answered.")
    context: Context = Field(
        description="Context that should be used to answer the question.")


class FactoidOutput(BaseModel):
    answer: str = Field(
        description="The factoid answer to the given question. The answer should be just a short fact (e.g., a single entity or a very short phrase).")


class FactoidPredict(Signature):
    """Answer the medical factoid question based on the given context (from a relevant medical abstract), basic medical knowledge, and current standard practices from medical guidelines. The answer should be based mostly on the given context if it is factually correct."""
    input: FactoidInput = InputField()
    output: FactoidOutput = OutputField()


class ListInput(BaseModel):
    question: str = Field(
        description="The list question that should be answered.")
    context: Context = Field(
        description="Context that should be used to answer the question.")


class ListOutput(BaseModel):
    answer: list[str] = Field(
        description="The list answer to the given question. The answer should contain up to 5 entities.")


class ListPredict(Signature):
    """Answer the medical list question based on the given context (from a relevant medical abstract), basic medical knowledge, and current standard practices from medical guidelines. The answer should be based mostly on the given context if it is factually correct."""
    input: ListInput = InputField()
    output: ListOutput = OutputField()


class LlmExactAnswerModule(Module, AutoExactAnswerModule):
    _yes_no_predict = TypedPredictor(signature=YesNoPredict)
    _factoid_predict = TypedPredictor(signature=FactoidPredict)
    _list_predict = TypedPredictor(signature=ListPredict)

    def _context(self, partial_answer: PartialAnswer) -> Context:
        if partial_answer.snippets is not None:
            return [
                snippet.text
                for snippet in partial_answer.snippets
            ]
        # TODO: Extract context from document.
        raise NotImplementedError()

    def forward_yes_no(
            self,
            question: Question,
            partial_answer: PartialAnswer,
    ) -> YesNoExactAnswer:
        input = YesNoInput(
            question=question.body,
            context=self._context(partial_answer)
        )
        prediction: Prediction = self._yes_no_predict.forward(input=input)
        output = cast(YesNoOutput, prediction.output)
        return output.answer

    def forward_factoid(
            self,
            question: Question,
            partial_answer: PartialAnswer,
    ) -> FactoidExactAnswer:
        input = FactoidInput(
            question=question.body,
            context=self._context(partial_answer)
        )
        prediction: Prediction = self._factoid_predict.forward(input=input)
        output = cast(FactoidOutput, prediction.output)
        # TODO: Additional validations using DSPy assertions?
        return [output.answer]

    def forward_list(
            self,
            question: Question,
            partial_answer: PartialAnswer,
    ) -> ListExactAnswer:
        input = ListInput(
            question=question.body,
            context=self._context(partial_answer)
        )
        prediction: Prediction = self._list_predict.forward(input=input)
        output = cast(ListOutput, prediction.output)
        # TODO: Additional validations using DSPy assertions?
        return [[item] for item in output.answer]
