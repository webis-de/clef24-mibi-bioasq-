from typing import TypeAlias, cast
from dspy import Module, Signature, Prediction, InputField, OutputField, TypedPredictor
from pydantic import BaseModel, Field

from mibi.model import Question, PartialAnswer, IdealAnswer
from mibi.modules import IdealAnswerModule


Context: TypeAlias = list[str]


class IdealInput(BaseModel):
    question: str = Field(
        description="The question that should be answered.")
    context: Context = Field(
        description="Context that should be used to answer the question.")


class IdealOutput(BaseModel):
    answer: str = Field(
        description="The long-form answer to the question consisting of 1 to 3 sentence that also contains a short explanation. The answer should be grammatically correct, concise, and precise.")


class IdealPredict(Signature):
    """Answer the medical question based on the given context (from a relevant medical abstract), basic medical knowledge, and current standard practices from medical guidelines. The answer should be based mostly on the given context if it is factually correct."""
    input: IdealInput = InputField()
    output: IdealOutput = OutputField()


class LlmIdealAnswerModule(Module, IdealAnswerModule):
    _ideal_predict = TypedPredictor(signature=IdealPredict)

    def _context(self, partial_answer: PartialAnswer) -> Context:
        if partial_answer.snippets is not None:
            return [
                snippet.text
                for snippet in partial_answer.snippets
            ]
        # TODO: Extract context from document.
        raise NotImplementedError()

    def forward(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> IdealAnswer:
        input = IdealInput(
            question=question.body,
            context=self._context(partial_answer)
        )
        prediction: Prediction = self._ideal_predict.forward(input=input)
        output = cast(IdealOutput, prediction.output)
        # TODO: Additional validations using DSPy assertions?
        return output.answer
