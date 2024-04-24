from typing import Annotated, Sequence, TypeAlias, cast
from warnings import catch_warnings, simplefilter

from annotated_types import Len
from dspy import Signature, Prediction, InputField, OutputField, TypedPredictor
from pydantic import AfterValidator, BaseModel, Field
from spacy import load as spacy_load
from spacy.language import Language

from mibi.model import ListExactAnswerItem, PartiallyAnsweredQuestion, Question, PartialAnswer, YesNoExactAnswer, FactoidExactAnswer, ListExactAnswer
from mibi.modules.helpers import AutoExactAnswerModule


Context: TypeAlias = list[str]


class YesNoInput(BaseModel):
    question: Annotated[
        str,
        Field(
            description="The yes-no question that should be answered.",
        ),
    ]
    context: Annotated[
        Context,
        Field(
            description="Context that should be used to answer the question.",
        ),
    ]


class YesNoOutput(BaseModel):
    answer: Annotated[
        YesNoExactAnswer,
        Field(
            description="The yes-no answer to the given question.",
        ),
    ]


class YesNoPredict(Signature):
    """Answer the medical yes-no question based on the given context (from a relevant medical abstract), basic medical knowledge, and current standard practices from medical guidelines. The answer should be based mostly on the given context if it is factually correct."""

    input: Annotated[
        YesNoInput,
        InputField(),
    ]
    output: Annotated[
        YesNoOutput,
        OutputField(),
    ]


class FactoidInput(BaseModel):
    question: Annotated[
        str,
        Field(
            description="The factoid question that should be answered.",
        ),
    ]
    context: Annotated[
        Context,
        Field(
            description="Context that should be used to answer the question.",
        ),
    ]


def _check_short_answer(value: str) -> str:
    with catch_warnings():
        simplefilter(action="ignore", category=FutureWarning)
        language: Language = spacy_load("en_core_sci_sm")
    doc = language(value)

    num_tokens = sum(1 for _ in doc)
    if num_tokens > 5:
        raise ValueError("Must not be longer than 5 words.")

    return value


_ShortFactoidExactAnswer: TypeAlias = Annotated[
    FactoidExactAnswer,
    AfterValidator(_check_short_answer),
]


class FactoidOutput(BaseModel):
    answer: Annotated[
        _ShortFactoidExactAnswer,
        Field(
            description="The factoid answer to the given question. The answer should contain just the name of the entity, number, or other similar short expression sought by the question, not a complete sentence.",
        ),
    ]


class FactoidPredict(Signature):
    """Answer the medical factoid question based on the given context (from a relevant medical abstract), basic medical knowledge, and current standard practices from medical guidelines. The answer should be based mostly on the given context if it is factually correct."""

    input: Annotated[
        FactoidInput,
        InputField(),
    ]
    output: Annotated[
        FactoidOutput,
        OutputField(),
    ]


class ListInput(BaseModel):
    question: Annotated[
        str,
        Field(
            description="The list question that should be answered.",
        ),
    ]
    context: Annotated[
        Context,
        Field(
            description="Context that should be used to answer the question.",
        ),
    ]


_ShortListExactAnswerItem: TypeAlias = Annotated[
    ListExactAnswerItem,
    AfterValidator(_check_short_answer),
]

_ShortListExactAnswer: TypeAlias = Annotated[
    Sequence[_ShortListExactAnswerItem],
    Len(min_length=1),
]


class ListOutput(BaseModel):
    answer: Annotated[
        _ShortListExactAnswer,
        Field(
            description="The list answer to the given question. The answer should contain up to 5 short names of the entities sought by the question.",
        ),
    ]


class ListPredict(Signature):
    """Answer the medical list question based on the given context (from a relevant medical abstract), basic medical knowledge, and current standard practices from medical guidelines. The answer should be based mostly on the given context if it is factually correct."""

    input: Annotated[
        ListInput,
        InputField(),
    ]
    output: Annotated[
        ListOutput,
        OutputField(),
    ]


class LlmExactAnswerModule(AutoExactAnswerModule):
    _yes_no_predict = TypedPredictor(signature=YesNoPredict, max_retries=3)
    _factoid_predict = TypedPredictor(signature=FactoidPredict, max_retries=5)
    _list_predict = TypedPredictor(signature=ListPredict, max_retries=10)

    def _context(
            self,
            question: Question,
            partial_answer: PartialAnswer,
    ) -> Context:
        partially_answered_question = PartiallyAnsweredQuestion.from_question(
            question, partial_answer)

        context = []

        if partially_answered_question.exact_answer_text is not None:
            context += [partially_answered_question.exact_answer_text]
        if partially_answered_question.ideal_answer is not None:
            context += [partially_answered_question.ideal_answer]
        if partially_answered_question.snippets is not None:
            context += [
                snippet.text
                for snippet in partially_answered_question.snippets
            ]
        # TODO: Extract context from document.
        return context

    def forward_yes_no(
            self,
            question: Question,
            partial_answer: PartialAnswer,
    ) -> YesNoExactAnswer:
        input = YesNoInput(
            question=question.body,
            context=self._context(question, partial_answer)
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
            context=self._context(question, partial_answer)
        )
        prediction: Prediction = self._factoid_predict.forward(input=input)
        output = cast(FactoidOutput, prediction.output)
        # TODO: Additional validations using DSPy assertions (e.g., not too long)?
        return output.answer

    def forward_list(
            self,
            question: Question,
            partial_answer: PartialAnswer,
    ) -> ListExactAnswer:
        input = ListInput(
            question=question.body,
            context=self._context(question, partial_answer)
        )
        prediction: Prediction = self._list_predict.forward(input=input)
        output = cast(ListOutput, prediction.output)
        # TODO: Additional validations using DSPy assertions (e.g., not too long)?
        return output.answer
