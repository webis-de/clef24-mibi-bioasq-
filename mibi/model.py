from typing import Annotated, Generic, Literal, TypeVar, TypeAlias
from annotated_types import Len
from pydantic import HttpUrl, BaseModel, Field


QuestionType = Literal[
    "yesno",
    "factoid",
    "list",
    "summary",
]

Q = TypeVar("Q", bound=QuestionType)

QuestionId = Annotated[
    str,
    Len(min_length=24, max_length=25),
]


class Question(BaseModel, Generic[Q]):
    id: QuestionId
    type: Q
    body: str


Document: TypeAlias = HttpUrl

Documents: TypeAlias = list[Document]


class Snippet(BaseModel):
    document: Document
    text: str
    offset_in_begin_section: int = Field(
        alias="offsetInBeginSection",
    )
    offset_in_end_section: int = Field(
        alias="offsetInEndSection",
    )
    begin_section: str = Field(
        alias="beginSection",
    )
    end_section: str = Field(
        alias="endSection",
    )


Snippets: TypeAlias = list[Snippet]

YesNoExactAnswer: TypeAlias = Literal["yes", "no"]

FactoidExactAnswer: TypeAlias = Annotated[
    list[str],
    Len(min_length=1),
]

ListExactAnswer: TypeAlias = Annotated[
    list[
        Annotated[
            list[str],
            Len(min_length=1),
        ]
    ],
    Len(min_length=1),
]

SummaryExactAnswer: TypeAlias = None

IdealAnswer: TypeAlias = Annotated[
    list[str],
    Len(min_length=1),
]

ExactAnswer: TypeAlias = YesNoExactAnswer | FactoidExactAnswer | ListExactAnswer | SummaryExactAnswer


class PartialAnswer(BaseModel):
    documents: Documents | None = None
    snippets: Snippets | None = None
    ideal_answer: IdealAnswer | None = None
    exact_answer: ExactAnswer | None = None


class Answer(BaseModel):
    documents: Documents
    snippets: Snippets
    ideal_answer: IdealAnswer
    exact_answer: ExactAnswer


class TrainingQuestion(Question, PartialAnswer):
    pass


class TrainingData(BaseModel):
    questions: list[TrainingQuestion]
