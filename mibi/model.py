from typing import Annotated, Final, Literal, TypeAlias, Sequence
from annotated_types import Len, Ge
from pydantic import AliasChoices, BaseModel, Field, UrlConstraints
from pydantic_core import Url


QuestionType: TypeAlias = Literal[
    "yesno",
    "factoid",
    "list",
    "summary",
]

QuestionId = Annotated[
    str,
    Len(min_length=24, max_length=25),
]


class Question(BaseModel):
    id: QuestionId
    type: QuestionType
    body: str


PubMedUrl = Annotated[
    Url,
    UrlConstraints(
        max_length=2083,
        allowed_schemes=['http', 'https'],
        host_required=True,
    ),
]

Document: TypeAlias = PubMedUrl

Documents: TypeAlias = Annotated[
    list[Document],
    Len(min_length=1),
]


class Snippet(BaseModel):
    document: Document = Field(frozen=True)
    text: str = Field(frozen=True)
    offset_in_begin_section: Annotated[
        int,
        Ge(0),
    ] = Field(
        frozen=True,
        validation_alias=AliasChoices(
            "offset_in_begin_section", "offsetInBeginSection"),
        serialization_alias="offsetInBeginSection",
    )
    offset_in_end_section: Annotated[
        int,
        Ge(0),
    ] = Field(
        frozen=True,
        validation_alias=AliasChoices(
            "offset_in_end_section", "offsetInEndSection"),
        serialization_alias="offsetInEndSection",
    )
    begin_section: str = Field(
        frozen=True,
        validation_alias=AliasChoices("begin_section", "beginSection"),
        serialization_alias="beginSection",
    )
    end_section: str = Field(
        frozen=True,
        validation_alias=AliasChoices("end_section", "endSection"),
        serialization_alias="endSection",
    )


Snippets: TypeAlias = Annotated[
    list[Snippet],
    Len(min_length=1),
]

YesNoExactAnswer: TypeAlias = Literal["yes", "no"]

FactoidExactAnswer: TypeAlias = Annotated[
    list[str],
    Len(min_length=1),
]

ListExactAnswer: TypeAlias = Annotated[
    Sequence[
        Annotated[
            list[str],
            Len(min_length=1),
        ]
    ],
    Len(min_length=1),
]


SummaryExactAnswer: TypeAlias = Literal["n/a"]
NOT_AVAILABLE: Final[SummaryExactAnswer] = "n/a"

IdealAnswer: TypeAlias = Annotated[
    list[str],
    Len(min_length=1),
]

ExactAnswer: TypeAlias = YesNoExactAnswer | FactoidExactAnswer | ListExactAnswer | SummaryExactAnswer


class PartialAnswer(BaseModel):
    documents: Documents | None = Field(frozen=True, default=None)
    snippets: Snippets | None = Field(frozen=True, default=None)
    ideal_answer: IdealAnswer | None = Field(frozen=True, default=None)
    exact_answer: ExactAnswer | None = Field(frozen=True, default=None)


class Answer(BaseModel):
    documents: Documents = Field(frozen=True)
    snippets: Snippets = Field(frozen=True)
    ideal_answer: IdealAnswer = Field(frozen=True)
    exact_answer: ExactAnswer = Field(frozen=True)


class TrainingQuestion(Question, PartialAnswer):
    pass


class TrainingData(BaseModel):
    questions: list[TrainingQuestion] = Field(frozen=True)
