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
    Sequence[Document],
    Len(min_length=1, max_length=10),
]


class Snippet(BaseModel):
    document: Document = Field(frozen=True)
    text: str = Field(frozen=True)
    begin_section: str = Field(
        frozen=True,
        validation_alias=AliasChoices("begin_section", "beginSection"),
        serialization_alias="beginSection",
    )
    offset_in_begin_section: Annotated[
        int,
        Ge(0),
    ] = Field(
        frozen=True,
        validation_alias=AliasChoices(
            "offset_in_begin_section", "offsetInBeginSection"),
        serialization_alias="offsetInBeginSection",
    )
    end_section: str = Field(
        frozen=True,
        validation_alias=AliasChoices("end_section", "endSection"),
        serialization_alias="endSection",
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


Snippets: TypeAlias = Annotated[
    Sequence[Snippet],
    Len(min_length=1, max_length=10),
]

YesNoExactAnswer: TypeAlias = Literal["yes", "no"]

FactoidExactAnswer: TypeAlias = Annotated[
    Sequence[str],
    Len(min_length=1),
]

ListExactAnswer: TypeAlias = Annotated[
    Sequence[
        Annotated[
            Sequence[str],
            Len(min_length=1),
        ]
    ],
    Len(min_length=1),
]


SummaryExactAnswer: TypeAlias = Literal["n/a"]
NOT_AVAILABLE: Final[SummaryExactAnswer] = "n/a"

IdealAnswer: TypeAlias = str

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


class AnsweredQuestion(Answer, Question):
    pass


class AnsweredQuestionData(BaseModel):
    questions: Sequence[AnsweredQuestion] = Field(frozen=True)


class PartiallyAnsweredQuestion(PartialAnswer, Question):

    @classmethod
    def from_question(cls, question: Question) -> "PartiallyAnsweredQuestion":
        return PartiallyAnsweredQuestion(
            id=question.id,
            type=question.type,
            body=question.body,
        )

    def merge(self, question: "PartiallyAnsweredQuestion") -> "PartiallyAnsweredQuestion":
        if self.id != question.id:
            raise ValueError(
                f"Question IDs do not match: "
                f"'{self.id}' != '{question.id}'")
        if self.type != question.type:
            raise ValueError(
                f"Question types do not match: "
                f"'{self.type}' != '{question.type}'")
        if self.body != question.body:
            raise ValueError(
                f"Question bodys do not match: "
                f"'{self.body}' != '{question.body}'")
        return PartiallyAnsweredQuestion(
            id=self.id,
            type=self.type,
            body=self.body,
            documents=question.documents if question.documents is not None else self.documents,
            snippets=question.snippets if question.snippets is not None else self.snippets,
            ideal_answer=question.ideal_answer if question.ideal_answer is not None else self.ideal_answer,
            exact_answer=question.exact_answer if question.exact_answer is not None else self.exact_answer,
        )


class PartiallyAnsweredQuestionData(BaseModel):

    questions: Sequence[PartiallyAnsweredQuestion] = Field(frozen=True)


class QuestionData(BaseModel):
    questions: Sequence[Question] = Field(frozen=True)
