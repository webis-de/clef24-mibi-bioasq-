from functools import cached_property
from re import compile as re_compile
from typing import Annotated, Final, Literal, TypeAlias, Sequence, TypeVar

from annotated_types import Len, Ge
from pydantic import AfterValidator, AliasChoices, ConfigDict, Field, PlainSerializer, PlainValidator, SerializationInfo, TypeAdapter, UrlConstraints, BaseModel, WithJsonSchema
from pydantic_core import Url


QuestionType: TypeAlias = Literal[
    "yesno",
    "factoid",
    "list",
    "summary",
]

QuestionId: TypeAlias = Annotated[
    str,
    Len(min_length=24, max_length=25),
]


class Question(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: QuestionId
    type: QuestionType
    body: str


# _PUBMED_URL_PATTERN = re_compile(
#     r"https?:\/\/(?:pubmed\.ncbi\.nlm\.nih\.gov|(?:www\.)?ncbi\.nlm\.nih\.gov\/pubmed)\/[1-9][0-9]*\/?")
_PUBMED_URL_PATTERN = re_compile(
    r"http:\/\/www\.ncbi\.nlm\.nih\.gov\/pubmed\/[1-9][0-9]*")


def _validate_pubmed_url(url: Url) -> Url:
    if _PUBMED_URL_PATTERN.fullmatch(f"{url}") is None:
        raise ValueError(f"Not a valid PubMed URL: {url}")
    return url


PubMedUrl = Annotated[
    Url,
    UrlConstraints(
        max_length=2083,
        allowed_schemes=['http', 'https'],
        host_required=True,
    ),
    AfterValidator(_validate_pubmed_url),
]

Document: TypeAlias = PubMedUrl

_T1 = TypeVar("_T1")


def _cut_off_at_10(value: Sequence[_T1]) -> Annotated[Sequence[_T1], Len(max_length=10)]:
    return value[:10]


Documents: TypeAlias = Annotated[
    Sequence[Document],
    # PlainValidator(_cut_off_at_10),
    PlainSerializer(_cut_off_at_10),
]


def _clamp_positive(value: int) -> int:
    return 0 if value == -1 else value


class Snippet(BaseModel):
    model_config = ConfigDict(frozen=True)

    document: Document
    text: str
    begin_section: Annotated[
        str,
        Field(
            validation_alias=AliasChoices(
                "begin_section",
                "beginSection",
            ),
            serialization_alias="beginSection",
        )
    ]
    offset_in_begin_section: Annotated[
        int,
        Ge(0),
        Field(
            validation_alias=AliasChoices(
                "offset_in_begin_section",
                "offsetInBeginSection",
            ),
            serialization_alias="offsetInBeginSection",
        ),
        PlainValidator(_clamp_positive),
        WithJsonSchema(
            TypeAdapter(int).json_schema(),
            mode="validation",
        ),
        WithJsonSchema(
            TypeAdapter(Annotated[int, Ge(0)]).json_schema(),  # type: ignore
            mode="serialization",
        ),
    ]
    end_section: Annotated[
        str,
        Field(
            validation_alias=AliasChoices(
                "end_section",
                "endSection",
            ),
            serialization_alias="endSection",
        ),
    ]
    offset_in_end_section: Annotated[
        int,
        Ge(0),
        Field(
            validation_alias=AliasChoices(
                "offset_in_end_section",
                "offsetInEndSection",
            ),
            serialization_alias="offsetInEndSection",
        ),
    ]


Snippets: TypeAlias = Annotated[
    Sequence[Snippet],
    # PlainValidator(_cut_off_at_10),
    PlainSerializer(_cut_off_at_10),
]


def _first_str_of_json_sequence(
        value: Annotated[Sequence[str], Len(min_length=1)] | str) -> str:
    if isinstance(value, str):
        return value
    else:
        return value[0]


def _wrap_str_as_json_sequence(
    value: str,
    info: SerializationInfo,
) -> Annotated[Sequence[str], Len(min_length=1, max_length=1)] | str:
    if info.mode == "json":
        return [value]
    else:
        return value


IdealAnswer: TypeAlias = Annotated[
    str,
    PlainValidator(_first_str_of_json_sequence),
    PlainSerializer(_wrap_str_as_json_sequence),
    WithJsonSchema(
        TypeAdapter(
            Annotated[Sequence[str], Len(min_length=1)]
        ).json_schema(),  # type: ignore
        mode="validation",
    ),
    WithJsonSchema(
        TypeAdapter(
            Annotated[Sequence[str], Len(min_length=1, max_length=1)]
        ).json_schema(),  # type: ignore
        mode="serialization",
    ),
]

YesNoExactAnswer: TypeAlias = Literal["yes", "no"]


FactoidExactAnswer: TypeAlias = Annotated[
    str,
    PlainValidator(_first_str_of_json_sequence),
    PlainSerializer(_wrap_str_as_json_sequence),
    WithJsonSchema(
        TypeAdapter(
            Annotated[Sequence[str], Len(min_length=1)]
        ).json_schema(),  # type: ignore
        mode="validation",
    ),
    WithJsonSchema(
        TypeAdapter(
            Annotated[Sequence[str], Len(min_length=1, max_length=1)]
        ).json_schema(),  # type: ignore
        mode="serialization",
    ),
]


ListExactAnswerItem: TypeAlias = Annotated[
    str,
    PlainValidator(_first_str_of_json_sequence),
    PlainSerializer(_wrap_str_as_json_sequence),
    WithJsonSchema(
        TypeAdapter(
            Annotated[Sequence[str], Len(min_length=1)]
        ).json_schema(),  # type: ignore
        mode="validation",
    ),
    WithJsonSchema(
        TypeAdapter(
            Annotated[Sequence[str], Len(min_length=1, max_length=1)]
        ).json_schema(),  # type: ignore
        mode="serialization",
    ),
]


ListExactAnswer: TypeAlias = Sequence[ListExactAnswerItem]


SummaryExactAnswer: TypeAlias = Literal["n/a"]
NOT_AVAILABLE: Final[SummaryExactAnswer] = "n/a"


ExactAnswer: TypeAlias = Annotated[
    # Attention! The order of this union matters for serialization.
    YesNoExactAnswer | SummaryExactAnswer | ListExactAnswer | FactoidExactAnswer,
    Field(union_mode="left_to_right")
]


class PartialAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    documents: Documents | None = None
    snippets: Snippets | None = None
    ideal_answer: IdealAnswer | None = None
    exact_answer: ExactAnswer | None = None


class Answer(PartialAnswer, BaseModel):
    model_config = ConfigDict(frozen=True)

    documents: Documents
    snippets: Snippets
    ideal_answer: IdealAnswer
    exact_answer: ExactAnswer


class PartiallyAnsweredQuestion(PartialAnswer, Question):

    @classmethod
    def from_question(
        cls,
        question: Question,
        partial_answer: PartialAnswer | None = None,
    ) -> "PartiallyAnsweredQuestion":
        return PartiallyAnsweredQuestion(
            id=question.id,
            type=question.type,
            body=question.body,
            documents=partial_answer.documents
            if partial_answer is not None else None,
            snippets=partial_answer.snippets
            if partial_answer is not None else None,
            ideal_answer=partial_answer.ideal_answer
            if partial_answer is not None else None,
            exact_answer=partial_answer.exact_answer
            if partial_answer is not None else None,
        )

    @cached_property
    def exact_answer_text(self) -> str | None:
        if self.exact_answer is None:
            return None
        if self.type == "summary":
            if self.exact_answer is not NOT_AVAILABLE:
                raise ValueError()
            return None
        elif self.type == "yesno":
            if not isinstance(self.exact_answer, str):
                raise ValueError()
            return self.exact_answer
        elif self.type == "factoid":
            if not isinstance(self.exact_answer, str):
                raise ValueError()
            return self.exact_answer
        elif self.type == "list":
            if not isinstance(self.exact_answer, Sequence):
                raise ValueError()
            return ", ".join(self.exact_answer)
        else:
            raise ValueError(
                f"Unknown question type: {self.type}")

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
    model_config = ConfigDict(frozen=True)

    questions: Sequence[PartiallyAnsweredQuestion]


class AnsweredQuestion(PartiallyAnsweredQuestion, Answer, Question):
    pass


class AnsweredQuestionData(BaseModel):
    model_config = ConfigDict(frozen=True)

    questions: Sequence[AnsweredQuestion]


class QuestionData(BaseModel):
    model_config = ConfigDict(frozen=True)

    questions: Sequence[Question]
