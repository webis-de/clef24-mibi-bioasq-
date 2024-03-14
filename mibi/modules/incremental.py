from typing import Callable, Literal, ParamSpec, TypeAlias, TypeVar, cast
from warnings import warn
from dspy import Module, Signature, Prediction, InputField, OutputField, TypedPredictor
from pydantic import BaseModel, Field
from pyrate_limiter.limiter import Limiter
from pyrate_limiter import Rate, Duration
from mibi.builder import AnswerBuilder
from mibi.model import Question, Answer, QuestionType
from mibi.modules import DocumentsModule, SnippetsModule, ExactAnswerModule, IdealAnswerModule


Task: TypeAlias = Literal[
    'retrieve documents',
    'retrieve snippets',
    'generate exact answer',
    'generate summary answer',
    'none',
]


YesNo: TypeAlias = Literal[
    'yes',
    'no',
]


class PredictNextTaskInput(BaseModel):
    question: str = Field(description="The question that should be answered.")
    question_type: QuestionType = Field(
        description="How the question should be answered.")
    has_documents: bool = Field(
        description="If documents were already retrieved for the question.")
    has_snippets: bool = Field(
        description="If snippets were already retrieved for the question.")
    has_exact_answer: bool = Field(
        description="If an exact answer was already generated for the question.")
    has_ideal_answer: bool = Field(
        description="If a human summary answer was already generated for the question.")
    last_task: Task = Field(description="The last task that was run.")
    last_task_successful: bool = Field(
        description="Whether the last task was successful.")


class PredictNextTaskOutput(BaseModel):
    task: Task = Field(
        description="The next task that should be run in order to answer the question. The task 'none' should only be chosen once all other tasks were run at least once.")


class PredictNextTask(Signature):
    """Chose the next task to run, in order to get the full answer. The next task should be chosen as to optimally answer the question given the data at hand. No specific order of the tasks is enforced. Tasks can be repeated."""
    input: PredictNextTaskInput = InputField()
    output: PredictNextTaskOutput = OutputField()


_limiter = Limiter([
    Rate(limit=1, interval=Duration.SECOND),
    Rate(limit=500, interval=Duration.HOUR),
], raise_when_fail=False)


def _limiter_mapping(*_args, **_kwargs):
    return "incremental", 1


_T = TypeVar("_T")
_P = ParamSpec("_P")


def _limit(f: Callable[_P, _T]) -> Callable[_P, _T]:
    return _limiter.as_decorator()(_limiter_mapping)(f)  # type: ignore


class IncrementalAnswerModule(
    Module,
    # AnswerModule,
):
    """
    Incrementally build the full answer by asking the LLM which
    part of the answer (i.e., documents, snippets, exact answer,
    or ideal answer) to make next.
    """

    _documents_module: DocumentsModule
    _snippets_module: SnippetsModule
    _exact_answer_module: ExactAnswerModule
    _ideal_answer_module: IdealAnswerModule
    _predict_next_task = TypedPredictor(signature=PredictNextTask)

    def __init__(
        self,
        documents_module: DocumentsModule,
        snippets_module: SnippetsModule,
        exact_answer_module: ExactAnswerModule,
        ideal_answer_module: IdealAnswerModule,
    ) -> None:
        self._documents_module = documents_module
        self._snippets_module = snippets_module
        self._exact_answer_module = exact_answer_module
        self._ideal_answer_module = ideal_answer_module

    @_limit
    def _next_task(
        self,
        builder: AnswerBuilder,
        last_task: Task,
        last_task_successful: bool,
    ) -> Task:
        input = PredictNextTaskInput(
            question=builder.question.body,
            question_type=builder.question.type,
            has_documents=builder.has_documents,
            has_snippets=builder.has_snippets,
            has_exact_answer=builder.has_exact_answer,
            has_ideal_answer=builder.has_ideal_answer,
            last_task=last_task,
            last_task_successful=last_task_successful,
        )
        prediction: Prediction = self._predict_next_task.forward(input=input)
        output = cast(PredictNextTaskOutput, prediction.output)
        return output.task

    def forward(self, question: Question) -> Answer:
        builder = AnswerBuilder(
            question=question,
            documents_module=self._documents_module,
            snippets_module=self._snippets_module,
            exact_answer_module=self._exact_answer_module,
            ideal_answer_module=self._ideal_answer_module,
        )
        next_task: Task = "none"
        last_task: Task = "none"
        last_task_successful: bool = True
        while not builder.is_ready or next_task != "none":
            next_task = self._next_task(
                builder=builder,
                last_task=last_task,
                last_task_successful=last_task_successful,
            )
            if next_task == last_task:
                warn("Cannot run the same task twice in a row.")
                last_task_successful = False
            elif next_task == "retrieve documents":
                print(
                    f"Making documents for question '{builder.question.body[:100]}'...")
                builder.make_documents()
                last_task_successful = True
            elif next_task == "retrieve snippets":
                print(
                    f"Making snippets for question '{builder.question.body[:100]}'...")
                builder.make_snippets()
                last_task_successful = True
            elif next_task == "generate exact answer":
                print(
                    f"Making exact for question '{builder.question.body[:100]}'...")
                builder.make_exact_answer()
                last_task_successful = True
            elif next_task == "generate summary answer":
                print(
                    f"Making ideal for question '{builder.question.body[:100]}'...")
                builder.make_ideal_answer()
                last_task_successful = True
            elif next_task == "none":
                if builder.is_ready:
                    print("Returning answer...")
                    return builder.answer
                else:
                    warn("Attempted to return answer but answer was not ready yet.")
                    last_task_successful = False
            else:
                raise RuntimeWarning(f"Unknown task: {next_task}")
            last_task = next_task
        return builder.answer
