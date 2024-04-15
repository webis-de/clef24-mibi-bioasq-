from typing import Literal, Sequence, TypeAlias, cast
from warnings import warn
from dspy import Module, Signature, Prediction, InputField, OutputField, TypedPredictor
from pydantic import BaseModel, Field
from pyrate_limiter.limiter import Limiter
from pyrate_limiter import Rate, Duration
from mibi.builder import AnswerBuilder
from mibi.model import Question, Answer, QuestionType
from mibi.modules import AnswerModule, DocumentsModule, SnippetsModule, ExactAnswerModule, IdealAnswerModule
from mibi.utils.rate_limiting import rate_limit


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


class HistoryItem(BaseModel):
    task: Task = Field(description="The task that was run.")
    successful: bool = Field(
        description="Whether the task was run successful.")


class NextTaskInput(BaseModel):
    question: str = Field(description="The question that should be answered.")
    question_type: QuestionType = Field(
        description="How the question should be answered.")
    history: Sequence[HistoryItem] = Field(
        description="The history of tasks that were run from oldest to most recent.")
    is_ready: bool = Field(
        description="Whether the answer to the question is ready to be returned (i.e., by choosing 'none' as the next task).")


class NextTaskOutput(BaseModel):
    task: Task = Field(
        description="The next task that should be run in order to answer the question.")


class NextTaskPredict(Signature):
    """Chose the next task to run, in order to get the full answer. The next task should be chosen as to optimally answer the question given the data at hand. No specific order of the tasks is enforced. Tasks can be repeated."""
    input: NextTaskInput = InputField()
    output: NextTaskOutput = OutputField()


class IncrementalAnswerModule(
    Module,
    AnswerModule,
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
    _next_task_predict = TypedPredictor(signature=NextTaskPredict)

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

    @rate_limit(Limiter([
        Rate(limit=1, interval=Duration.SECOND),
        Rate(limit=500, interval=Duration.HOUR),
    ], raise_when_fail=False))
    def _next_task(
        self,
        builder: AnswerBuilder,
        history: Sequence[HistoryItem],
        **kwargs,
    ) -> Task:
        input = NextTaskInput(
            question=builder.question.body,
            question_type=builder.question.type,
            history=history,
            is_ready=builder.is_ready,
        )
        prediction: Prediction = self._next_task_predict.forward(
            input=input, **kwargs)
        output = cast(NextTaskOutput, prediction.output)
        return output.task

    def _run_next_task(
        self,
        builder: AnswerBuilder,
        history: Sequence[HistoryItem],
        **kwargs,
    ) -> HistoryItem | None:
        task = self._next_task(
            builder=builder,
            history=history,
        )
        if len(history) > 0 and task == history[-1].task:
            warn("Cannot run the same task twice in a row.")
            return HistoryItem(task=task, successful=False)
        elif task == "retrieve documents":
            print(
                f"Making documents for question '{builder.question.body[:100]}'...")
            builder.make_documents()
            return HistoryItem(task=task, successful=True)
        elif task == "retrieve snippets":
            print(
                f"Making snippets for question '{builder.question.body[:100]}'...")
            builder.make_snippets()
            return HistoryItem(task=task, successful=True)
        elif task == "generate exact answer":
            print(
                f"Making exact for question '{builder.question.body[:100]}'...")
            builder.make_exact_answer()
            return HistoryItem(task=task, successful=True)
        elif task == "generate summary answer":
            print(
                f"Making ideal for question '{builder.question.body[:100]}'...")
            builder.make_ideal_answer()
            return HistoryItem(task=task, successful=True)
        elif task == "none":
            if builder.is_ready:
                print("Returning answer...")
                return None
            else:
                warn("Attempted to return answer but answer was not ready yet.")
                return HistoryItem(task=task, successful=False)
        else:
            raise RuntimeError(f"Unknown task: {task}")

    def forward(self, question: Question, **kwargs) -> Answer:
        builder = AnswerBuilder(
            question=question,
            documents_module=self._documents_module,
            snippets_module=self._snippets_module,
            exact_answer_module=self._exact_answer_module,
            ideal_answer_module=self._ideal_answer_module,
        )
        history: list[HistoryItem] = []
        while not (builder.is_ready and
                   len(history) > 0 and
                   history[-1].task == "none"):
            history_item = self._run_next_task(
                builder=builder,
                history=history,
                **kwargs,
            )
            if history_item is None:
                break
            history.append(history_item)
        return builder.answer
