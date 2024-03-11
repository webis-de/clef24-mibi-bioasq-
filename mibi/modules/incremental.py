from enum import Enum
from typing import Callable, ParamSpec, TypeVar
from warnings import warn
from dspy import Module, Signature, Predict, InputField, OutputField
from pydantic import BaseModel
from pyrate_limiter.limiter import Limiter
from pyrate_limiter import Rate, Duration
from mibi.builder import AnswerBuilder
from mibi.model import Question, Answer
from mibi.modules import DocumentsModule, SnippetsModule, ExactAnswerModule, IdealAnswerModule


class Task(Enum):
    make_documents = 'retrieve documents'
    make_snippets = 'retrieve snippets'
    make_exact_answer = 'generate exact answer'
    make_ideal_answer = 'generate summary answer'
    none = 'none'


class YesNo(Enum):
    yes = 'yes'
    no = 'no'


class PredictNextTaskInputs(BaseModel):
    question: str = InputField(
        prefix="Question:",
        desc="The question that should be answered."
    )
    question_type: str = InputField(
        prefix="Question type:",
        desc="How the question should be answered."
    )
    has_documents: str = InputField(
        prefix="Has retrieved documents:",
        desc="If documents were already retrieved for the question."
    )
    has_snippets: str = InputField(
        prefix="Has retrieved snippets:",
        desc="If snippets were already retrieved for the question."
    )
    has_exact_answer: str = InputField(
        prefix="Has generated exact answer:",
        desc="If an exact answer was already generated for the question."
    )
    has_ideal_answer: str = InputField(
        prefix="Has generated summary answer:",
        desc="If a human summary answer was already generated for the question."
    )
    last_task: str = InputField(
        prefix="Last task:",
        desc=(
            f"The last task that was run. One of "
            f"'{Task.make_documents.value}', "
            f"'{Task.make_snippets.value}', "
            f"'{Task.make_exact_answer.value}', "
            f"'{Task.make_ideal_answer.value}', or "
            f"'{Task.none.value}'."
        ),
    )
    last_task_successful: str = InputField(
        prefix="Last task successful:",
        desc=(
            f"'{YesNo.yes}' if the last task was successful, "
            f"'{YesNo.no}' otherwise."
        )
    )


class PredictNextTaskOutputs(BaseModel):
    task: str = OutputField(
        prefix="Next task:",
        desc=(
            f"Must be one of "
            f"'{Task.make_ideal_answer.value}', "
            f"'{Task.make_exact_answer.value}', "
            f"'{Task.make_snippets.value}', "
            f"'{Task.make_documents.value}', or "
            f"'{Task.none.value}', but the order is arbitrary. "
            f"The task '{Task.none.value}' is only valid once all other tasks were run at least once."
        ),
    )


class PredictNextTask(PredictNextTaskInputs, PredictNextTaskOutputs, Signature):
    f"""
    Chose the next task to run, in order to get the full answer. The next task should be chosen as to optimally answer the question given the data at hand. For example, the next task could be to hypothesize an answer ('{Task.make_exact_answer}') or to retrieve a first set of documents ('{Task.make_documents}'). Tasks can be repeated.
    """


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


def _to_yes_no(flag: bool) -> YesNo:
    return YesNo.yes if flag else YesNo.no


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
    _predict_next_task = Predict(signature=PredictNextTask)

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
        predict_next_task_inputs = PredictNextTaskInputs(
            question=builder.question.body,
            question_type=builder.question.type,
            has_documents=_to_yes_no(builder.has_documents).value,
            has_snippets=_to_yes_no(builder.has_snippets).value,
            has_exact_answer=_to_yes_no(builder.has_exact_answer).value,
            has_ideal_answer=_to_yes_no(builder.has_ideal_answer).value,
            last_task=last_task.value,
            last_task_successful=_to_yes_no(last_task_successful).value,
        )
        predict_next_task_outputs: PredictNextTaskOutputs = \
            self._predict_next_task.forward(
                **predict_next_task_inputs.model_dump())  # type: ignore
        next_task = predict_next_task_outputs.task
        next_task = next_task.lower().strip()
        if next_task == Task.make_documents.value:
            return Task.make_documents
        elif next_task == Task.make_snippets.value:
            return Task.make_snippets
        elif next_task == Task.make_exact_answer.value:
            return Task.make_exact_answer
        elif next_task == Task.make_ideal_answer.value:
            return Task.make_ideal_answer
        elif next_task == Task.none.value:
            return Task.none
        else:
            raise ValueError(f"Could not parse next task: {next_task!r}")

    def forward(self, question: Question) -> Answer:
        builder = AnswerBuilder(
            question=question,
            documents_module=self._documents_module,
            snippets_module=self._snippets_module,
            exact_answer_module=self._exact_answer_module,
            ideal_answer_module=self._ideal_answer_module,
        )
        next_task: Task = Task.none
        last_task: Task = Task.none
        last_task_successful: bool = True
        while not builder.is_ready or next_task != Task.none:
            next_task = self._next_task(
                builder=builder,
                last_task=last_task,
                last_task_successful=last_task_successful,
            )
            if next_task == last_task:
                warn("Cannot run the same task twice in a row.")
                last_task_successful = False
            elif next_task == Task.make_documents:
                print(
                    f"Making documents for question '{builder.question.body[:100]}'...")
                builder.make_documents()
                last_task_successful = True
            elif next_task == Task.make_snippets:
                print(
                    f"Making snippets for question '{builder.question.body[:100]}'...")
                builder.make_snippets()
                last_task_successful = True
            elif next_task == Task.make_exact_answer:
                print(
                    f"Making exact for question '{builder.question.body[:100]}'...")
                builder.make_exact_answer()
                last_task_successful = True
            elif next_task == Task.make_ideal_answer:
                print(
                    f"Making ideal for question '{builder.question.body[:100]}'...")
                builder.make_ideal_answer()
                last_task_successful = True
            elif next_task == Task.none:
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
  #
