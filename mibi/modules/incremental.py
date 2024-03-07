from enum import Enum
from warnings import warn
from dspy import Module, Signature, Predict, InputField, OutputField
from mibi.builder import AnswerBuilder
from mibi.model import Question, Answer
from mibi.modules import AnswerModule, DocumentsModule, SnippetsModule, ExactAnswerModule, IdealAnswerModule


class Task(Enum):
    make_documents = 'retrieve documents'
    make_snippets = 'retrieve snippets'
    make_exact_answer = 'generate exact answer'
    make_ideal_answer = 'generate summary answer'
    none = 'none'


class PredictNextTask(Signature):
    """
    Chose the next task to run, in order to get the full answer.
    """
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
    task: str = OutputField(
        desc=(
            f"Must always be one of "
            f"'{Task.make_documents.value}', "
            f"'{Task.make_snippets.value}', "
            f"'{Task.make_exact_answer.value}', "
            f"'{Task.make_ideal_answer.value}', or "
            f"'{Task.none.value}'."
        ),
        prefix="Next task:",
    )


# class IncrementalAnswerModule(AnswerModule, Module):
class IncrementalAnswerModule(Module):
    """
    Incrementally build the full answer by asking the LLM which
    part of the answer (i.e., documents, snippets, exact answer,
    or ideal answer) to make next.
    """

    _documents_module: DocumentsModule
    _snippets_module: SnippetsModule
    _exact_answer_module: ExactAnswerModule
    _ideal_answer_module: IdealAnswerModule
    _next_task = Predict(signature=PredictNextTask)

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

    def forward(self, question: Question) -> Answer:
        builder = AnswerBuilder(
            question=question,
            documents_module=self._documents_module,
            snippets_module=self._snippets_module,
            exact_answer_module=self._exact_answer_module,
            ideal_answer_module=self._ideal_answer_module,
        )
        next_task: str = Task.none.value
        while not builder.is_ready or next_task != Task.none.value:
            current_answer = builder.partial_answer
            next_task_prediction = self._next_task.forward(
                has_documents=(
                    "yes"
                    if current_answer.documents is not None
                    else "no"
                ),
                has_snippets=(
                    "yes"
                    if current_answer.snippets is not None
                    else "no"
                ),
                has_exact_answer=(
                    "yes"
                    if current_answer.exact_answer is not None
                    else "no"
                ),
                has_ideal_answer=(
                    "yes"
                    if current_answer.ideal_answer is not None
                    else "no"
                ),
            )
            next_task = next_task_prediction.task
            next_task = next_task.lower().strip()
            if next_task == Task.make_documents.value:
                print("Making documents...")
                builder.make_documents()
            elif next_task == Task.make_snippets.value:
                print("Making snippets...")
                builder.make_snippets()
            elif next_task == Task.make_exact_answer.value:
                print("Making exact answer...")
                builder.make_exact_answer()
            elif next_task == Task.make_ideal_answer.value:
                print("Making ideal answer...")
                builder.make_ideal_answer()
            elif next_task == Task.none.value:
                print("Returning answer...")
                return builder.answer
            else:
                raise RuntimeWarning(
                    f"Could not parse next task: {next_task!r}"
                )
        return builder.answer
