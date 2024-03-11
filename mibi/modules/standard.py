from dataclasses import dataclass
from mibi.builder import AnswerBuilder
from mibi.model import Question, Answer
from mibi.modules import AnswerModule, DocumentsModule, SnippetsModule, ExactAnswerModule, IdealAnswerModule


@dataclass(frozen=True)
class StandardAnswerModule(AnswerModule):
    """
    Build the full answer by first retrieving documents, then snippets, then finding the exact answer, and finally the ideal answer.
    """

    documents_module: DocumentsModule
    snippets_module: SnippetsModule
    exact_answer_module: ExactAnswerModule
    ideal_answer_module: IdealAnswerModule

    def forward(self, question: Question) -> Answer:
        builder = AnswerBuilder(
            question=question,
            documents_module=self.documents_module,
            snippets_module=self.snippets_module,
            exact_answer_module=self.exact_answer_module,
            ideal_answer_module=self.ideal_answer_module,
        )
        builder.make_documents()
        builder.make_snippets()
        builder.make_exact_answer()
        builder.make_ideal_answer()
        return builder.answer
