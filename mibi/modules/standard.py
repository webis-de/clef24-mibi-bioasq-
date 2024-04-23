from dataclasses import dataclass

from mibi.builder import AnswerBuilder
from mibi.model import Question, Answer
from mibi.modules import AnswerModule, DocumentsModule, SnippetsModule, ExactAnswerModule, IdealAnswerModule


@dataclass(frozen=True)
class _AnswerBuilderModule(AnswerModule):
    documents_module: DocumentsModule
    snippets_module: SnippetsModule
    exact_answer_module: ExactAnswerModule
    ideal_answer_module: IdealAnswerModule

    def builder(self, question: Question) -> AnswerBuilder:
        return AnswerBuilder(
            question=question,
            documents_module=self.documents_module,
            snippets_module=self.snippets_module,
            exact_answer_module=self.exact_answer_module,
            ideal_answer_module=self.ideal_answer_module,
        )


class RetrieveThenGenerateAnswerModule(_AnswerBuilderModule):
    """
    Build the full answer by first retrieving documents, then snippets, then finding the exact answer, and finally the ideal answer.
    """

    def forward(self, question: Question) -> Answer:
        builder = self.builder(question)
        builder.make_documents()
        builder.make_snippets()
        builder.make_exact_answer()
        builder.make_ideal_answer()
        return builder.answer


class GenerateThenRetrieveAnswerModule(_AnswerBuilderModule):
    """
    Build the full answer by first guessing the exact answer, then the ideal answer, then retrieving documents, and finally snippets.
    """

    def forward(self, question: Question) -> Answer:
        builder = self.builder(question)
        builder.make_exact_answer()
        builder.make_ideal_answer()
        builder.make_documents()
        builder.make_snippets()
        return builder.answer


class RetrieveThenGenerateThenRetrieveAnswerModule(_AnswerBuilderModule):
    """
    Build the full answer by first retrieving documents, then snippets, then finding the exact answer, and finally the ideal answer. The answers are used to again retrieve documents and snippets.
    """

    def forward(self, question: Question) -> Answer:
        builder = self.builder(question)
        builder.make_documents()
        builder.make_snippets()
        builder.make_exact_answer()
        builder.make_ideal_answer()
        builder.make_documents()
        builder.make_snippets()
        return builder.answer


class GenerateThenRetrieveThenGenerateAnswerModule(_AnswerBuilderModule):
    """
    Build the full answer by first guessing the exact answer, then the ideal answer, then retrieving documents, and finally snippets. The documents and snippets are used to refine the answers.
    """

    def forward(self, question: Question) -> Answer:
        builder = self.builder(question)
        builder.make_exact_answer()
        builder.make_ideal_answer()
        builder.make_documents()
        builder.make_snippets()
        builder.make_exact_answer()
        builder.make_ideal_answer()
        return builder.answer
