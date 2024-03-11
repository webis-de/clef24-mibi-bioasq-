from dataclasses import dataclass
from mibi.model import PartialAnswer, Question, Answer
from mibi.modules import AnswerModule, DocumentsModule, SnippetsModule, ExactAnswerModule, IdealAnswerModule


@dataclass(frozen=True)
class IndependentAnswerModule(AnswerModule):
    """
    Build the full answer by independently retrieving documents, snippets, finding the exact answer, and the ideal answer. The result of neither task depends on one another.
    """

    documents_module: DocumentsModule
    snippets_module: SnippetsModule
    exact_answer_module: ExactAnswerModule
    ideal_answer_module: IdealAnswerModule

    def forward(self, question: Question) -> Answer:
        empty_answer = PartialAnswer()
        documents = self.documents_module.forward(question, empty_answer)
        snippets = self.snippets_module.forward(question, empty_answer)
        exact_answer = self.exact_answer_module.forward(question, empty_answer)
        ideal_answer = self.ideal_answer_module.forward(question, empty_answer)
        return Answer(
            documents=documents,
            snippets=snippets,
            exact_answer=exact_answer,
            ideal_answer=ideal_answer,
        )
