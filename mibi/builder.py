from mibi.model import PartialAnswer, Question, Answer, PartiallyAnsweredQuestion
from mibi.modules import DocumentsModule, SnippetsModule, ExactAnswerModule, IdealAnswerModule


class AnswerBuilder:
    _documents_module: DocumentsModule
    _snippets_module: SnippetsModule
    _exact_answer_module: ExactAnswerModule
    _ideal_answer_module: IdealAnswerModule
    _question: Question
    _partial_answer: PartialAnswer

    def __init__(
        self,
        question: Question,
        documents_module: DocumentsModule,
        snippets_module: SnippetsModule,
        exact_answer_module: ExactAnswerModule,
        ideal_answer_module: IdealAnswerModule,
    ) -> None:
        self._question = question
        if isinstance(question, PartiallyAnsweredQuestion):
            self._partial_answer = question
        else:
            self._partial_answer = PartialAnswer()
        self._documents_module = documents_module
        self._snippets_module = snippets_module
        self._exact_answer_module = exact_answer_module
        self._ideal_answer_module = ideal_answer_module

    def make_documents(self) -> None:
        print(f"Making documents for question '{self.question.body}'...")
        self._partial_answer = PartialAnswer(
            documents=self._documents_module.forward(
                question=self._question,
                partial_answer=self._partial_answer,
            ),
            snippets=self._partial_answer.snippets,
            exact_answer=self._partial_answer.exact_answer,
            ideal_answer=self._partial_answer.ideal_answer,
        )
        print("Made documents.")

    def make_snippets(self) -> None:
        print(f"Making snippets for question '{self.question.body}'...")
        self._partial_answer = PartialAnswer(
            documents=self._partial_answer.documents,
            snippets=self._snippets_module.forward(
                question=self._question,
                partial_answer=self._partial_answer,
            ),
            exact_answer=self._partial_answer.exact_answer,
            ideal_answer=self._partial_answer.ideal_answer,
        )
        print("Made snippets.")

    def make_exact_answer(self) -> None:
        print(f"Making exact answer for question '{self.question.body}'...")
        self._partial_answer = PartialAnswer(
            documents=self._partial_answer.documents,
            snippets=self._partial_answer.snippets,
            exact_answer=self._exact_answer_module.forward(
                question=self._question,
                partial_answer=self._partial_answer,
            ),
            ideal_answer=self._partial_answer.ideal_answer,
        )
        print("Made exact answer.")

    def make_ideal_answer(self) -> None:
        print(f"Making ideal answer for question '{self.question.body}'...")
        self._partial_answer = PartialAnswer(
            documents=self._partial_answer.documents,
            snippets=self._partial_answer.snippets,
            exact_answer=self._partial_answer.exact_answer,
            ideal_answer=self._ideal_answer_module.forward(
                question=self._question,
                partial_answer=self._partial_answer,
            ),
        )
        print("Made ideal answer.")

    @property
    def question(self) -> Question:
        return self._question

    @property
    def partial_answer(self) -> PartialAnswer:
        return self._partial_answer
    
    @property
    def has_documents(self) -> bool:
        return self._partial_answer.documents is not None
    
    @property
    def has_snippets(self) -> bool:
        return self._partial_answer.snippets is not None
    
    @property
    def has_exact_answer(self) -> bool:
        return self._partial_answer.exact_answer is not None
    
    @property
    def has_ideal_answer(self) -> bool:
        return self._partial_answer.ideal_answer is not None

    @property
    def is_ready(self) -> bool:
        if self._partial_answer.documents is None:
            return False
        if self._partial_answer.snippets is None:
            return False
        if self._partial_answer.exact_answer is None:
            return False
        if self._partial_answer.ideal_answer is None:
            return False
        return True

    @property
    def answer(self) -> Answer:
        if self._partial_answer.documents is None:
            raise RuntimeError("Answer not yet known.")
        if self._partial_answer.snippets is None:
            raise RuntimeError("Answer not yet known.")
        if self._partial_answer.exact_answer is None:
            raise RuntimeError("Answer not yet known.")
        if self._partial_answer.ideal_answer is None:
            raise RuntimeError("Answer not yet known.")
        return Answer(
            documents=self._partial_answer.documents,
            snippets=self._partial_answer.snippets,
            exact_answer=self._partial_answer.exact_answer,
            ideal_answer=self._partial_answer.ideal_answer,
        )
