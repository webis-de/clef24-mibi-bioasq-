from mibi.model import PartialAnswer, Question, Answer
from mibi.modules import DocumentsMaker, SnippetsMaker, ExactAnswerMaker, IdealAnswerMaker


class AnswerBuilder:
    _documents_maker: DocumentsMaker
    _snippets_maker: SnippetsMaker
    _exact_answer_maker: ExactAnswerMaker
    _ideal_answer_maker: IdealAnswerMaker
    _question: Question
    _partial_answer: PartialAnswer

    def __init__(
        self,
        question: Question,
        documents_maker: DocumentsMaker,
        snippets_maker: SnippetsMaker,
        exact_answer_maker: ExactAnswerMaker,
        ideal_answer_maker: IdealAnswerMaker,
    ) -> None:
        self.question = question
        self._partial_answer = PartialAnswer()
        self.documents_maker = documents_maker
        self.snippets_maker = snippets_maker
        self.exact_answer_maker = exact_answer_maker
        self.ideal_answer_maker = ideal_answer_maker

    def make_documents(self) -> None:
        self._partial_answer = PartialAnswer(
            documents=self.documents_maker(
                question=self.question,
                partial_answer=self._partial_answer,
            ),
            snippets=self._partial_answer.snippets,
            exact_answer=self._partial_answer.exact_answer,
            ideal_answer=self._partial_answer.ideal_answer,
        )

    def make_snippets(self) -> None:
        self._partial_answer = PartialAnswer(
            documents=self._partial_answer.documents,
            snippets=self.snippets_maker(
                question=self.question,
                partial_answer=self._partial_answer,
            ),
            exact_answer=self._partial_answer.exact_answer,
            ideal_answer=self._partial_answer.ideal_answer,
        )

    def make_exact_answer(self) -> None:
        self._partial_answer = PartialAnswer(
            documents=self._partial_answer.documents,
            snippets=self._partial_answer.snippets,
            exact_answer=self.exact_answer_maker(
                question=self.question,
                partial_answer=self._partial_answer,
            ),
            ideal_answer=self._partial_answer.ideal_answer,
        )

    def make_ideal_answer(self) -> None:
        self._partial_answer = PartialAnswer(
            documents=self._partial_answer.documents,
            snippets=self._partial_answer.snippets,
            exact_answer=self._partial_answer.exact_answer,
            ideal_answer=self.ideal_answer_maker(
                question=self.question,
                partial_answer=self._partial_answer,
            ),
        )

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
