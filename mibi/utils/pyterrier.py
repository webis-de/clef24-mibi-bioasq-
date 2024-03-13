from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar
from pandas import DataFrame

from pyterrier.transformer import Transformer
from mibi.model import NOT_AVAILABLE, Document,  PartialAnswer, Question, Snippet


_T = TypeVar("_T")


@dataclass(frozen=True)
class PyTerrierModule(Generic[_T], ABC):
    transformer: Transformer

    @staticmethod
    def _question_data(
        question: Question,
        partial_answer: PartialAnswer,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "qid": question.id,
            "query": question.body,
            "query_type": question.type,
        }

        if partial_answer.exact_answer is not None:
            if partial_answer.exact_answer == NOT_AVAILABLE:
                data["exact_answer"] = None
            elif isinstance(partial_answer.exact_answer, str):
                data["exact_answer"] = partial_answer.exact_answer
            elif isinstance(partial_answer.exact_answer, Sequence):
                if len(partial_answer.exact_answer) == 0:
                    data["exact_answer"] = None
                if isinstance(partial_answer.exact_answer[0], str):
                    data["exact_answer"] = partial_answer.exact_answer[0]
                elif isinstance(partial_answer.exact_answer[0], Sequence):
                    data["exact_answer"] = ", ".join(
                        item[0]
                        for item in partial_answer.exact_answer
                    )
                else:
                    raise ValueError(
                        f"Unknown exact answer type: {partial_answer.exact_answer}")
            else:
                raise ValueError(
                    f"Unknown exact answer type: {partial_answer.exact_answer}")

        if partial_answer.ideal_answer is not None:
            data["ideal_answer"] = partial_answer.ideal_answer[0]

        return data

    @staticmethod
    def _document_data(
        document: Document,
    ) -> dict[str, Any]:
        document_url_path = document.path
        if document_url_path is None:
            raise RuntimeError(f"Invalid PubMed URL: {document_url_path}")
        document_id = document_url_path.split("/")[-1]
        return {
            "docno": document_id,
            "url": str(document)
        }

    @staticmethod
    def _snippet_data(
        snippet: Snippet,
    ) -> dict[str, Any]:
        document_data = PyTerrierModule._document_data(snippet.document)
        return {
            **document_data,
            "snippet_text": snippet.text,
        }

    @abstractmethod
    def parse(self, res: DataFrame) -> _T:
        raise NotImplementedError()

    def forward(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> _T:
        question_data = self._question_data(
            question=question,
            partial_answer=partial_answer,
        )

        res_snippets: DataFrame | None = None
        if partial_answer.snippets is not None:
            res_snippets = DataFrame([
                {
                    **question_data,
                    **self._snippet_data(snippet)
                }
                for snippet in partial_answer.snippets
            ])
        res_documents: DataFrame | None = None
        if partial_answer.documents is not None:
            res_documents = DataFrame([
                {
                    **question_data,
                    **self._document_data(document)
                }
                for document in partial_answer.documents
            ])

        res: DataFrame
        if res_snippets is not None and res_documents is not None:
            res = res_snippets.join(
                res_documents,
                on=["docno", "url"],
                how="outer",
            )
        elif res_snippets is not None:
            res = res_snippets
        elif res_documents is not None:
            res = res_documents
        else:
            res = DataFrame([question_data])
        res = self.transformer.transform(res)
        return self.parse(res)
