from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Sequence, TypeVar

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
        # TODO fold.
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
            "snippet_begin_section": snippet.begin_section,
            "snippet_offset_in_begin_section": snippet.offset_in_begin_section,
            "snippet_end_section": snippet.end_section,
            "snippet_offset_in_end_section": snippet.offset_in_end_section,
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
            res = res_snippets.merge(
                res_documents,
                on=["qid", "query", "query_type", "docno", "url"],
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


@dataclass(frozen=True)
class ExportDocumentsTransformer(Transformer):
    path: Path

    def __post_init__(self):
        self.path.mkdir(parents=True, exist_ok=True)

    def _export(self, res: DataFrame) -> None:
        return

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if not isinstance(topics_or_res, DataFrame):
            raise RuntimeError("Can only transform data frames.")
        if not {"qid", "query", "query_type", "docno", "score", "rank", "title", "abstract", "text", "url"}.issubset(topics_or_res.columns):
            raise RuntimeError(
                "Expected qid, query, query_type, docno, score, rank, title, abstract, text, and url columns.")

        for qid, res in topics_or_res.groupby(
            by="qid",
            as_index=False,
            sort=False,
        ):
            query_path = self.path / f"{qid}.csv"
            res.assign(
                questionno=res["qid"],
                question=res["query"],
                questiontype=res["query_type"],
            ).to_csv(query_path)

        return topics_or_res


@dataclass(frozen=True)
class ExportSnippetsTransformer(Transformer):
    path: Path

    def __post_init__(self):
        self.path.mkdir(parents=True, exist_ok=True)

    def _export(self, res: DataFrame) -> None:
        return

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if not isinstance(topics_or_res, DataFrame):
            raise RuntimeError("Can only transform data frames.")
        if not {"qid", "query", "query_type", "docno", "score", "rank", "text", "url", "snippet_begin_section", "snippet_offset_in_begin_section", "snippet_end_section", "snippet_offset_in_end_section"}.issubset(topics_or_res.columns):
            raise RuntimeError(
                "Expected qid, query, query_type, docno, score, rank, text, url, snippet_begin_section, snippet_offset_in_begin_section, snippet_end_section, and snippet_offset_in_end_section columns.")

        for qid, res in topics_or_res.groupby(
            by="qid",
            as_index=False,
            sort=False,
        ):
            query_path = self.path / f"{qid}.csv"
            res.assign(
                questionno=res["qid"],
                question=res["query"],
                questiontype=res["query_type"],
            ).to_csv(query_path)

        return topics_or_res


@dataclass(frozen=True)
class CachableTransformer(Transformer):
    wrapped: Transformer = field(repr=False)
    key: str

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return self.wrapped.transform(topics_or_res)


@dataclass(frozen=True)
class CutoffRerank(Transformer):
    candidates: Transformer
    reranker: Transformer
    cutoff: int

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        topics_or_res = self.candidates.transform(topics_or_res)
        pipeline = Transformer.from_df(
            input=topics_or_res,
            uniform=True,
        )
        pipeline = ((pipeline % self.cutoff) >> self.reranker) ^ pipeline
        return pipeline.transform(topics_or_res)


@dataclass(frozen=True)
class ConditionalTransformer(Transformer):
    condition: Callable[[DataFrame], bool]
    transformer_true: Transformer
    transformer_false: Transformer

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if self.condition(topics_or_res):
            return self.transformer_true.transform(topics_or_res)
        else:
            return self.transformer_false.transform(topics_or_res)
