from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from pandas import DataFrame, concat
from pyterrier.transformer import Transformer
from pyterrier.model import add_ranks

from mibi.model import Document, PartialAnswer, PartiallyAnsweredQuestion, Question, Snippet
from mibi.modules import ABCModule


_T = TypeVar("_T")


@dataclass(frozen=True)
class PyTerrierModule(Generic[_T], ABCModule):
    transformer: Transformer

    @staticmethod
    def _question_data(
        question: Question,
        partial_answer: PartialAnswer,
    ) -> dict[str, Any]:
        partially_answered_question = PartiallyAnsweredQuestion.from_question(
            question, partial_answer)

        data: dict[str, Any] = {
            "qid": question.id,
            "query": question.body,
            "query_type": question.type,
        }

        if partially_answered_question.exact_answer_text is not None:
            data["exact_answer"] = partially_answered_question.exact_answer_text
        if partially_answered_question.ideal_answer is not None:
            data["ideal_answer"] = partially_answered_question.ideal_answer

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
            "url": str(document),
            "score": 0,
        }

    @staticmethod
    def _snippet_data(
        snippet: Snippet,
    ) -> dict[str, Any]:
        document_data = PyTerrierModule._document_data(snippet.document)
        docno = document_data.pop("docno")
        return {
            **document_data,
            "docno": f"{docno}%p({snippet.begin_section},{snippet.offset_in_begin_section:d},{snippet.end_section},{snippet.offset_in_end_section:d})",
            "text": snippet.text,
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
            res = concat([
                res_snippets,
                res_documents,
            ])
            res = add_ranks(res)
        elif res_snippets is not None:
            res = res_snippets
            res = add_ranks(res)
        elif res_documents is not None:
            res = res_documents
            res = add_ranks(res)
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
        topics_or_res = pipeline.transform(topics_or_res)

        return topics_or_res


@dataclass(frozen=True)
class MaybeDePassager(Transformer):
    """
    De-passages only rows that are passages and keeps the others as is.
    Existing documents are preferred in case of duplicates.
    """

    de_passager: Transformer

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if "docno" not in topics_or_res.columns:
            return topics_or_res

        if topics_or_res["docno"].isna().any():
            raise RuntimeError("Empty docno found.")

        topics_or_res = topics_or_res.copy()

        is_passage = topics_or_res["docno"].str.contains("%p")
        topics_or_res = concat([
            topics_or_res[~is_passage],
            self.de_passager.transform(
                topics_or_res[is_passage].reset_index()),
        ])

        topics_or_res = topics_or_res.groupby("docno").first().reset_index()

        if "score" in topics_or_res.columns:
            topics_or_res.sort_values(
                by=["qid", "score"],
                ascending=[True, False],
                inplace=True,
            )
            topics_or_res = add_ranks(topics_or_res)

        topics_or_res.reset_index(drop=True, inplace=True)
        return topics_or_res


@dataclass(frozen=True)
class MaybePassager(Transformer):
    """
    Passages only rows that are not yet passages and keeps the others as is.
    Existing passages are preferred in case of duplicates.
    """

    passager: Transformer

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if "docno" not in topics_or_res.columns:
            return topics_or_res

        if topics_or_res["docno"].isna().any():
            raise RuntimeError("Empty docno found.")

        topics_or_res = topics_or_res.copy()

        is_passage = topics_or_res["docno"].str.contains("%p")
        topics_or_res = concat([
            topics_or_res[is_passage],
            self.passager.transform(
                topics_or_res[~is_passage].reset_index()),
        ])

        topics_or_res = topics_or_res.groupby("docno").first().reset_index()

        if "score" in topics_or_res.columns:
            topics_or_res.sort_values(
                by=["qid", "score"],
                ascending=[True, False],
                inplace=True,
            )
            topics_or_res = add_ranks(topics_or_res)

        topics_or_res.reset_index(drop=True, inplace=True)
        return topics_or_res


@dataclass(frozen=True)
class WithDocumentIds(Transformer):
    """
    Temporarily use document ID instead of passage ID.
    """

    transformer: Transformer

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        topics_or_res = topics_or_res.copy()

        topics_or_res["olddocno"] = topics_or_res["docno"]
        topics_or_res[["docno", "pid"]] = \
            topics_or_res["olddocno"].str.split("%p", expand=True)

        topics_or_res = self.transformer.transform(topics_or_res)

        topics_or_res["docno"] = topics_or_res["olddocno"]
        topics_or_res.drop(columns=["olddocno", "pid"], inplace=True)

        topics_or_res.reset_index(drop=True, inplace=True)
        return topics_or_res
