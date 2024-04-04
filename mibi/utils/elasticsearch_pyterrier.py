from dataclasses import dataclass
from functools import cached_property
from itertools import islice
from typing import Any, Callable, Generic, Hashable, Iterable, Type, TypeVar

from elasticsearch7 import Elasticsearch
from elasticsearch7_dsl import Document
from elasticsearch7_dsl.query import Query
from elasticsearch7_dsl.response import Hit
from pandas import DataFrame, Series
from pyterrier.model import add_ranks
from pyterrier.transformer import Transformer
from tqdm.auto import tqdm


T = TypeVar("T", bound=Document)


@dataclass(frozen=True)
class ElasticsearchRetrieve(Generic[T], Transformer):
    document_type: Type[T]
    client: Elasticsearch
    query_builder: Callable[[str], Query]
    result_builder: Callable[[T], dict[Hashable, Any]]
    num_results: int = 10
    index: str | None = None
    verbose: bool = False

    def _merge_result(
            self,
            row: dict[Hashable, Any],
            hit: Hit
    ) -> dict[Hashable, Any]:
        result: T = self.document_type.from_es(hit)
        return {
            **row,
            "docno": hit.meta.id,
            "score": hit.meta.score,
            **self.result_builder(result),
        }

    def _transform_query(self, topic: DataFrame) -> DataFrame:
        if len(topic.index) != 1:
            raise RuntimeError("Can only transform one query at a time.")

        row: Series = topic.iloc[0]
        query = self.query_builder(row["query"])

        search = self.document_type.search(using=self.client, index=self.index)
        search.query(query)

        results: Iterable[Hit] = search.scan()
        results = islice(results, self.num_results)

        return DataFrame([
            self._merge_result(row.to_dict(), result)
            for result in results
        ])

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if not isinstance(topics_or_res, DataFrame):
            raise RuntimeError("Can only transform data frames.")
        if not {"qid", "query"}.issubset(topics_or_res.columns):
            raise RuntimeError("Needs qid and query columns.")
        if len(topics_or_res) == 0:
            return topics_or_res

        topics_by_query = topics_or_res.groupby(
            by=["qid", "query"],
            as_index=False,
            sort=False,
        )
        retrieved: DataFrame
        if self.verbose:
            tqdm.pandas(
                desc="Searching with Elasticsearch",
                unit="query",
            )
            retrieved = topics_by_query.progress_apply(
                self._transform_query
            )  # type: ignore
        else:
            retrieved = topics_by_query.apply(self._transform_query)

        retrieved = retrieved.reset_index(drop=True)
        retrieved.sort_values(by=["score"], ascending=False)
        retrieved = add_ranks(retrieved)

        return retrieved


@dataclass(frozen=True)
class ElasticsearchGet(Generic[T], Transformer):
    document_type: Type[T]
    client: Elasticsearch
    result_builder: Callable[[T], dict[Hashable, Any]]
    index: str | None = None

    def _merge_result(
            self,
            row: dict[Hashable, Any],
            hit: Any
    ) -> dict[Hashable, Any]:
        result: T = self.document_type.from_es(hit)
        return {
            **row,
            **self.result_builder(result),
        }

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if not isinstance(topics_or_res, DataFrame):
            raise RuntimeError("Can only transform data frames.")
        if "docno" not in topics_or_res.columns:
            raise RuntimeError("Needs docno column.")
        if len(topics_or_res) == 0:
            return topics_or_res

        ids = {str(id) for id in topics_or_res["docno"].to_list()}
        sorted_ids = sorted(ids)
        sorted_documents = self.document_type.mget(
            docs=sorted_ids,
            using=self.client,
            index=self.index,
        )

        documents = dict(zip(sorted_ids, sorted_documents))

        return DataFrame([
            self._merge_result(row.to_dict(), documents[row["docno"]])
            for _, row in topics_or_res.iterrows()
        ])


@dataclass(frozen=True)
class ElasticsearchRetrieveOrGet(Generic[T], Transformer):
    document_type: Type[T]
    client: Elasticsearch
    query_builder: Callable[[str], Query]
    result_builder: Callable[[T], dict[Hashable, Any]]
    num_results: int = 10
    index: str | None = None
    verbose: bool = False

    @cached_property
    def _retrieve(self) -> ElasticsearchRetrieve:
        return ElasticsearchRetrieve(
            document_type=self.document_type,
            client=self.client,
            query_builder=self.query_builder,
            result_builder=self.result_builder,
            num_results=self.num_results,
            index=self.index,
            verbose=self.verbose,
        )

    @cached_property
    def _get(self) -> ElasticsearchGet:
        return ElasticsearchGet(
            document_type=self.document_type,
            client=self.client,
            result_builder=self.result_builder,
            index=self.index,
        )

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if not isinstance(topics_or_res, DataFrame):
            raise RuntimeError("Can only transform data frames.")
        if "docno" in topics_or_res.columns:
            return self._get.transform(topics_or_res)
        if {"qid", "query"}.issubset(topics_or_res.columns):
            return self._retrieve.transform(topics_or_res)
        raise RuntimeError("Needs qid and query columns or docno column.")
