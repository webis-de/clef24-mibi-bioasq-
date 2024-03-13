from dataclasses import dataclass
from typing import Any, Generic, Iterable, Sized, Type, TypeVar, Iterator

from elasticsearch7 import Elasticsearch
from elasticsearch7.helpers import streaming_bulk
from elasticsearch7_dsl import Document
from tqdm.auto import tqdm


T = TypeVar("T", bound=Document)


@dataclass(frozen=True)
class ElasticsearchIndexer(Generic[T]):
    document_type: Type[T]
    client: Elasticsearch
    index: str | None = None
    progress: bool = False

    def _action(self, document: T) -> dict:
        if self.index is not None:
            setattr(document.meta, "index", self.index)
        return document.to_dict(include_meta=True)

    def iter_index(self, documents: Iterable[T]) -> Iterator[None]:
        total = len(documents) if isinstance(documents, Sized) else None

        print(f"Prepare Elasticsearch index: {self.index}")
        self.document_type.init(
            using=self.client,
            index=self.index,
        )

        print(f"Indexing to Elasticsearch index: {self.index}")
        actions = (
            self._action(document)
            for document in documents
        )
        results: Iterable[tuple[bool, Any]] = streaming_bulk(
            client=self.client,
            actions=actions,
            max_retries=20
        )
        if self.progress:
            results = tqdm(
                results,
                total=total,
                desc="Indexing",
                unit="doc",
            )
        for ok, info in results:
            if not ok:
                raise RuntimeError(f"Indexing error: {info}")
            yield

    def index_all(self, articles: Iterable[T]) -> None:
        for _ in self.iter_index(articles):
            pass
