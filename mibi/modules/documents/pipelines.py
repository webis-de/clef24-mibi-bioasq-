from dataclasses import dataclass
from functools import cached_property
from typing import Any, Hashable
from warnings import catch_warnings, simplefilter
from elasticsearch7_dsl.query import Query, Match, Exists, Nested, Bool, Terms
from pandas import DataFrame, Series
from pyterrier.transformer import Transformer
from pyterrier.text import MaxPassage
from pyterrier.apply import query
from spacy import load as spacy_load
from spacy.language import Language

from mibi import PROJECT_DIR
from mibi.modules.documents.pubmed import Article
from mibi.utils.elasticsearch import elasticsearch_connection
from mibi.utils.elasticsearch_pyterrier import ElasticsearchTransformer
from mibi.utils.pyterrier import ExportDocumentsTransformer, MaybeDePassager


_DISALLOWED_PUBLICATION_TYPES = [
    "Letter",
    "Comment",
    "Editorial",
    "News",
    "Biography",
    "Congress",
    "Video-Audio Media",
    "Interview",
    "Overall",
    "Retraction of Publication",
    "Retracted Publication",
    "Newspaper Article",
    "Bibliography",
    "Legal Case",
    "Directory",
    "Personal Narrative",
    "Address",
    "Randomized Controlled Trial, Veterinary",
    "Autobiography",
    "Dataset",
    "Clinical Trial, Veterinary",
    "Festschrift",
    "Webcast",
    "Observational Study, Veterinary",
    "Dictionary",
    "Periodical Index",
    "Interactive Tutorial",
]


def build_query(row: dict[Hashable, Any]) -> Query:
    query = str(row["query"])

    # TODO: Rank based on the query type.
    # query_type = str(row["query_type"])

    with catch_warnings():
        simplefilter(action="ignore", category=FutureWarning)
        language: Language = spacy_load("en_core_sci_sm")
    doc = language(query)

    query_stop_words_removed = " ".join(
        token.text
        for token in doc
        if not token.is_stop
    )

    es_query = Bool(
        filter=[
            # Only consider articles with an abstract.
            Exists(field="abstract"),
            # Remove certain publication types.
            ~Nested(
                path="publication_types",
                query=Terms(term=_DISALLOWED_PUBLICATION_TYPES),
            ),
        ],
        must=[
            Bool(
                must=[
                    # Require to at least match the title or abstract.
                    Bool(
                        should=[
                            Match(title=query_stop_words_removed),
                            Match(abstract=query_stop_words_removed),
                        ]
                    )
                ],
                should=[
                    # Prefer documents that also have matching MeSH terms.
                    Nested(
                        path="mesh_terms",
                        query=Bool(
                            should=[
                                Match(term=entity.text)
                                for entity in doc.ents
                            ]
                        ),
                    ),
                ]
            ),
        ],
    )
    return es_query


def build_result(article: Article) -> dict[Hashable, Any]:
    return {
        "title": article.title,
        "abstract": article.abstract,
        "text": f"{article.title} {article.abstract}",
        "url": article.pubmed_url,
    }


def _expand_query(row: Series) -> str:
    query = str(row["query"])

    if "exact_answer" in row.keys():
        exact_answer = str(row["exact_answer"])
        exact_answer = exact_answer.capitalize().removesuffix(".")
        query = f"{query} {exact_answer}."

    if "ideal_answer" in row.keys():
        ideal_answer = str(row["ideal_answer"])
        ideal_answer = ideal_answer.capitalize().removesuffix(".")
        query = f"{query} {ideal_answer}."

    return query


expand_query = query(_expand_query, verbose=True)


@dataclass(frozen=True)
class DocumentsPipeline(Transformer):
    elasticsearch_url: str
    elasticsearch_username: str | None
    elasticsearch_password: str | None
    elasticsearch_index: str | None

    @cached_property
    def _pipeline(self) -> Transformer:
        pipeline = Transformer.identity()

        # Expand the query with previous answers.
        pipeline = pipeline >> expand_query

        # Snippets need to be de-passaged, but oterwise skip de-passaging.
        de_passager = MaxPassage()
        de_passager = MaybeDePassager(de_passager)
        pipeline = pipeline >> de_passager

        # Retrieve or re-rank documents with Elasticsearch (BM25).
        elasticsearch = ElasticsearchTransformer(
            document_type=Article,
            client=elasticsearch_connection(
                elasticsearch_url=self.elasticsearch_url,
                elasticsearch_username=self.elasticsearch_username,
                elasticsearch_password=self.elasticsearch_password,
            ),
            query_builder=build_query,
            result_builder=build_result,
            num_results=10,
            index=self.elasticsearch_index,
            verbose=True,
        )
        pipeline = pipeline >> elasticsearch

        # Cut off at 10 documents as per BioASQ requirements.
        pipeline = pipeline % 10  # type: ignore

        # FIXME: Export documents temporarily, to manually import them to the answer generation stage.
        pipeline = pipeline >> ExportDocumentsTransformer(
            path=PROJECT_DIR / "data" / "documents")

        return pipeline

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return self._pipeline.transform(topics_or_res)
