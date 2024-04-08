from typing import Any, Hashable
from elasticsearch7 import Elasticsearch
from elasticsearch7_dsl.query import Query, Match, Exists, Nested, Bool
from pyterrier.transformer import Transformer
from spacy import load as spacy_load
from spacy.language import Language

from mibi import PROJECT_DIR
from mibi.modules.documents.pubmed import Article
from mibi.modules.documents.pyterrier import FoldSnippets
from mibi.utils.elasticsearch_pyterrier import ElasticsearchTransformer
from mibi.utils.pyterrier import ExportTransformer


def build_documents_pipeline(
        elasticsearch: Elasticsearch,
        index: str,
) -> Transformer:

    pipeline = FoldSnippets()

    def build_query(query: str) -> Query:
        language: Language = spacy_load("en_core_sci_sm")
        doc = language(query)

        entities = doc.ents
        entities_queries = [
            Match(term=entity.text)
            for entity in entities
        ]

        query_stop_words_removed = " ".join(
            token.text for token in doc if not token.is_stop)

        # TODO: Improve query.
        es_query = (
            Exists(field="abstract") &
            (
                (
                    Match(title=query_stop_words_removed) |
                    Match(abstract=query_stop_words_removed)
                ) |
                Nested(
                    path="mesh_terms",
                    query=Bool(should=entities_queries),
                )
            )
        )
        return es_query

    def build_result(article: Article) -> dict[Hashable, Any]:
        return {
            "title": article.title,
            "abstract": article.abstract,
            # TODO: Maybe we want the concatenated title and abstract here?
            "text": article.abstract,
            "url": article.pubmed_url,
            "doi": article.doi,
            "doi_url": article.doi_url,
            # TODO: Add more article fields.
        }

    pipeline = pipeline >> ElasticsearchTransformer(
        document_type=Article,
        client=elasticsearch,
        query_builder=build_query,
        result_builder=build_result,
        num_results=10,
        index=index,
        verbose=True,
    )

    # TODO: Re-rank documents.

    # FIXME: Export documents temporarily, to manually import them to the answer generation stage.
    pipeline = pipeline >> ExportTransformer(path=PROJECT_DIR / "data" / "documents")

    return pipeline


