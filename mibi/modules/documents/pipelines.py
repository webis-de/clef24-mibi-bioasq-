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
from mibi.modules.snippets.pyterrier import SNIPPETS_COLS
from mibi.utils.elasticsearch import elasticsearch_connection
from mibi.utils.elasticsearch_pyterrier import ElasticsearchTransformer
from mibi.utils.pyterrier import ConditionalTransformer, ExportDocumentsTransformer


def _has_snippet_columns(topics_or_res: DataFrame) -> bool:
    return SNIPPETS_COLS.issubset(topics_or_res.columns)


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




def _build_query(row: dict[Hashable, Any]) -> Query:
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


def _build_result(article: Article) -> dict[Hashable, Any]:
    return {
        "title": article.title,
        "abstract": article.abstract,
        # TODO: Do we want the concatenated title and abstract or just the abstract here?
        "text": article.abstract,
        # "text": f"{article.title} {article.abstract}",
        "url": article.pubmed_url,
        "doi": article.doi,
        "doi_url": article.doi_url,
        "pmc_id": article.pmc_id,
        "pmc_url": article.pmc_url,
        "nlm_id": article.nlm_id,
        "author_full_names": [
            f"{author.forename} {author.lastname}"
            for author in article.authors
            if author.forename is not None and author.lastname is not None
        ],
        "author_orcids": [
            author.orcid
            for author in article.authors
            if author.orcid is not None
        ],
        "mesh_terms": [
            term.term
            for term in article.mesh_terms
        ],
        "publication_types": [
            term.term
            for term in article.publication_types
        ],
        "chemicals": [
            term.term
            for term in article.chemicals
        ],
        "keywords": list(article.keywords),
        "all_terms": [
            term.term
            for terms in (
                article.mesh_terms,
                article.publication_types,
                article.chemicals,
            )
            for term in terms
        ] + list(article.keywords),
        "publication_date": article.publication_date,
        "journal": article.journal,
        "journal_abbreviation": article.journal_abbreviation,
        "issn": article.issn,
        "country": article.country,
        "languages": article.languages,
    }


def _expand_query(row: Series) -> str:
    query = str(row["query"])

    if "exact_answer" in row.keys():
        exact_answer = str(row["exact_answer"])
        exact_answer = exact_answer.capitalize().removesuffix(".")
        query = f"{query} {exact_answer}."
        print(query)

    if "ideal_answer" in row.keys():
        ideal_answer = str(row["ideal_answer"])
        ideal_answer = ideal_answer.capitalize().removesuffix(".")
        query = f"{query} {ideal_answer}."
        print(query)

    return query

expand_query = query(_expand_query)


@dataclass(frozen=True)
class DocumentsPipeline(Transformer):
    elasticsearch_url: str
    elasticsearch_username: str | None
    elasticsearch_password: str | None
    elasticsearch_index: str | None

    @cached_property
    def _pipeline(self) -> Transformer:
        # If snippets are given, de-passage them.
        de_passager = MaxPassage()
        pipeline = ConditionalTransformer(
            condition=_has_snippet_columns,
            transformer_true=de_passager,
            transformer_false=Transformer.identity(),
        )

        # Expand the query with previous answers.
        pipeline = pipeline >> expand_query

        # Retrieve or re-rank documents with Elasticsearch (BM25).
        pipeline = pipeline >> ElasticsearchTransformer(
            document_type=Article,
            client=elasticsearch_connection(
                elasticsearch_url=self.elasticsearch_url,
                elasticsearch_username=self.elasticsearch_username,
                elasticsearch_password=self.elasticsearch_password,
            ),
            query_builder=_build_query,
            result_builder=_build_result,
            num_results=10,
            index=self.elasticsearch_index,
            verbose=True,
        )

        # TODO: Re-rank documents?

        # Cut off at 10 documents as per BioASQ requirements.
        pipeline = pipeline % 10  # type: ignore

        # FIXME: Export documents temporarily, to manually import them to the answer generation stage.
        pipeline = pipeline >> ExportDocumentsTransformer(
            path=PROJECT_DIR / "data" / "documents")

        return pipeline

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        return self._pipeline.transform(topics_or_res)
