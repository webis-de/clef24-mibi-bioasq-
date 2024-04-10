from typing import Any, Hashable
from elasticsearch7 import Elasticsearch
from elasticsearch7_dsl.query import Query, Match, Exists, Nested, Bool, Terms
from pyterrier.transformer import Transformer
from spacy import load as spacy_load
from spacy.language import Language

from mibi import PROJECT_DIR
from mibi.modules.documents.pubmed import Article
from mibi.modules.documents.pyterrier import FoldSnippets
from mibi.utils.elasticsearch_pyterrier import ElasticsearchTransformer
from mibi.utils.pyterrier import ExportDocumentsTransformer

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


def build_documents_pipeline(
        elasticsearch: Elasticsearch,
        index: str,
) -> Transformer:

    pipeline = FoldSnippets()

    def build_query(query: str) -> Query:
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

    pipeline = (pipeline >> ElasticsearchTransformer(
        document_type=Article,
        client=elasticsearch,
        query_builder=build_query,
        result_builder=build_result,
        num_results=10,
        index=index,
        verbose=True,
    ))

    # TODO: Re-rank documents.

    # Cut off at 10 documents as per BioASQ requirements.
    pipeline = pipeline % 10  # type: ignore

    # FIXME: Export documents temporarily, to manually import them to the answer generation stage.
    pipeline = pipeline >> ExportDocumentsTransformer(
        path=PROJECT_DIR / "data" / "documents")

    return pipeline
