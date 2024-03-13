from dataclasses import dataclass
from datetime import datetime
from functools import cache
from itertools import chain
from pathlib import Path
from typing import Iterator, Iterable, Sized

from elasticsearch7 import Elasticsearch
from elasticsearch7_dsl import Document, Date, Text, Keyword, InnerDoc, Nested
from elasticsearch7_dsl.aggs import Terms
from lxml.etree import parse  # nosec: B410
from pubmed_parser import parse_medline_xml
from tqdm.auto import tqdm


class Author(InnerDoc):
    lastname: str | None = Text(
        fields={
            "keyword": Keyword()
        }
    )  # type: ignore
    forename: str | None = Text(
        fields={
            "keyword": Keyword()
        }
    )  # type: ignore
    initials: str | None = Keyword()  # type: ignore
    orcid: str | None = Keyword()  # type: ignore
    affiliation: str | None = Text(
        fields={
            "keyword": Keyword()
        }
    )  # type: ignore

    @property
    def pmc_id_url(self) -> str | None:
        if self.pmc_id is None:
            return None
        return f"https://ncbi.nlm.nih.gov/pmc/articles/{self.pmc_id}"


class MeshTerm(InnerDoc):
    mesh_id: str = Keyword()  # type: ignore
    """MeSH ID"""
    term: str = Keyword()  # type: ignore
    qualifiers: list[str] = Keyword()  # type: ignore


class Article(Document):
    class Index:
        settings = {
            "number_of_shards": 3,
            "number_of_replicas": 2,
        }

    pubmed_id: str = Keyword(required=True)  # type: ignore
    """PubMed ID"""
    pmc_id: str | None = Keyword()  # type: ignore
    """PubMed Central ID"""
    doi: str | None = Keyword()  # type: ignore
    """DOI"""
    other_ids: list[str] = Keyword(multi=True)  # type: ignore
    """Other IDs of the same article."""
    title: str | None = Text()  # type: ignore
    """Title of the article."""
    abstract: str | None = Text()  # type: ignore
    """Abstract of the article."""
    authors: list[Author] = Nested(Author)  # type: ignore
    mesh_terms: list[MeshTerm] = Nested(MeshTerm)  # type: ignore
    """List of MeSH terms."""
    publication_types: list[MeshTerm] = Nested(MeshTerm)  # type: ignore
    """List of publication types."""
    keywords: list[str] = Keyword(multi=True)  # type: ignore
    """List of keywords."""
    chemicals: list[MeshTerm] = Nested(MeshTerm)  # type: ignore
    """List of chemical terms."""
    publication_date: datetime = Date(
        default_timezone="UTC", required=True)  # type: ignore
    """Publication date."""
    journal: str | None = Text(
        fields={
            "keyword": Keyword()
        }
    )  # type: ignore
    """Journal of the article."""
    journal_abbreviation: str | None = Keyword()  # type: ignore
    """Journal abbreviation."""
    nlm_id: str = Keyword(required=True)  # type: ignore
    """NLM unique identification."""
    issn: str | None = Keyword()  # type: ignore
    """ISSN of the journal."""
    country: str | None = Keyword()  # type: ignore
    """Country of the journal."""
    references_pubmed_ids: list[str] = Keyword(multi=True)  # type: ignore
    """PubMed IDs of references made to the article."""
    languages: list[str] = Keyword(multi=True)  # type: ignore
    """List of languages."""
    source_file: str = Keyword(required=True)  # type: ignore
    """Basename of the XML file that contains this article."""

    @property
    def pubmed_id_url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pubmed_id}"

    @property
    def pmc_id_url(self) -> str | None:
        if self.pmc_id is None:
            return None
        return f"https://ncbi.nlm.nih.gov/pmc/articles/{self.pmc_id}"

    @property
    def doi_url(self) -> str | None:
        if self.doi is None:
            return None
        return f"https://doi.org/{self.doi}"

    @staticmethod
    def _parse_required(value: str) -> str:
        if len(value) == 0:
            raise RuntimeError("String must not be empty.")
        return value

    @staticmethod
    def _parse_optional(value: str) -> str | None:
        if len(value) == 0:
            return None
        return value

    @staticmethod
    def _parse_list(value: str) -> list[str]:
        if len(value) == 0:
            return []
        return [item.strip() for item in value.split("; ")]

    @staticmethod
    def _parse_orcid(value: str) -> str | None:
        if len(value) == 0:
            return None
        for prefix in (
            "http://orcid.org/",
            "https://orcid.org/",
            "http//:www.orcid.org/",
            "https//:www.orcid.org/",
            "%20",
            "s",
        ):
            if value.startswith(prefix):
                value = value.removeprefix(prefix)
                break
        if len(value) == 19 and value.count("-") == 3:
            return value
        if len(value) == 18 and value.count("-") == 3:
            return f"{value}X"
        if len(value) == 16 and (
            value.isnumeric() or (
                value[:15].isnumeric() and value[15] == "X")):
            return f"{value[0:4]}-{value[4:8]}-{value[8:12]}-{value[12:16]}"
        raise RuntimeError(f"Could not parse author identifier: {value}")

    @staticmethod
    def _parse_authors(values: list[dict[str, str]]) -> list[Author]:
        return [
            Author(
                lastname=Article._parse_optional(author["lastname"]),
                forename=Article._parse_optional(author["forename"]),
                initials=Article._parse_optional(author["initials"]),
                orcid=Article._parse_orcid(author["identifier"]),
                affiliation=Article._parse_optional(author["affiliation"]),
            )
            for author in values
        ]

    @staticmethod
    def _parse_mesh_terms(value: str) -> list[MeshTerm]:
        if len(value) == 0:
            return []
        mesh_terms_split: Iterable[list[str]] = (
            mesh_term.strip().split(":", maxsplit=1)
            for mesh_term in value.split("; ")
        )
        mesh_terms: list[MeshTerm] = []
        for mesh_id_term in mesh_terms_split:
            if len(mesh_id_term) == 1 and len(mesh_terms) > 0:
                mesh_terms[-1].qualifiers.append(mesh_id_term[0])
            else:
                mesh_id, term = mesh_id_term
                mesh_terms.append(MeshTerm(
                    mesh_id=mesh_id.strip(),
                    term=term.strip(),
                    qualifiers=[],
                ))
        return mesh_terms

    @staticmethod
    def _parse_date(value: str) -> datetime:
        if value.count("-") == 0:
            return datetime.strptime(value, "%Y")
        elif value.count("-") == 1:
            return datetime.strptime(value, "%Y-%m")
        elif value.count("-") == 2:
            return datetime.strptime(value, "%Y-%m-%d")
        else:
            raise RuntimeError(f"Unsupported date format: {value}")

    @classmethod
    def parse(cls, article: dict, path: Path) -> "Article | None":
        if article["delete"]:
            return None
        pubmed_id = cls._parse_required(article["pmid"])
        pmc_id = cls._parse_optional(article["pmc"])
        doi = cls._parse_optional(article["doi"])
        other_ids = cls._parse_list(article["other_id"])
        title = cls._parse_optional(article["title"])
        abstract = cls._parse_optional(article["abstract"])
        authors = cls._parse_authors(article["authors"])
        mesh_terms: list[MeshTerm] = cls._parse_mesh_terms(
            article["mesh_terms"])
        publication_types: list[MeshTerm] = cls._parse_mesh_terms(
            article["publication_types"])
        keywords = cls._parse_list(article["keywords"])
        chemicals: list[MeshTerm] = cls._parse_mesh_terms(
            article["chemical_list"])
        publication_date = cls._parse_date(article["pubdate"])
        journal = cls._parse_optional(article["journal"])
        journal_abbreviation = cls._parse_optional(article["medline_ta"])
        nlm_id = cls._parse_required(article["nlm_unique_id"])
        issn = cls._parse_optional(article["issn_linking"])
        country = cls._parse_optional(article["country"])
        references_pubmed_ids = cls._parse_list(article.get("reference", ""))
        languages = cls._parse_list(article.get("languages", ""))
        return Article(
            meta={"id": pubmed_id},
            pubmed_id=pubmed_id,
            pmc_id=pmc_id,
            doi=doi,
            other_ids=other_ids,
            title=title,
            abstract=abstract,
            authors=authors,
            mesh_terms=mesh_terms,
            publication_types=publication_types,
            keywords=keywords,
            chemicals=chemicals,
            publication_date=publication_date,
            journal=journal,
            journal_abbreviation=journal_abbreviation,
            nlm_id=nlm_id,
            issn=issn,
            country=country,
            references_pubmed_ids=references_pubmed_ids,
            languages=languages,
            source_file=path.name,
        )


@dataclass(frozen=True)
class PubMedBaseline(Iterable[Article], Sized):
    directory: Path

    def __post_init__(self):
        if not self.directory.is_dir():
            raise RuntimeError(
                f"Cannot read PubMed baseline from: {self.directory}")

    def _pmids(self, path: Path) -> Iterator[str]:
        print(f"Find PubMed IDs in: {path}")
        root = parse(path, parser=None)  # nosec: B320
        pmid_results = chain(
            root.iterfind(".//PubmedArticle/MedlineCitation/PMID"),
            root.iterfind(
                ".//PubmedArticle/PubmedData/ArticleIdList/ArticleId[@IdType=\"pmid\"]"),
        )
        for pmid in pmid_results:
            pmid_text = pmid.text
            if pmid_text is not None:
                yield pmid_text

    @cache
    def _count(self, path: Path) -> int:
        return sum(1 for _ in self._pmids(path))

    @staticmethod
    def _parse_articles(path: Path) -> Iterator[Article]:
        articles = parse_medline_xml(
            path=str(path),
            year_info_only=False,
            nlm_category=False,
            author_list=True,
            reference_list=False,
        )
        for article in tqdm(
            articles,
            desc=f"Parse {path}",
            unit="article",
        ):
            parsed = Article.parse(
                article=article,
                path=path,
            )
            if parsed is None:
                continue
            yield parsed

    def _paths(self) -> list[Path]:
        return list(self.directory.glob("pubmed*n*.xml.gz"))

    def __iter__(self) -> Iterator[Article]:
        for path in self._paths():
            yield from self._parse_articles(path)

    def __len__(self) -> int:
        return sum(
            self._count(path)
            for path in self._paths()
        )


@dataclass(frozen=True)
class PubMedBaselineNonIndexedElasticsearch(PubMedBaseline):
    client: Elasticsearch
    index: str | None = None

    def _count_unique(self, path: Path) -> int:
        return len(set(self._pmids(path)))

    @cache
    def _paths(self) -> list[Path]:
        paths = super()._paths()
        if len(paths) == 0:
            return paths

        print(
            f"Fetching indexed counts from Elasticsearch index: {self.index}")
        source_file_search = Article.search(
            using=self.client,
            index=self.index,
        )
        source_file_search.aggs.bucket(
            name="source_file",
            agg_type=Terms(field="source_file")
        )
        source_file_buckets = source_file_search.execute()\
            .aggs.source_file.buckets
        source_file_counts: dict[str, int] = {
            str(bucket.key): int(str(bucket.doc_count))
            for bucket in source_file_buckets
        }

        return [
            path
            for path in paths
            if source_file_counts.get(path.name, 0) < self._count_unique(path)
        ]
