from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Iterable

from elasticsearch7_dsl import Document, Date, Text, Keyword, InnerDoc, Nested
from pubmed_parser import parse_medline_xml
from tqdm.auto import tqdm


class Author(InnerDoc):
    lastname: str = Text(
        fields={
            "keyword": Keyword()
        }
    )  # type: ignore
    forename: str = Text(
        fields={
            "keyword": Keyword()
        }
    )  # type: ignore
    initials: str = Keyword()  # type: ignore
    identifier: str = Keyword()  # type: ignore
    affiliation: str = Text(
        fields={
            "keyword": Keyword()
        }
    )  # type: ignore


class MeshTerm(InnerDoc):
    mesh_id: str = Keyword()  # type: ignore
    """MeSH ID"""
    term: str = Keyword()  # type: ignore
    qualifiers: list[str] = Keyword()  # type: ignore


class Article(Document):
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


@dataclass(frozen=True)
class PubMedBaseline(Iterable[Article]):
    directory: Path

    def __post_init__(self):
        if not self.directory.is_dir():
            raise RuntimeError(
                f"Cannot read PubMed baseline from: {self.directory}")

    @staticmethod
    def _parse_abstracts(path: Path) -> Iterator[Article]:
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
            if article["delete"]:
                continue
            pubmed_id: str = article["pmid"]
            if len(pubmed_id) == 0:
                raise RuntimeError("Empty PubMed ID.")
            _pmc_id: str = article["pmc"]
            pmc_id: str | None = _pmc_id
            if len(_pmc_id) == 0:
                pmc_id = None
            _doi: str = article["doi"]
            doi: str | None = _doi
            if len(_doi) == 0:
                doi = None
            _other_ids: str = article["other_id"]
            other_ids: list[str]
            if len(_other_ids) == 0:
                other_ids = []
            else:
                other_ids = [
                    other_id.strip()
                    for other_id in _other_ids.split("; ")
                ]
            _title: str = article["title"]
            title: str | None = _title
            if len(_title) == 0:
                title = None
            _abstract: str = article["abstract"]
            abstract: str | None = _abstract
            if len(_abstract) == 0:
                abstract = None
            _authors: list[dict[str, str]] = article["authors"]
            authors: list[Author] = [
                Author(
                    lastname=author["lastname"],
                    forename=author["forename"],
                    initials=author["initials"],
                    identifier=author["identifier"],
                    affiliation=author["affiliation"],
                )
                for author in _authors
            ]
            _mesh_terms: str = article["mesh_terms"]
            mesh_terms: list[MeshTerm]
            if len(_mesh_terms) == 0:
                mesh_terms = []
            else:
                _mesh_terms_split: Iterator[list[str]] = (
                    mesh_term.strip().split(":", maxsplit=1)
                    for mesh_term in _mesh_terms.split("; ")
                )
                mesh_terms = [
                    MeshTerm(
                        mesh_id=mesh_id.strip(),
                        term=term.strip(),
                    )
                    for mesh_id, term in _mesh_terms_split
                ]
            _publication_types: str = article["publication_types"]
            publication_types: list[MeshTerm]
            if len(_publication_types) == 0:
                publication_types = []
            else:
                _publication_types_split: Iterator[list[str]] = (
                    publication_type.strip().split(":", maxsplit=1)
                    for publication_type in _publication_types.split("; ")
                )
                publication_types = [
                    MeshTerm(
                        mesh_id=mesh_id.strip(),
                        term=term.strip(),
                    )
                    for mesh_id, term in _publication_types_split
                ]
            _keywords: str = article["keywords"].strip()
            keywords: list[str]
            if len(_keywords) == 0:
                keywords = []
            else:
                keywords = [
                    keyword.strip()
                    for keyword in _keywords.split("; ")
                ]
            _chemicals: str = article["chemical_list"]
            chemicals: list[MeshTerm]
            if len(_chemicals) == 0:
                chemicals = []
            else:
                _chemicals_split: Iterator[list[str]] = (
                    chemical.strip().split(":", maxsplit=1)
                    for chemical in _chemicals.split("; ")
                )
                chemicals = [
                    MeshTerm(
                        mesh_id=mesh_id.strip(),
                        term=term.strip(),
                    )
                    for mesh_id, term in _chemicals_split
                ]
            _publication_date: str = article["pubdate"]
            publication_date: datetime
            if _publication_date.count("-") == 0:
                publication_date = datetime.strptime(
                    _publication_date, "%Y")
            elif _publication_date.count("-") == 1:
                publication_date = datetime.strptime(
                    _publication_date, "%Y-%m")
            elif _publication_date.count("-") == 2:
                publication_date = datetime.strptime(
                    _publication_date, "%Y-%m-%d")
            else:
                raise RuntimeError(
                    f"Unsupported date format: {_publication_date}")
            _journal: str = article["journal"]
            journal: str | None = _journal
            if len(_journal) == 0:
                journal = None
            _journal_abbreviation: str = article["medline_ta"]
            journal_abbreviation: str | None = _journal_abbreviation
            if len(_journal_abbreviation) == 0:
                journal_abbreviation = None
            nlm_id: str = article["nlm_unique_id"]
            if len(nlm_id) == 0:
                raise RuntimeError("Empty NLM ID.")
            _issn: str = article["issn_linking"]
            issn: str | None = _issn
            if len(_issn) == 0:
                issn = None
            _country: str = article["country"]
            country: str | None = _country
            if len(_country) == 0:
                country = None
            _references_pubmed_ids: str = article.get("reference", "")
            references_pubmed_ids: list[str]
            if len(_references_pubmed_ids) == 0:
                references_pubmed_ids = []
            else:
                references_pubmed_ids = [
                    references_pubmed_id.strip()
                    for references_pubmed_id in _references_pubmed_ids.split("; ")
                ]
            _languages: str = article.get("languages", "")
            languages: list[str]
            if len(_languages) == 0:
                languages = []
            else:
                languages = [
                    language.strip()
                    for language in _languages.split("; ")
                ]
            yield Article(
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
            )

    def __iter__(self) -> Iterator[Article]:
        paths = list(self.directory.glob("pubmed*n*.xml.gz"))
        for path in tqdm(
            paths,
            desc="Parse PubMed files",
            unit="file",
        ):
            yield from self._parse_abstracts(path)
