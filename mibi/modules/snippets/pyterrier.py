from dataclasses import dataclass
from itertools import chain
from typing import Iterator, NamedTuple

from more_itertools import sliding_window
from nltk import sent_tokenize
from nltk.downloader import Downloader
from pandas import DataFrame, Series, isna
from pydantic_core import Url
from pyterrier.transformer import Transformer

from mibi.model import Snippet, Snippets
from mibi.modules import SnippetsModule
from mibi.utils.pyterrier import PyTerrierModule


class PyTerrierSnippetsModule(PyTerrierModule[Snippets], SnippetsModule):
    def parse(self, res: DataFrame) -> Snippets:
        for col in (
            "text",
            "snippet_begin_section",
            "snippet_offset_in_begin_section",
            "snippet_end_section",
            "snippet_offset_in_end_section",
        ):
            if col not in res.columns:
                raise ValueError(
                    f"Cannot parse documents from results with columns: {res.columns}")
            if any(res[col].isna()):
                raise ValueError(
                    f"Cannot parse documents due to missing `{col}` values.")
        if "docno" in res.columns:
            if any(res["docno"].isna()):
                raise ValueError(
                    "Cannot parse documents due to missing `docno` values.")
            return [
                Snippet(
                    document=Url(
                        f"https://pubmed.ncbi.nlm.nih.gov/{row['docno']}"),
                    text=row["text"],
                    begin_section=row["snippet_begin_section"],
                    offset_in_begin_section=row["snippet_offset_in_begin_section"],
                    end_section=row["snippet_end_section"],
                    offset_in_end_section=row["snippet_offset_in_end_section"],
                )
                for _, row in res.iterrows()
            ]
        elif "url" in res.columns:
            if any(res["url"].isna()):
                raise ValueError(
                    "Cannot parse documents due to missing `url` values.")
            return [
                Snippet(
                    document=Url(row["url"]),
                    text=row["text"],
                    begin_section=row["snippet_begin_section"],
                    offset_in_begin_section=row["snippet_offset_in_begin_section"],
                    end_section=row["snippet_end_section"],
                    offset_in_end_section=row["snippet_offset_in_end_section"],
                )
                for _, row in res.iterrows()
            ]
        else:
            raise ValueError(
                f"Cannot parse documents from results with columns: {res.columns}")


_SNIPPETS_COLS = {
    "text",
    "snippet_begin_section",
    "snippet_offset_in_begin_section",
    "snippet_end_section",
    "snippet_offset_in_end_section",
}


class _Sentence(NamedTuple):
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class FallbackRetrieveSnippets(Transformer):
    retrieve: Transformer
    max_sentences: int

    def __post_init__(self):
        downloader = Downloader()
        if not downloader.is_installed("punkt"):
            downloader.download("punkt")

    def _iter_title_snippets(self, row: Series) -> Iterator[Snippet]:
        if isna(row["title"]):
            return
        url = Url(row["url"])
        title = row["title"]
        yield Snippet(
            document=url,
            text=title,
            begin_section="title",
            offset_in_begin_section=0,
            end_section="title",
            offset_in_end_section=len(title)
        )

    def _iter_abstract_snippets(self, row: Series) -> Iterator[Snippet]:
        if isna(row["abstract"]):
            return
        url = Url(row["url"])
        abstract: str = row["abstract"]

        sentences: list[_Sentence] = []
        last_offset = 0
        for sentence in sent_tokenize(
            text=abstract,
            language="english",
        ):
            start = abstract.find(sentence, last_offset)
            end = start + len(sentence)
            sentences.append(_Sentence(
                text=sentence,
                start=start,
                end=end,
            ))
            last_offset = end

        sentence_tuples = chain.from_iterable(
            sliding_window(sentences, n)
            for n in range(1, self.max_sentences + 1)
        )
        for sentence_tuple in sentence_tuples:
            yield Snippet(
                document=url,
                text=" ".join(
                    sentence.text
                    for sentence in sentence_tuple
                ),
                begin_section="abstract",
                offset_in_begin_section=min(
                    sentence.start
                    for sentence in sentence_tuple
                ),
                end_section="abstract",
                offset_in_end_section=max(
                    sentence.end
                    for sentence in sentence_tuple
                )
            )

    def _iter_snippets(self, row: Series) -> Iterator[Snippet]:
        yield from self._iter_title_snippets(row)
        yield from self._iter_abstract_snippets(row)

    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if _SNIPPETS_COLS.issubset(topics_or_res.columns):
            return topics_or_res

        topics_or_res = self.retrieve.transform(topics_or_res)
        return DataFrame([
            {
                **row,
                "text": snippet.text,
                "snippet_begin_section": snippet.begin_section,
                "snippet_offset_in_begin_section": snippet.offset_in_begin_section,
                "snippet_end_section": snippet.end_section,
                "snippet_offset_in_end_section": snippet.offset_in_end_section,
            }
            for _, row in topics_or_res.iterrows()
            for snippet in self._iter_snippets(row)
        ])
