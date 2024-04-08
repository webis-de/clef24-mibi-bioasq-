from pandas import DataFrame
from pydantic_core import Url
from pyterrier.transformer import Transformer

from mibi.model import Documents
from mibi.modules import DocumentsModule
from mibi.utils.pyterrier import PyTerrierModule


class PyTerrierDocumentsModule(PyTerrierModule[Documents], DocumentsModule):
    def parse(self, res: DataFrame) -> Documents:
        if "docno" in res.columns:
            if any(res["docno"].isna()):
                raise ValueError(
                    "Cannot parse documents due to missing `docno` values.")
            return [
                Url(f"https://pubmed.ncbi.nlm.nih.gov/{document_id}")
                for document_id in res["docno"]
            ]
        elif "url" in res.columns:
            if any(res["url"].isna()):
                raise ValueError(
                    "Cannot parse documents due to missing `url` values.")
            return [
                Url(url)
                for url in res["url"]
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


class FoldSnippets(Transformer):
    def transform(self, topics_or_res: DataFrame) -> DataFrame:
        if _SNIPPETS_COLS.issubset(topics_or_res.columns):
            topics_or_res = topics_or_res.groupby(
                by=list(_SNIPPETS_COLS),
                sort=False,
            ).first().reset_index(drop=True)
        return topics_or_res
