from pandas import DataFrame
from pydantic_core import Url

from mibi.model import Snippet, Snippets
from mibi.modules import SnippetsModule
from mibi.utils.pyterrier import PyTerrierModule


class PyTerrierSnippetsModule(SnippetsModule, PyTerrierModule[Snippets]):
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
