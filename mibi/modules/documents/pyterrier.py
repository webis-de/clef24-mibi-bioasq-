from pandas import DataFrame
from pydantic_core import Url
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
                # Url(f"https://pubmed.ncbi.nlm.nih.gov/{document_id}")
                Url(f"http://www.ncbi.nlm.nih.gov/pubmed/{document_id}")
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
