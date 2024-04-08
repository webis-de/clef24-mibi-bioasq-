from elasticsearch7 import Elasticsearch
from pyterrier.transformer import Transformer

from mibi import PROJECT_DIR
from mibi.modules.documents.pipelines import build_documents_pipeline
from mibi.modules.snippets.pyterrier import FallbackRetrieveSnippets
from mibi.utils.pyterrier import ExportSnippetsTransformer

def build_snippets_pipeline(
        elasticsearch: Elasticsearch,
        index: str,
) -> Transformer:
    pipeline = FallbackRetrieveSnippets(
        retrieve=build_documents_pipeline(
            elasticsearch=elasticsearch,
            index=index,
        ),
        max_sentences=3,
    )
    
    # TODO: Re-rank snippets

    # Cut off at 10 snippets as per BioASQ requirements.
    pipeline = pipeline % 10  # type: ignore

    # FIXME: Export documents temporarily, to manually import them to the answer generation stage.
    pipeline = pipeline >> ExportSnippetsTransformer(path=PROJECT_DIR / "data" / "snippets")

    return pipeline