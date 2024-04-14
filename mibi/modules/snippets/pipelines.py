from elasticsearch7 import Elasticsearch
from pandas import DataFrame
from pyterrier.batchretrieve import TextScorer
from pyterrier.transformer import Transformer
from pyterrier.rewrite import tokenise, reset as reset_query
from pyterrier_t5 import DuoT5ReRanker
# from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
from pyterrier_dr import TasB
# from pyterrier_dr import TasB, TctColBert, Ance

from mibi import PROJECT_DIR
from mibi.modules.documents.pipelines import build_documents_pipeline
from mibi.modules.snippets.pyterrier import SNIPPETS_COLS, PubMedSentencePassager
from mibi.utils.pyterrier import ConditionalTransformer, CachableTransformer, CutoffRerank, ExportSnippetsTransformer


def _has_snippet_columns(topics_or_res: DataFrame) -> bool:
    return SNIPPETS_COLS.issubset(topics_or_res.columns)


def build_snippets_pipeline(
        elasticsearch: Elasticsearch,
        index: str,
) -> Transformer:
    # If no snippets are given, run the documents pipeline and split passages.
    documents_pipeline = build_documents_pipeline(
        elasticsearch=elasticsearch,
        index=index,
    )
    passager = PubMedSentencePassager(max_sentences=3)    
    pipeline = ConditionalTransformer(
        condition=_has_snippet_columns,
        transformer_true=Transformer.identity(),
        transformer_false=documents_pipeline >> passager,
    )

    # Re-rank texts with BM25 (based on candidate set text statistics!)
    bm25 = CachableTransformer(
        wrapped=TextScorer(
            body_attr="text",
            wmodel="BM25",
        ),
        key="TextScorer(body_attr='text',wmodel='BM25')"
    )
    pipeline = (pipeline >> tokenise() >> bm25 >> reset_query())

    # Re-rank the top-100 snippets pointwise.
    # mono_t5 = MonoT5ReRanker(
    #     model="castorini/monot5-base-msmarco",
    #     verbose=True,
    # )
    tas_b = TasB(
        model_name="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
        verbose=True,
    )
    # tct_colbert = TctColBert(
    #     model_name="castorini/tct_colbert-msmarco",
    #     verbose=True,
    # )
    # ance = Ance(
    #     model_name = "sentence-transformers/msmarco-roberta-base-ance-firstp",
    #     verbose=True,
    # )
    # TODO: Choose re-ranker, maybe combine them linearly?
    pipeline = CutoffRerank(
        candidates=pipeline,
        # reranker=mono_t5,
        reranker=tas_b,
        # reranker=tct_colbert,
        # reranker=ance,
        cutoff=100,
    )

    # Re-re-rank top-5 snippets pairwise.
    duo_t5 = DuoT5ReRanker(
        model="castorini/duot5-base-msmarco",
        verbose=True,
    )
    pipeline = CutoffRerank(
        candidates=pipeline,
        reranker=duo_t5,
        cutoff=5,
    )

    # TODO: Axiomatically re-rank snippets.

    # Cut off at 10 snippets as per BioASQ requirements.
    pipeline = pipeline % 10  # type: ignore

    # FIXME: Export documents temporarily, to manually import them to the answer generation stage.
    pipeline = pipeline >> ExportSnippetsTransformer(
        path=PROJECT_DIR / "data" / "snippets")

    return pipeline
