from dataclasses import dataclass
from functools import cached_property
from pandas import DataFrame
from pyterrier.batchretrieve import TextScorer
from pyterrier.transformer import Transformer
from pyterrier.rewrite import tokenise, reset as reset_query
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
from pyterrier_dr import TasB, TctColBert, Ance

from mibi import PROJECT_DIR
from mibi.modules.documents.pipelines import DocumentsPipeline
from mibi.modules.snippets.pyterrier import SNIPPETS_COLS, PubMedSentencePassager
from mibi.utils.pyterrier import ConditionalTransformer, CachableTransformer, CutoffRerank, ExportSnippetsTransformer


def _has_snippet_columns(topics_or_res: DataFrame) -> bool:
    return SNIPPETS_COLS.issubset(topics_or_res.columns)


@dataclass(frozen=True)
class SnippetsPipeline(Transformer):
    elasticsearch_url: str
    elasticsearch_username: str | None
    elasticsearch_password: str | None
    elasticsearch_index: str | None
    bm25_rerank: bool = True
    # pointwise_model: str = "castorini/monot5-base-msmarco"  # monoT5
    pointwise_model: str = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"  # TAS-B
    # pointwise_model: str = "sentence-transformers/msmarco-distilbert-base-tas-b"  # TAS-B
    # pointwise_model: str = "castorini/tct_colbert-msmarco"  # TCT-ColBERT
    # pointwise_model: str = "castorini/tct_colbert-v2-msmarco"  # TCT-ColBERT
    # pointwise_model: str = "sentence-transformers/msmarco-roberta-base-ance-firstp"  # ANCE
    pairwise_model: str = "castorini/duot5-base-msmarco"  # duoT5

    @cached_property
    def _pipeline(self) -> Transformer:
        # If no snippets are given, run the documents pipeline and split passages.
        documents_pipeline = DocumentsPipeline(
            elasticsearch_url=self.elasticsearch_url,
            elasticsearch_username=self.elasticsearch_username,
            elasticsearch_password=self.elasticsearch_password,
            elasticsearch_index=self.elasticsearch_index,
        )
        passager = PubMedSentencePassager(max_sentences=3)
        pipeline = ConditionalTransformer(
            condition=_has_snippet_columns,
            transformer_true=Transformer.identity(),
            transformer_false=documents_pipeline >> passager,
        )

        if self.bm25_rerank:
            # Re-rank texts with BM25 (based on candidate set text statistics!)
            bm25_scorer = CachableTransformer(
                wrapped=TextScorer(
                    body_attr="text",
                    wmodel="BM25",
                ),
                key="TextScorer(body_attr='text',wmodel='BM25')"
            )
            pipeline = (pipeline >> tokenise() >> bm25_scorer >> reset_query())

        # Re-rank the top-100 snippets pointwise.
        pointwise_reranker: Transformer | None
        if "monot5" in self.pointwise_model:
            pointwise_reranker = MonoT5ReRanker(
                model=self.pointwise_model, verbose=True)
        elif ("tas-b" in self.pointwise_model or
              "tas_b" in self.pointwise_model):
            pointwise_reranker = TasB(
                model_name=self.pointwise_model, verbose=True)
        elif ("tct-colbert" in self.pointwise_model or
              "tct_colbert" in self.pointwise_model):
            pointwise_reranker = TctColBert(
                model_name=self.pointwise_model, verbose=True)
        elif "ance" in self.pointwise_model:
            pointwise_reranker = Ance(
                model_name=self.pointwise_model, verbose=True)
        else:
            pointwise_reranker = None
        if pointwise_reranker is not None:
            pipeline = CutoffRerank(
                candidates=pipeline,
                reranker=pointwise_reranker,
                cutoff=100,
            )

        # Re-re-rank top-5 snippets pairwise.
        # TODO: Choose pointwise re-ranker.
        pairwise_reranker: Transformer | None
        if "duot5" in self.pairwise_model:
            pairwise_reranker = DuoT5ReRanker(
                model=self.pairwise_model, verbose=True)
        else:
            pairwise_reranker = None
        if pairwise_reranker is not None:
            pipeline = CutoffRerank(
                candidates=pipeline,
                reranker=pairwise_reranker,
                cutoff=5,
            )

        # TODO: Axiomatically re-rank snippets.

        # Cut off at 10 snippets as per BioASQ requirements.
        pipeline = pipeline % 10  # type: ignore

        # FIXME: Export documents temporarily, to manually import them to the answer generation stage.
        pipeline = pipeline >> ExportSnippetsTransformer(
            path=PROJECT_DIR / "data" / "snippets")

        return pipeline
