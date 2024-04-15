from pickle import PicklingError, dumps  # nosec: B403

from pyterrier import started, init

from mibi.modules.build import build_answer_module
from mibi.utils.elasticsearch import elasticsearch_connection


def is_picklable(obj):
    try:
        dumps(obj)
    except PicklingError:
        return False
    except TypeError:
        return False
    return True


def test_es_piklable() -> None:
    elasticsearch = elasticsearch_connection(
        elasticsearch_url="https://example.com",
        elasticsearch_username=None,
        elasticsearch_password=None,
    )
    assert elasticsearch is not None
    assert not is_picklable(elasticsearch)


def test_es_documents_pipeline_piklable() -> None:
    if not started():
        init()
    from mibi.modules.documents.pipelines import DocumentsPipeline
    pipeline = DocumentsPipeline(
        elasticsearch_url="https://example.com",
        elasticsearch_username=None,
        elasticsearch_password=None,
        elasticsearch_index="example",
    )
    assert is_picklable(pipeline)


def test_es_documents_module_piklable() -> None:
    if not started():
        init()
    from pyterrier.transformer import Transformer
    from mibi.modules.documents.pyterrier import PyTerrierDocumentsModule
    pipeline = Transformer.identity()
    documents_module = PyTerrierDocumentsModule(pipeline)
    assert is_picklable(documents_module)


def test_es_snippets_pipeline_piklable() -> None:
    if not started():
        init()
    from mibi.modules.snippets.pipelines import SnippetsPipeline
    pipeline = SnippetsPipeline(
        elasticsearch_url="https://example.com",
        elasticsearch_username=None,
        elasticsearch_password=None,
        elasticsearch_index="example",
    )
    assert is_picklable(pipeline)


def test_es_snippets_module_piklable() -> None:
    if not started():
        init()
    from pyterrier.transformer import Transformer
    from mibi.modules.snippets.pyterrier import PyTerrierSnippetsModule
    pipeline = Transformer.identity()
    snippets_module = PyTerrierSnippetsModule(pipeline)
    assert is_picklable(snippets_module)


def test_answer_module_piklable() -> None:
    answer_module = build_answer_module(
        documents_module_type="mock",
        snippets_module_type="mock",
        exact_answer_module_type="mock",
        ideal_answer_module_type="mock",
        answer_module_type="standard",
        language_model_name="gpt-3.5-turbo",
        elasticsearch_url="https://example.com",
        elasticsearch_username=None,
        elasticsearch_password=None,
        elasticsearch_index="example",
    )
    assert is_picklable(answer_module)
