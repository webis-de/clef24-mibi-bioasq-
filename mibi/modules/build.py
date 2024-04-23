from typing import Literal

from pyterrier import started, init

from mibi.modules import AnswerModule, DocumentsModule, ExactAnswerModule, IdealAnswerModule, SnippetsModule
from mibi.modules.exact_answer.llm import LlmExactAnswerModule
from mibi.modules.ideal_answer.llm import LlmIdealAnswerModule
from mibi.modules.incremental import IncrementalAnswerModule
from mibi.modules.independent import IndependentAnswerModule
from mibi.modules.mock import MockDocumentsModule, MockExactAnswerModule, MockIdealAnswerModule, MockSnippetsModule
from mibi.modules.standard import RetrieveThenGenerateAnswerModule, GenerateThenRetrieveAnswerModule, RetrieveThenGenerateThenRetrieveAnswerModule, GenerateThenRetrieveThenGenerateAnswerModule
from mibi.utils.language_models import init_language_model_clients


def build_answer_module(
    documents_module_type: Literal[
        "mock",
        "pyterrier",
    ],
    snippets_module_type: Literal[
        "mock",
        "pyterrier",
    ],
    exact_answer_module_type: Literal[
        "mock",
        "llm",
    ],
    ideal_answer_module_type: Literal[
        "mock",
        "llm",
    ],
    answer_module_type: Literal[
        "retrieve-then-generate", "rtg",
        "generate-then-retrieve", "gtr",
        "retrieve-then-generate-then-retrieve", "rtgtr",
        "generate-retrieve-then-generate", "gtrtg",
        "incremental",
        "independent",
    ],
    language_model_name: str,
    elasticsearch_url: str | None,
    elasticsearch_username: str | None,
    elasticsearch_password: str | None,
    elasticsearch_index: str | None,
) -> AnswerModule:
    print("Build answer module.")

    # Init language models.
    init_language_model_clients(language_model_name)

    # Create documents module.
    documents_module: DocumentsModule
    if documents_module_type == "mock":
        documents_module = MockDocumentsModule()
    elif documents_module_type == "pyterrier":
        if not started():
            init()
        from mibi.modules.documents.pipelines import DocumentsPipeline
        from mibi.modules.documents.pyterrier import PyTerrierDocumentsModule
        if elasticsearch_url is None or elasticsearch_index is None:
            raise ValueError("Must provide Elasticsearch URL and index.")
        pipeline = DocumentsPipeline(
            elasticsearch_url=elasticsearch_url,
            elasticsearch_username=elasticsearch_username,
            elasticsearch_password=elasticsearch_password,
            elasticsearch_index=elasticsearch_index,
        )
        documents_module = PyTerrierDocumentsModule(pipeline)
    else:
        raise ValueError("Unknown documents module type.")

    # Create snippets module.
    snippets_module: SnippetsModule
    if snippets_module_type == "mock":
        snippets_module = MockSnippetsModule()
    elif snippets_module_type == "pyterrier":
        if not started():
            init()
        from mibi.modules.snippets.pipelines import SnippetsPipeline
        from mibi.modules.snippets.pyterrier import PyTerrierSnippetsModule
        if elasticsearch_url is None or elasticsearch_index is None:
            raise ValueError("Must provide Elasticsearch URL and index.")
        pipeline = SnippetsPipeline(
            elasticsearch_url=elasticsearch_url,
            elasticsearch_username=elasticsearch_username,
            elasticsearch_password=elasticsearch_password,
            elasticsearch_index=elasticsearch_index,
        )
        snippets_module = PyTerrierSnippetsModule(pipeline)
    else:
        raise ValueError("Unknown snippets module type.")

    # Create exact answer module.
    exact_answer_module: ExactAnswerModule
    if exact_answer_module_type == "mock":
        exact_answer_module = MockExactAnswerModule()
    elif exact_answer_module_type == "llm":
        exact_answer_module = LlmExactAnswerModule()
    else:
        raise ValueError("Unknown exact answer module type.")

    # Create ideal answer module.
    ideal_answer_module: IdealAnswerModule
    if ideal_answer_module_type == "mock":
        ideal_answer_module = MockIdealAnswerModule()
    elif exact_answer_module_type == "llm":
        ideal_answer_module = LlmIdealAnswerModule()
    else:
        raise ValueError("Unknown ideal answer module type.")

    # Assemble full answer module.
    answer_module: AnswerModule
    if answer_module_type in ("retrieve-then-generate", "rtg"):
        answer_module = RetrieveThenGenerateAnswerModule(
            documents_module=documents_module,
            snippets_module=snippets_module,
            exact_answer_module=exact_answer_module,
            ideal_answer_module=ideal_answer_module,
        )
    elif answer_module_type in ("generate-then-retrieve", "gtr"):
        answer_module = GenerateThenRetrieveAnswerModule(
            documents_module=documents_module,
            snippets_module=snippets_module,
            exact_answer_module=exact_answer_module,
            ideal_answer_module=ideal_answer_module,
        )
    elif answer_module_type in ("retrieve-then-generate-then-retrieve", "rtgtr"):
        answer_module = RetrieveThenGenerateThenRetrieveAnswerModule(
            documents_module=documents_module,
            snippets_module=snippets_module,
            exact_answer_module=exact_answer_module,
            ideal_answer_module=ideal_answer_module,
        )
    elif answer_module_type in ("generate-retrieve-then-generate", "gtrtg"):
        answer_module = GenerateThenRetrieveThenGenerateAnswerModule(
            documents_module=documents_module,
            snippets_module=snippets_module,
            exact_answer_module=exact_answer_module,
            ideal_answer_module=ideal_answer_module,
        )
    elif answer_module_type == "incremental":
        answer_module = IncrementalAnswerModule(
            documents_module=documents_module,
            snippets_module=snippets_module,
            exact_answer_module=exact_answer_module,
            ideal_answer_module=ideal_answer_module,
        )
    elif answer_module_type == "independent":
        answer_module = IndependentAnswerModule(
            documents_module=documents_module,
            snippets_module=snippets_module,
            exact_answer_module=exact_answer_module,
            ideal_answer_module=ideal_answer_module,
        )
    else:
        raise ValueError("Unknown documents module type.")
    return answer_module
