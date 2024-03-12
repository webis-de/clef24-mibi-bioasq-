from pathlib import Path
from click import UsageError, group, Path as PathParam, option


@group()
def index() -> None:
    pass


@index.command()
@option(
    "--p", "--pubmed-baseline", "pubmed_baseline_path",
    type=PathParam(
        path_type=Path,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        writable=False,
        resolve_path=True,
        allow_dash=False
    ),
    envvar="PUBMED_BASELINE_PATH",
)
@option(
    "--elasticsearch-url",
    type=str,
    envvar="ELASTICSEARCH_URL",
    required=True,
)
@option(
    "--elasticsearch-username",
    type=str,
    envvar="ELASTICSEARCH_USERNAME",
)
@option(
    "--elasticsearch-password",
    type=str,
    envvar="ELASTICSEARCH_PASSWORD",
)
@option(
    "--elasticsearch-index",
    type=str,
    envvar="ELASTICSEARCH_INDEX_PUBMED",
    required=True,
)
def pubmed(
    pubmed_baseline_path: Path,
    elasticsearch_url: str,
    elasticsearch_username: str | None,
    elasticsearch_password: str | None,
    elasticsearch_index: str,
) -> None:
    from elasticsearch7 import Elasticsearch
    from mibi.modules.documents.pubmed import Article, PubMedBaseline
    from mibi.utils.elasticsearch import ElasticsearchIndexer

    elasticsearch_auth: tuple[str, str] | None
    if elasticsearch_username is not None and elasticsearch_password is None:
        raise UsageError("Must provide both username and password or none.")
    elif elasticsearch_password is not None and elasticsearch_username is None:
        raise UsageError("Must provide both password and username or none.")
    elif elasticsearch_username is not None and elasticsearch_password is not None:
        elasticsearch_auth = (elasticsearch_username, elasticsearch_password)
    else:
        elasticsearch_auth = None

    articles = PubMedBaseline(
        directory=pubmed_baseline_path,
    )
    elasticsearch = Elasticsearch(
        hosts=elasticsearch_url,
        http_auth=elasticsearch_auth,
    )
    indexer = ElasticsearchIndexer(
        document_type=Article,
        client=elasticsearch,
        index=elasticsearch_index,
        progress=False,
    )
    indexer.index_all(articles)
