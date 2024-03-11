from pathlib import Path
from click import UsageError, group, Path as PathParam, option, echo


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
    from elasticsearch7.helpers import streaming_bulk
    from elasticsearch7_dsl import Index
    from tqdm.auto import tqdm
    from mibi.modules.documents.pubmed import Article, PubMedBaseline

    elasticsearch_auth: tuple[str, str] | None
    if elasticsearch_username is not None and elasticsearch_password is None:
        raise UsageError("Must provide both username and password or none.")
    elif elasticsearch_password is not None and elasticsearch_username is None:
        raise UsageError("Must provide both password and username or none.")
    elif elasticsearch_username is not None and elasticsearch_password is not None:
        elasticsearch_auth = (elasticsearch_username, elasticsearch_password)
    else:
        elasticsearch_auth = None

    elasticsearch = Elasticsearch(
        hosts=elasticsearch_url,
        http_auth=elasticsearch_auth,
    )

    echo(
        f"Prepare PubMed baseline index "
        f"on Elasticsearch: {elasticsearch_index}")
    Article.init(index=elasticsearch_index, using=elasticsearch)
    Index(elasticsearch_index).settings(
        number_of_shards=3,
        number_of_replicas=2,
    )

    pubmed = PubMedBaseline(pubmed_baseline_path)

    echo(
        f"Indexing PubMed baseline from {pubmed_baseline_path} to Elasticsearch index {elasticsearch_index}...")
    actions = (
        {
            **article.to_dict(include_meta=True),
            "_index": elasticsearch_index,
        }
        for article in tqdm(pubmed)
    )
    results = streaming_bulk(elasticsearch, actions, max_retries=20)
    for ok, info in tqdm(
        results,
        desc="Index to Elasticsearch",
    ):
        if not ok:
            raise RuntimeError(f"Indexing error: {info}")
