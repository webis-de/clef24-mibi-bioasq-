from pathlib import Path
from typing import Iterable
from click import UsageError, argument, group, Path as PathParam, option


@group()
def index() -> None:
    pass


@index.command()
@option(
    "-p", "--pubmed-baseline", "pubmed_baseline_path",
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

    elasticsearch = Elasticsearch(
        hosts=elasticsearch_url,
        http_auth=elasticsearch_auth,
        request_timeout=60,
        read_timeout=60,
        max_retries=10,
    )
    articles: Iterable[Article] = PubMedBaseline(pubmed_baseline_path)
    indexer = ElasticsearchIndexer(
        document_type=Article,
        client=elasticsearch,
        index=elasticsearch_index,
        progress=True,
    )
    indexer.index_all(articles)


@index.command
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
@argument(
    "pubmed_ids_path",
    type=PathParam(
        path_type=Path,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        writable=False,
        resolve_path=True,
        allow_dash=False
    ),
)
@argument(
    "pubmed_docs_path",
    type=PathParam(
        path_type=Path,
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=False,
        writable=True,
        resolve_path=True,
        allow_dash=False
    ),
)
def export(pubmed_ids_path: Path, pubmed_docs_path: Path,
           elasticsearch_url: str,
           elasticsearch_username: str | None,
           elasticsearch_password: str | None,
           elasticsearch_index: str,) -> None:
    from mibi.modules.documents.pubmed import Article
    from csv import DictReader
    from json import dumps
    from more_itertools import chunked
    from tqdm import tqdm
    from elasticsearch7 import Elasticsearch
    from datetime import datetime

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
        request_timeout=60,
        read_timeout=60,
        max_retries=10,
    )
    with pubmed_ids_path.open("rt") as pubmed_ids_file:
        pubmed_ids: Iterable[str] = [
            row["pubmed_id"]
            for row in DictReader(pubmed_ids_file)
        ]
    pubmed_ids = tqdm(pubmed_ids)
    with pubmed_docs_path.open("wt") as pubmed_docs_file:
        pubmed_ids_chunks = chunked(pubmed_ids, 100)
        for pubmed_ids_chunk in pubmed_ids_chunks:
            articles = Article.mget(
                list(pubmed_ids_chunk), using=elasticsearch, index=elasticsearch_index)
            for article in articles:
                if article is None:
                    article_dict = None
                else:
                    article_dict = article.to_dict()
                    article_dict = {k: str(v) if isinstance(v, datetime) else v for k, v in article_dict.items()}
                pubmed_docs_file.write(f"{dumps(article_dict)}\n")
