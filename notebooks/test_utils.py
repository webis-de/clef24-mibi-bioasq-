import os
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from typing import List, Literal
import time

from test_templates import *


import nltk
from nltk.corpus import stopwords
import string

# nltk.download("punkt")
# nltk.download("stopwords")


def remove_stopwords_and_punctuation(text):
    words = nltk.word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    punctuation = set(string.punctuation)
    filtered_words = [
        word
        for word in words
        if word.lower() not in stop_words and word not in punctuation
    ]
    filtered_text = " ".join(filtered_words)
    return filtered_text


api_key = os.getenv("OPENAI_API_KEY")
client = instructor.patch(OpenAI(api_key=api_key))


class Snippets(BaseModel):
    title_sentences: List = Field(
        ...,
        description="ONLY sentences or phrases from title that directly answer the given question, or empty list if no answer present",
    )
    abstract_sentences: List = Field(
        ...,
        description="ONLY sentences or phrases from abstract that directly answer the question, or empty list if no answer present",
    )
    score: float = Field(
        ..., description="how confident are you: score between 0 and 1"
    )
    chain_of_thought: str = Field(
        ...,
        description="Think step by step to make a good decision. Are there extracted sentences that directly answer the question?",
    )


class ExtractedSnippets(BaseModel):
    snippets: Snippets


def get_snippets(question: str, title: str, abstract: str) -> ExtractedSnippets:
    submission = SNIPPET_TEMPLATE.format(
        question=question, title=title, abstract=abstract
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "system",
                "content": "You are a world class system to extract relevant sentences from titles and abstracts answering questions",
            },
            {
                "role": "user",
                "content": "Extract from the title and abstract ONLY sentences of phrases that directly answer the question",
            },
            {
                "role": "user",
                "content": submission,
            },
        ],
        temperature=0,
        response_model=Snippets,
        # max_tokens=10000,
    )
    return resp.title_sentences, resp.abstract_sentences


import logging as logger

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from xml.etree.ElementTree import fromstring, Element

from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy
from pyterrier.transformer import Transformer
from requests import get
from tqdm.auto import tqdm

from mibi.modules import DocumentsModule, Question


@dataclass
class PubMedApiRetrieve(DocumentsModule):
    name = "PubMedApiRetrieve"

    eutils_api_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    esearch_api_url = f"{eutils_api_base_url}/esearch.fcgi"
    efetch_api_url = f"{eutils_api_base_url}/efetch.fcgi"

    # query_field: str = "query"
    num_results: Optional[int] = 10
    verbose: bool = field(repr=False, default=False)

    def _transform_query(self, question: Question) -> DataFrame:
        # if len(topic.index) != 1:
        #     raise RuntimeError("Can only transform one query at a time.")

        # row: Dict[str, Any] = topic.to_dict(orient="records")[0]

        query: str = question.body
        query = query.lower()
        query = query.replace(" ", "+")

        search_response = get(
            f"{self.esearch_api_url}?"
            f"db=pubmed&term={query}&retmax={self.num_results}"
        )
        search_root = fromstring(search_response.text)
        search_id_list = search_root.find("IdList")
        search_ids: list[str] = [
            element.text.strip() for element in search_id_list.findall("Id")
        ]
        logger.debug(f"Found {len(search_ids)} articles for '{query}'.")

        assert len(search_ids) <= self.num_results

        search_ids_string = ",".join(search_ids)
        fetch_response = get(
            f"{self.efetch_api_url}?" f"db=pubmed&id={search_ids_string}&retmode=xml"
        )
        fetch_root = fromstring(fetch_response.text)
        fetch_articles: list[Element]
        if fetch_root.tag == "PubmedArticleSet":
            fetch_articles = [
                (
                    element.find("BookDocument").find("Book")
                    if element.find("BookDocument") is not None
                    else element.find("MedlineCitation").find("Article")
                )
                for element in fetch_root
            ]
        elif fetch_root.tag == "eFetchResult":
            fetch_articles = []
        else:
            raise Exception(f"Unexpected root tag '{fetch_root.tag}'.")
        logger.debug(
            f"Found {len(fetch_articles)} article texts "
            f"for ids '{search_ids_string}'."
        )

        assert len(search_ids) == len(fetch_articles)

        results: list[dict[str, Any]] = []
        for i, (doc_id, article) in enumerate(zip(search_ids, fetch_articles)):
            title_text = (
                article.find("BookTitle").text
                if article.find("BookTitle") is not None
                else article.find("ArticleTitle").text
            )
            title = title_text.strip() if title_text is not None else ""
            abstract_element = article.find("Abstract")
            abstract_texts = (
                (text.text for text in abstract_element.findall("AbstractText"))
                if abstract_element is not None
                else []
            )
            abstract = " ".join(
                [text.strip() for text in abstract_texts if text is not None]
            )
            results.append(
                {
                    # **row,
                    "docno": doc_id,
                    "score": len(results) - i,
                    "rank": i + 1,
                    "title": title,
                    "text": abstract,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{doc_id}/",
                }
            )
        time.sleep(1)
        return DataFrame(results)

    def transform(self, questions: List[Question]) -> DataFrame:
        # if not {"qid", "query"}.issubset(topics.columns):
        #     raise RuntimeError("Needs qid and query columns.")

        # if len(questions) == 0:
        #     return self._transform_query(topics)

        # topics_by_query: DataFrameGroupBy = topics.groupby(
        #     by=["qid"],
        #     as_index=False,
        #     sort=False,
        # )
        # if self.verbose:
        #     # Show progress during reranking queries.
        #     tqdm.pandas(
        #         desc="Searching with PubMed API",
        #         unit="query",
        #     )
        #     topics_by_query = topics_by_query.progress_apply(self._transform_query)
        # else:
        #     topics_by_query = topics_by_query.apply(self._transform_query)

        # retrieved: DataFrame = topics_by_query.reset_index(drop=True)
        for question in questions:
            retrieved = self._transform_query(question)
        return retrieved


def retrieve_bm25(query, corpus):
    x = [
        " ".join(d[0] + d[1]) for d in zip(corpus.title.tolist(), corpus.text.tolist())
    ]
    tokenized_corpus = [
        remove_stopwords_and_punctuation(x).split() for x in x
    ]  # corpus["text"].tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = remove_stopwords_and_punctuation(query.body).lower().split(" ")
    return bm25.get_scores(tokenized_query)


def rerank_biencoder(question, retrieved):
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    # embedder = SentenceTransformer("tavakolih/all-MiniLM-L6-v2-pubmed-full")
    # x = [
    #     " ".join(d[0] + d[1])
    #     for d in zip(retrieved.title.tolist(), retrieved.text.tolist())
    # ]
    x = retrieved.text.tolist()
    corpus_embeddings = embedder.encode(x, convert_to_tensor=False)
    query_embedding = embedder.encode(question.body, convert_to_tensor=False)
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    # combined = zip(cos_scores, x)
    # sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
    # _, reranked = zip(*sorted_combined)
    return cos_scores


def rerank_crossencoder(question, retrieved):
    encoder = CrossEncoder("cross-encoder/msmarco-MiniLM-L6-en-de-v1", max_length=512)
    # encoder = CrossEncoder("sentence-transformers/all-MiniLM-L6-v2")
    # encoder = CrossEncoder('cross-encoder/msmarco-MiniLM-L12-en-de-v1', max_length=512)
    # encoder = CrossEncoder("tavakolih/all-MiniLM-L6-v2-pubmed-full")
    # texts = [
    #     " ".join(d[0] + d[1])
    #     for d in zip(retrieved.title.tolist(), retrieved.text.tolist())
    # ]
    texts = retrieved.text.tolist()
    scores = encoder.predict([(question.body, text) for text in texts])
    # combined = zip(scores, post)
    # sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
    # _, reranked = zip(*sorted_combined)
    return scores


def get_offset(snippets, text):
    offsets = []
    for snippet in snippets:
        if snippet:
            try:
                offset = re.search(snippet, text).span()
            except Exception as e:
                offset = ()
        offsets.append(offset)
    return offsets


class YesNoExact(BaseModel):
    answer: List[Literal["yes", "no"]]


class ListExact(BaseModel):
    answer: List[List[str]] = Field(
        ...,
        description="List of lists with single entities, each inner list contains only one entity, maximum 100 inner lists, no synonyms, sorted by the relevance to the question: first is the most relevant",
    )


class FactoidExact(BaseModel):
    answer: List[str] = Field(
        ...,
        description="List of lists with single entities or short phrases, each inner list contains only one short fact, maximum 5 inner lists, no synonyms, sorted by the relevance to the question: first is the most relevant",
    )


class Summary(BaseModel):
    answer: List[str] = Field(
        ...,
        description="A summary of the retrieved context in 2 or 3 sentences. It also contains a short explanation",
    )


class IdealAnswer(BaseModel):
    answer: str = Field(
        ...,
        description="The ideal answer to the question in one longer sentence that also contains a short explanation. The ideal answer is grammatically complete with subjects, objects, and predicates, is concise and precise.",
    )


def response_exact_answer(query: str, q_type: str, text_chunks: str):

    text_chunks = text_chunks.split()[:1000]
    text_chunks = " ".join(text_chunks)

    response_model_dict = {
        "yesno": YesNoExact,
        "list": ListExact,
        "factoid": FactoidExact,
        "summary": Summary,
    }

    response_model = response_model_dict[q_type]

    try:
        messages = [
            {
                "role": "system",
                "content": "You are a medical doctor answering real-world medical entrance exam questions.",
            },
            {
                "role": "system",
                "content": "Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following question.",
            },
            {
                "role": "system",
                "content": "Base your answer on the current and standard practices referenced in medical guidelines. Please use as much as possible the retrieved context given below if it is factually correct. You must nor start the answer with 'based on the provided context'",
            },
            {"role": "system", "content": f"Context: {text_chunks}"},
            # {"role": "system", 'content': f'Please write the answer as {answer_type}'},
            {"role": "user", "content": f"Question: {query}"},
        ]

        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0613",
            model="gpt-4-0613",
            temperature=0,
            response_model=response_model,
            messages=messages,
            max_tokens=1750,
        )

        if q_type == "summary":
            return [" ".join(response.answer)]
        else:
            return response.answer
    except:
        print("Trying another way (exact answer)")
        response_mode_dict = {
            "yesno": "yes or no",
            "list": "List of lists with single entities, each inner list contains only one entity, maximum 100 inner lists, no synonyms, sorted by the relevance to the question: first is the most relevant",
            "factoid": "List of lists with single entities or short phrases, each inner list contains a very short fact, maximum 5 inner lists, no synonyms, sorted by the relevance to the question: first is the most relevant",
            "summary": "A summary of the retrieved context in 2 or 3 sentences. It also contains a short explanation",
        }

        response_mode = response_mode_dict[q_type]
        messages = [
            {
                "role": "system",
                "content": "You are a medical doctor answering real-world medical entrance exam questions.",
            },
            {
                "role": "system",
                "content": "Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following question.",
            },
            {
                "role": "system",
                "content": "Base your answer on the current and standard practices referenced in medical guidelines. Please use as much as possible the retrieved context given below if it is factually correct",
            },
            {
                "role": "system",
                "content": f"The answer must be formatted as {response_mode}",
            },
            {"role": "system", "content": f"Context: {text_chunks}"},
            # {"role": "system", 'content': f'Please write the answer as {answer_type}'},
            {"role": "user", "content": f"Question: {query}"},
        ]

        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0613",
            model="gpt-4-0613",
            temperature=0,
            # response_model=response_model,
            messages=messages,
            max_tokens=1750,
        )

        return [response.choices[0].message.content]


def response_ideal_answer(query: str, q_type: str, text_chunks: str):

    text_chunks = text_chunks.split()[:1000]
    text_chunks = " ".join(text_chunks)

    if q_type == "summary":
        response_model = Summary
    else:
        response_model = IdealAnswer
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a medical doctor answering real-world medical entrance exam questions.",
            },
            {
                "role": "system",
                "content": "Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following question.",
            },
            {
                "role": "system",
                "content": "Base your answer on the current and standard practices referenced in medical guidelines. Please use as much as possible the retrieved context given below if it is factually correct",
            },
            {"role": "system", "content": f"Context: {text_chunks}"},
            {"role": "user", "content": f"Question: {query}"},
        ]

        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0613",
            model="gpt-4-0613",
            temperature=0,
            response_model=response_model,
            messages=messages,
            max_tokens=1750,
        )

        if q_type == "summary":
            return [" ".join(response.answer)]
        else:
            return [response.answer]

    except:
        print("Trying another way")
        messages = [
            {
                "role": "system",
                "content": "You are a medical doctor answering real-world medical entrance exam questions.",
            },
            {
                "role": "system",
                "content": "Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, answer the following question.",
            },
            {
                "role": "system",
                "content": "Base your answer on the current and standard practices referenced in medical guidelines. Please use as much as possible the retrieved context given below if it is factually correct",
            },
            {
                "role": "system",
                "content": "The ideal answer to the question in one longer sentence that also contains a short explanation. The ideal answer is grammatically complete with subjects, objects, and predicates, is concise and precise.",
            },
            {"role": "system", "content": f"Context: {text_chunks}"},
            {"role": "user", "content": f"Question: {query}"},
        ]

        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0613",
            model="gpt-4-0613",
            temperature=0,
            # response_model=response_model,
            messages=messages,
            max_tokens=1750,
        )
        # print(response.choices[0].message.content)

        if q_type == "summary":
            return [" ".join(response.choices[0].message.content)]
        else:
            return [response.choices[0].message.content]


def response_exact_answer_mistral(question, q_type, text_chunks, model):

    response_model_dict = {
        "yesno": ANSWER_TEMPLATE_EXACT_YESNO,
        "list": ListExact,
        "factoid": FactoidExact,
        "summary": Summary,
    }

    TEMPLATE = response_model_dict[q_type]
    submission = TEMPLATE.format(question=question, document=text_chunks)

    resp = client_blablador.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": submission,
            },
        ],
        temperature=0,
        max_tokens=100,
    )
    resp = resp.choices[0].message.content
    return resp


def response_ideal_answer_mistral(question, q_type, text_chunks, model):

    response_model_dict = {
        "yesno": ANSWER_TEMPLATE_IDEAL_YESNO,
        "list": ListExact,
        "factoid": FactoidExact,
        "summary": Summary,
    }

    TEMPLATE = response_model_dict[q_type]
    submission = TEMPLATE.format(question=question, document=text_chunks)

    resp = client_blablador.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": submission,
            },
        ],
        temperature=0,
        max_tokens=1000,
    )
    resp = resp.choices[0].message.content
    return resp


def flat_list(l):
    return [item for sublist in l for item in sublist if item]


api_key_blablador = os.getenv("BLABLADOR_API_KEY")
client_blablador = instructor.patch(
    OpenAI(
        api_key=api_key_blablador,
        base_url="https://helmholtz-blablador.fz-juelich.de:8000/v1/",
    ),
    # mode=instructor.Mode.MD_JSON,
    mode=instructor.Mode.TOOLS,
)


def get_snippets_blablador(question: str, title: str, abstract: str, model):
    submission = SNIPPET_TEMPLATE.format(
        question=question, title=title, abstract=abstract
    )
    resp = client_blablador.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a world class system to extract relevant sentences from titles and abstracts answering questions",
            },
            {
                "role": "user",
                "content": "Extract from the title and abstract ONLY sentences of phrases that directly answer the question",
            },
            {
                "role": "user",
                "content": submission,
            },
        ],
        temperature=0,
        max_tokens=1000,
    )
    resp = resp.choices[0].message.content
    matches1, matches2 = [], []
    pattern1 = r"\[([^]]+)\]"

    matches = re.findall(pattern1, resp)

    list1 = matches[0].strip() if matches else []
    list2 = matches[1].strip() if len(matches) > 1 else []

    pattern2 = r'".*?"(?=,|$)'

    try:
        matches1 = re.findall(pattern2, list1)
    except Exception as e:
        matches1 = []
    try:
        matches2 = re.findall(pattern2, list2)
    except Exception as e:
        matches2 = []

    if matches1:
        matches1 = [m.strip('"') if "empty list" not in m else "" for m in matches1]
    if matches2:
        matches2 = [m.strip('"') if "empty list" not in m else "" for m in matches2]

    return matches1, matches2
