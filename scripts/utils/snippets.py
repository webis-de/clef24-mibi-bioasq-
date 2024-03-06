import nltk
from pydantic import BaseModel, Field
from transformers import pipeline
from typing import List

from .config import client, load_toml_params
from .prompts import SNIPPET_PROMPT


params = load_toml_params("config.toml")


class SnippetExtractorQA:
    model: str
    tokenizer: str

    def __init__(self, model, tokenizer):
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            top_k=5,
            max_seq_len=512,
            max_question_len=15,
            max_answer_len=512,
            handle_impossible_answer=False,
            # torch_dtype=torch.bfloat16
        )

    def extract(self, question: str, title: str, abstract: str):
        snippets_abstract = self.qa_pipeline(question, abstract)
        snippets_title = self.qa_pipeline(question, title)
        return snippets_title, snippets_abstract


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


class SnippetExtractorGPT:

    def extract(self, question: str, title: str, abstract: str) -> Snippets:
        submission = SNIPPET_PROMPT.format(
            question=question, title=title, abstract=abstract
        )
        resp = client.chat.completions.create(
            model=params["generation"]["model"],
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
        )
        return resp


def sent_tokenize(text: str):
    return nltk.sent_tokenize(text)
