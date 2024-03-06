# The submitted answer to BioASQ should depend on the question type and be: ideal (longer) and exact (shorter)
# TODO: implement ideal answer
# TODO: for each Q type implement seperate pydantic class and use fn calling (does fn calling work with the llama summarizer?)

import json

from pydantic import BaseModel, Field
from typing import List, Literal

from .config import client, load_toml_params


params = load_toml_params("config.toml")


class YesNoExact(BaseModel):
    answer: Literal["yes", "no"]


class ListExact(BaseModel):
    answer: List[List[str]] = Field(
        ..., description="List of lists with single entities"
    )


class FactoidExact(BaseModel):
    answer: List[str] = Field(
        ...,
        description="A very short fact for example, a single entity or a short phrase",
    )


class Summary(BaseModel):
    answer: List[str] = Field(
        ...,
        description="A summary of the retrieved context in 2 or 3 sentences. It also contains a short explanation",
    )


class IdealAnswer(BaseModel):
    answer: str = Field(
        ...,
        description="""The ideal answer to the question in one longer sentence that also contains a short explanation.
                                       The ideal answer is grammatically complete with subjects, objects, and predicates, is concise and precise.""",
    )


class Answer:
    def response_exact_answer(query: str, q_type: str, text_chunks: str):

        # answer_type_dict = {
        #     "yesno": "only yes or no",
        #     "list": "a python list of lists with entities",
        #     "factoid": "a very short fact only a single entity included in a Python list with one element",
        #     "summary": "a summary of the retrieved context in 2 or 3 sentences included in a Python list with one element"
        # }

        response_model_dict = {
            "yesno": YesNoExact,
            "list": ListExact,
            "factoid": FactoidExact,
            "summary": Summary,
        }

        response_model = response_model_dict[q_type]

        # answer_type = answer_type_dict[q_type]

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
            # {"role": "system", 'content': f'Please write the answer as {answer_type}'},
            {"role": "user", "content": f"Question: {query}"},
        ]

        response = client.chat.completions.create(
            model=params["generation"]["model"],
            temperature=0,
            response_model=response_model,
            messages=messages,
            max_tokens=1000,
        )

        if q_type == "summary":
            return [" ".join(response.answer)]
        else:
            return response.answer

    def response_ideal_answer(query: str, q_type: str, text_chunks: str):

        response_model: type[BaseModel]
        if q_type == "summary":
            response_model = Summary
        else:
            response_model = IdealAnswer

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
            model=params["generation"]["model"],
            temperature=0,
            response_model=response_model,
            messages=messages,
            max_tokens=1000,
        )

        if q_type == "summary":
            return [" ".join(response.answer)]
        else:
            return [response.answer]


with open("data/bioasq/questions_selected.json", "r") as f:
    data = json.load(f)


# summary-type questions do not have exact_answer
def print_exact_answer():
    for d in data:
        query = d["body"]
        q_type = d["type"]
        retrieved_results = [e["text"] for e in d["snippets"]]
        text_chunks = " ".join([i for i in retrieved_results])
        print(f"QUERY: {query}, TYPE: {q_type}")
        print(f"GENERATED: {response_exact_answer(query, q_type, text_chunks)}")
        if d["type"] != "summary":
            print(f"TRUTH    : {d['exact_answer']}")
            print()
        else:
            print(f"TRUTH    : {d['ideal_answer']}")
            print()


def print_ideal_answer():
    for d in data:
        query = d["body"]
        q_type = d["type"]
        retrieved_results = [e["text"] for e in d["snippets"]]
        text_chunks = " ".join([i for i in retrieved_results])
        print(f"QUERY: {query}, TYPE: {q_type}")
        print(f"GENERATED: {response_ideal_answer(query, q_type, text_chunks)}")
        print(f"TRUTH    : {d['ideal_answer']}")
        print()


# print_exact_answer()

print_ideal_answer()


# Don't get a proper answer to the factoid question
# QUERY: Name synonym of Acrokeratosis paraneoplastica., TYPE: factoid
# GENERATED: ['Bazex syndrome']
# TRUTH    : ['Acrokeratosis paraneoplastic (Bazex syndrome) is a rare, but distinctive paraneoplastic ...
