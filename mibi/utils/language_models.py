from os import environ
from typing import Literal
from dspy import OpenAI as DSPyOpenAI, settings as dspy_settings


def init_language_model_clients(
    language_model_name: Literal[
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "text-davinci-003",
        "Mixtral-8x7B-Instruct-v0.1",
        "Mistral-7B-Instruct-v0.2",
    ],
) -> None:
    if language_model_name in (
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "text-davinci-003",
        "Mixtral-8x7B-Instruct-v0.1",
        "Mistral-7B-Instruct-v0.2",
    ) and "BLABLADOR_API_KEY" in environ.keys():
        dspy_settings.configure(
            lm=DSPyOpenAI(
                model=language_model_name,
                api_key=environ["BLABLADOR_API_KEY"],
                api_base="https://helmholtz-blablador.fz-juelich.de:8000/v1/"
            ),
        )
    elif language_model_name in (
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
    ):
        dspy_settings.configure(
            lm=DSPyOpenAI(
                model=language_model_name,
            ),
        )
    else:
        raise ValueError("Unknown language model.")
