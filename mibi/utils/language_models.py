from os import environ
from typing import Literal
from dspy import OpenAI as DSPyOpenAI, HFModel, settings as dspy_settings
from dsp import LM


def init_language_model_clients(
    language_model_name: Literal[
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "text-davinci-003",
        "Mixtral-8x7B-Instruct-v0.1",
        "Mistral-7B-Instruct-v0.2",
    ],
) -> None:
    lm: LM
    if "OPENAI_API_KEY" in environ.keys():
        lm = DSPyOpenAI(
            model=language_model_name,
            api_key=environ["OPENAI_API_KEY"],
            api_base=environ.get("OPENAI_API_BASE")
        )
    else:
        lm = HFModel(
            model=language_model_name,
        )
    dspy_settings.configure(
        lm=lm,
    )
