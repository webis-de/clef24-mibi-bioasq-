from os import environ
from dspy import OpenAI as DSPyOpenAI, HFModel, settings as dspy_settings, OllamaLocal
from dsp import LM


_BLABLADOR_MODEL_NAMES = {
    "Mistral-7B-Instruct-v0.2": "1 - Mistral-7B-Instruct-v0.2 - the best option in general - fast and good",
    "Mixtral-8x7B-Instruct-v0.1": "2 - Mixtral-8x7B-Instruct-v0.1 Slower with higher quality",
    "starcoder2-15b": "3 - starcoder2-15b - A model for programming",
    "GritLM-7B": "5 - GritLM-7B - For Text-Embeddings",
}


def init_language_model_clients(language_model_name: str) -> LM:
    lm: LM
    if "OLLAMA_API_BASE" in environ.keys():
        print(
            f"Using Ollama language model '{language_model_name}' "
            f"from {environ['OPENAI_API_BASE']}."
        )
        lm = OllamaLocal(
            model=language_model_name,
            base_url=environ["OLLAMA_API_BASE"],
        )
    elif "OPENAI_API_KEY" in environ.keys():
        print(
            f"Using OpenAI language model '{language_model_name}' "
            f"from {environ.get('OPENAI_API_BASE', 'default enpoint')}."
        )
        if (
            "helmholtz-blablador.fz-juelich.de" in environ.get("OPENAI_API_BASE", "") and
                language_model_name in _BLABLADOR_MODEL_NAMES.keys()):
            language_model_name = _BLABLADOR_MODEL_NAMES[language_model_name]
        lm = DSPyOpenAI(
            model=language_model_name,
            api_key=environ["OPENAI_API_KEY"],
            api_base=environ.get("OPENAI_API_BASE")
        )
    else:
        print(
            f"Using Hugging Face language model '{language_model_name}'."
        )
        lm = HFModel(
            model=language_model_name,
        )
    dspy_settings.configure(
        lm=lm,
    )
    return lm
