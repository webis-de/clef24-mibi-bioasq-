from .prompts import *
from transformers import pipeline


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
