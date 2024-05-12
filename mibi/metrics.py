from dataclasses import dataclass
from functools import cached_property
from math import isnan, nan
from typing import AbstractSet, Collection, Literal, Protocol
from statistics import harmonic_mean
from warnings import catch_warnings, simplefilter

from rouge_score.rouge_scorer import RougeScorer
from rouge_score.tokenizers import Tokenizer as RougeTokenizer
from spacy import load as spacy_load
from spacy.language import Language

from mibi.model import NOT_AVAILABLE, Answer, Question


class Measure(Protocol):

    def metric(self, question: Question, ground_truth_answer: Answer, predicted_answer: Answer) -> float:
        raise NotImplementedError()

    def __call__(self, question: Question, ground_truth_answer: Answer, predicted_answer: Answer) -> float:
        return self.metric(
            question=question,
            ground_truth_answer=ground_truth_answer,
            predicted_answer=predicted_answer,
        )


@dataclass(frozen=True)
class _LemmaTokenizer(RougeTokenizer):

    @cached_property
    def _language(self) -> Language:
        with catch_warnings():
            simplefilter(action="ignore", category=FutureWarning)
            return spacy_load("en_core_sci_sm")

    def tokenize(self, text: str) -> list[str]:
        doc = self._language(text)
        return [token.lemma_ for token in doc]


_LEMMA_TOKENIZER = _LemmaTokenizer()


def _normalize(text: str) -> str:
    tokens = _LEMMA_TOKENIZER.tokenize(text)
    return " ".join(tokens).lower()


@dataclass(frozen=True)
class IdealAnswerRouge(Measure):
    rouge_types: AbstractSet[Literal[
        "rougeL",
        "rougeLsum",
        "rouge1",
        "rouge2",
        "rouge3",
        "rouge4",
        "rouge5",
        "rouge6",
        "rouge7",
        "rouge8",
        "rouge9",
    ]]

    @cached_property
    def _scorer(self) -> RougeScorer:
        return RougeScorer(
            rouge_types=list(self.rouge_types),
            use_stemmer=False,
            tokenizer=_LEMMA_TOKENIZER,
        )

    def metric(
        self,
            question: Question,
            ground_truth_answer: Answer,
            predicted_answer: Answer,
    ) -> float:
        rouge_scores = self._scorer.score(
            target=ground_truth_answer.ideal_answer,
            prediction=predicted_answer.ideal_answer,
        )
        score = harmonic_mean([
            score.fmeasure
            for score in rouge_scores.values()
        ])

        print(
            f"Ideal answer score: {score:.2f} (ground truth: '{ground_truth_answer.ideal_answer}', predicted: '{predicted_answer.ideal_answer}')")
        return score


@dataclass(frozen=True)
class ExactAnswerScore(Measure):
    def metric(
        self,
            question: Question,
            ground_truth_answer: Answer,
            predicted_answer: Answer,
    ) -> float:
        if question.type == "yesno":
            if ground_truth_answer.exact_answer not in ("yes", "no") or predicted_answer.exact_answer not in ("yes", "no"):
                raise RuntimeError(
                    "Expected exact answer to be either 'yes' or 'no'.")
            # Exact match
            yes_no_score: float = 1.0 if ground_truth_answer.exact_answer == predicted_answer.exact_answer else 0.0
            print(
                f"Exact answer score ({question.type}): {yes_no_score:.2f} (ground truth: '{ground_truth_answer.exact_answer}', predicted: '{predicted_answer.exact_answer}')")
            return yes_no_score
        elif question.type == "factoid":
            if not isinstance(ground_truth_answer.exact_answer, str) or not isinstance(predicted_answer.exact_answer, str):
                raise RuntimeError("Expected exact answer to be a string.")
            ground_truth_exact_answer = _normalize(
                ground_truth_answer.exact_answer)
            predicted_exact_answer = _normalize(predicted_answer.exact_answer)
            # Exact match
            factoid_score: float = 1.0 if ground_truth_exact_answer == predicted_exact_answer else 0.0
            print(
                f"Exact answer score ({question.type}): {factoid_score:.2f} (ground truth: '{ground_truth_answer.exact_answer}', predicted: {predicted_answer.exact_answer})")
            return factoid_score
        elif question.type == "list":
            if not isinstance(ground_truth_answer.exact_answer, Collection) or not isinstance(predicted_answer.exact_answer, Collection):
                raise RuntimeError("Expected exact answer to be a collection.")
            ground_truth_exact_answers = {
                _normalize(
                    item)
                for item in ground_truth_answer.exact_answer
            }
            predicted_exact_answers = {
                _normalize(item)
                for item in predicted_answer.exact_answer
            }
            correct_exact_answers = predicted_exact_answers & ground_truth_exact_answers
            exact_answer_precision: float = (
                (
                    len(correct_exact_answers) /
                    len(predicted_exact_answers)
                )
                if len(predicted_exact_answers) > 0 else 0.0
            )
            exact_answer_recall: float = (
                (
                    len(correct_exact_answers) /
                    len(ground_truth_exact_answers)
                )
                if len(ground_truth_exact_answers) > 0 else 0.0
            )
            # F1 score
            list_score: float = harmonic_mean([
                exact_answer_precision,
                exact_answer_recall,
            ])
            print(
                f"Exact answer score ({question.type}): {list_score:.2f} (ground truth: '{ground_truth_answer.exact_answer}', predicted: '{predicted_answer.exact_answer}')")
            return list_score
        elif question.type == "summary":
            if ground_truth_answer.exact_answer != NOT_AVAILABLE or predicted_answer.exact_answer != NOT_AVAILABLE:
                raise RuntimeError("Expected exact answer to be empty.")
            return nan
        else:
            raise RuntimeError("Unknown exact answer type.")


@dataclass(frozen=True)
class DefaultMeasure(Measure):

    @cached_property
    def ideal_answer_rouge(self) -> IdealAnswerRouge:
        return IdealAnswerRouge({"rouge1", "rougeL"})

    @cached_property
    def exact_answer_score(self) -> ExactAnswerScore:
        return ExactAnswerScore()

    def metric(
        self,
            question: Question,
            ground_truth_answer: Answer,
            predicted_answer: Answer,
    ) -> float:
        scores = [
            measure.metric(
                question=question,
                ground_truth_answer=ground_truth_answer,
                predicted_answer=predicted_answer,
            )
            for measure in (
                self.ideal_answer_rouge,
                self.exact_answer_score,
            )
        ]
        score = harmonic_mean(score for score in scores if score is not None and not isnan(score))
        print(f"Answer score: {score:.2f}")
        return score
