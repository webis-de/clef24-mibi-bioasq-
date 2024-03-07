from random import randint, random
from pydantic_core import Url
from mibi.model import Question, PartialAnswer, Documents, Snippets, IdealAnswer, Snippet, YesNoExactAnswer, FactoidExactAnswer, ListExactAnswer
from mibi.modules import DocumentsMaker, SnippetsMaker, SwitchExactAnswerMaker, IdealAnswerMaker


class MockDocumentsMaker(DocumentsMaker):
    def make_documents(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> Documents:
        id = str(randint(1086751, 2286751))  # nosec: B311
        return [Url(f"http://www.ncbi.nlm.nih.gov/pubmed/{id}")]


class MockSnippetsMaker(SnippetsMaker):
    def make_snippets(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> Snippets:
        id = str(randint(1086751, 2286751))  # nosec: B311
        begin_section = randint(0, 10)  # nosec: B311
        end_section = randint(begin_section, begin_section+10)  # nosec: B311
        begin_offset = randint(0, 10)  # nosec: B311
        end_offset = randint(begin_offset, begin_offset+10)  # nosec: B311
        return [
            Snippet(
                document=Url(f"http://www.ncbi.nlm.nih.gov/pubmed/{id}"),
                text="",
                offset_in_begin_section=begin_offset,
                offset_in_end_section=end_offset,
                begin_section=f"section.{begin_section}",
                end_section=f"section.{end_section}",
            )
        ]


class MockExactAnswerMaker(SwitchExactAnswerMaker):
    def make_yes_no_exact_answer(
            self,
            question: Question,
            partial_answer: PartialAnswer,
    ) -> YesNoExactAnswer:
        return "yes" if random() > 0.5 else "no"  # nosec: B311

    def make_factoid_exact_answer(
            self,
            question: Question,
            partial_answer: PartialAnswer,
    ) -> FactoidExactAnswer:
        return ["foo"] if random() > 0.5 else ["bar"]  # nosec: B311

    def make_list_exact_answer(
            self,
            question: Question,
            partial_answer: PartialAnswer,
    ) -> ListExactAnswer:
        return ([["foo"], ["baz"]]
                if random() > 0.5 else  # nosec: B311
                [["bar"], ["baz"]])


class MockIdealAnswerMaker(IdealAnswerMaker):
    def make_ideal_answer(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> IdealAnswer:
        return ["Lorem ipsum"]
