from random import randint, random
from pydantic_core import Url
from mibi.model import Question, PartialAnswer, Documents, Snippets, IdealAnswer, Snippet, YesNoExactAnswer, FactoidExactAnswer, ListExactAnswer
from mibi.modules import DocumentsModule, SnippetsModule, IdealAnswerModule
from mibi.modules.helpers import AutoExactAnswerModule


class MockDocumentsModule(DocumentsModule):
    def forward(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> Documents:
        num_docs = randint(1, 25)  # nosec: B311
        return [
            # nosec: B311
            Url(
                f"http://www.ncbi.nlm.nih.gov/pubmed/{randint(1086751, 2286751)}")
            for _ in range(num_docs)
        ]


def _random_snippet() -> Snippet:
    id = str(randint(1086751, 2286751))  # nosec: B311
    begin_section = randint(0, 10)  # nosec: B311
    end_section = randint(begin_section, begin_section+10)  # nosec: B311
    begin_offset = randint(0, 10)  # nosec: B311
    end_offset = randint(begin_offset, begin_offset+10)  # nosec: B311
    return Snippet(
        document=Url(f"http://www.ncbi.nlm.nih.gov/pubmed/{id}"),
        text="",
        offset_in_begin_section=begin_offset,
        offset_in_end_section=end_offset,
        begin_section=f"section.{begin_section}",
        end_section=f"section.{end_section}",
    )


class MockSnippetsModule(SnippetsModule):
    def forward(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> Snippets:
        num_docs = randint(1, 25)  # nosec: B311
        return [
            _random_snippet()
            for _ in range(num_docs)
        ]


class MockExactAnswerModule(AutoExactAnswerModule):
    def forward_yes_no(
            self,
            question: Question,
            partial_answer: PartialAnswer,
    ) -> YesNoExactAnswer:
        return "yes" if random() > 0.5 else "no"  # nosec: B311

    def forward_factoid(
            self,
            question: Question,
            partial_answer: PartialAnswer,
    ) -> FactoidExactAnswer:
        return "foo" if random() > 0.5 else "bar"  # nosec: B311

    def forward_list(
            self,
            question: Question,
            partial_answer: PartialAnswer,
    ) -> ListExactAnswer:
        return (
            ["foo", "baz"]
            if random() > 0.5 else  # nosec: B311
            ["bar", "baz"]
        )


class MockIdealAnswerModule(IdealAnswerModule):
    def forward(
        self,
        question: Question,
        partial_answer: PartialAnswer,
    ) -> IdealAnswer:
        return "Lorem ipsum"
