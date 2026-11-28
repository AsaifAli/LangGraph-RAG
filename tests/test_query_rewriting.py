import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from agents.custom_langgraph_poc import _needs_contextual_rewrite, _rewrite_followup_query


class StubRewriteModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.response)


def test_standalone_question_does_not_need_contextual_rewriting():
    history = [
        HumanMessage(content="What does the agreement say about renewal?"),
        AIMessage(content="It renews annually unless notice is given."),
    ]

    assert not _needs_contextual_rewrite("What is the governing law?", history)


def test_pronoun_and_follow_up_questions_are_recognized_as_contextual():
    history = [HumanMessage(content="Summarize the employment agreement.")]

    assert _needs_contextual_rewrite("What about its termination notice?", history)
    assert _needs_contextual_rewrite("And the exceptions?", history)
    assert _needs_contextual_rewrite("What does the second one say?", history)


def test_follow_up_is_rewritten_as_a_standalone_question_using_chat_context():
    history = [
        HumanMessage(content="What are the termination and renewal provisions in the contract?"),
        AIMessage(content="The contract renews annually and requires 60 days' notice for termination."),
    ]
    model = StubRewriteModel(
        '{"rewritten_query":"What renewal notice period and termination notice period does the contract require?"}'
    )

    rewritten = asyncio.run(
        _rewrite_followup_query(model, "What about its notice period?", history)
    )

    assert rewritten == "What renewal notice period and termination notice period does the contract require?"
    assert len(model.calls) == 1
    assert "termination and renewal provisions" in model.calls[0][1].content
    assert "What about its notice period?" in model.calls[0][1].content


def test_malformed_rewrite_response_falls_back_to_the_original_question():
    history = [HumanMessage(content="Explain the supplier agreement.")]
    model = StubRewriteModel("The supplier requires 30 days of notice.")

    rewritten = asyncio.run(_rewrite_followup_query(model, "What about its term?", history))

    assert rewritten == "What about its term?"


def test_rewriter_is_not_called_for_standalone_questions_or_without_history():
    history = [HumanMessage(content="Tell me about the supplier agreement.")]
    model = StubRewriteModel('{"rewritten_query":"incorrect rewrite"}')

    standalone = asyncio.run(_rewrite_followup_query(model, "What is the payment schedule?", history))
    first_turn = asyncio.run(_rewrite_followup_query(model, "What about it?", []))

    assert standalone == "What is the payment schedule?"
    assert first_turn == "What about it?"
    assert model.calls == []
