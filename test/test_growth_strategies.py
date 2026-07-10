from dataclasses import dataclass
from functools import partial

from vision_enabled_dialogue.conversation_history.parts import (
    ConversationSummary,
    Frame,
    FrameSummary,
    Turn,
)
from vision_enabled_dialogue.conversation_history.strategies import (
    KeepLastNStrategy,
    SummariseHeavy,
)
from vision_enabled_dialogue.conversation_history.summarisers import (
    FrameSummariser,
    MixedSummariser,
)
from vision_enabled_dialogue.llm import LLM


@dataclass(frozen=True)
class HeavyConversationPart(Frame):
    pass


@dataclass(frozen=True)
class LightConversationPart(Turn):
    author: str = "LIGHT"


def test_summarise_heavy():
    initial = [
        LightConversationPart(stamp=0, content="a"),
        LightConversationPart(stamp=1, content="b"),
        HeavyConversationPart(stamp=2, content="c"),
        LightConversationPart(stamp=3, content="d"),
        LightConversationPart(stamp=4, content="e"),
        HeavyConversationPart(stamp=5, content="f"),
        HeavyConversationPart(stamp=6, content="g"),
        LightConversationPart(stamp=7, content="h"),
        HeavyConversationPart(stamp=8, content="i"),
        LightConversationPart(stamp=9, content="j"),
        HeavyConversationPart(stamp=10, content="k"),
        HeavyConversationPart(stamp=11, content="l"),
        HeavyConversationPart(stamp=12, content="m"),
        LightConversationPart(stamp=13, content="n"),
    ]

    expected = [
        LightConversationPart(stamp=0, content="a"),
        LightConversationPart(stamp=1, content="b"),
        FrameSummary(
            stamp=2,
            content="Image summary: c",
            children=[HeavyConversationPart(stamp=2, content="c")],
            gen_data={},
        ),
        LightConversationPart(stamp=3, content="d"),
        LightConversationPart(stamp=4, content="e"),
        FrameSummary(
            stamp=6,
            content="Image summary: f, g",
            children=[
                HeavyConversationPart(stamp=5, content="f"),
                HeavyConversationPart(stamp=6, content="g"),
            ],
            gen_data={},
        ),
        LightConversationPart(stamp=7, content="h"),
        FrameSummary(
            stamp=8,
            content="Image summary: i",
            children=[HeavyConversationPart(stamp=8, content="i")],
            gen_data={},
        ),
        LightConversationPart(stamp=9, content="j"),
        FrameSummary(
            stamp=10,
            content="Image summary: k",
            children=[HeavyConversationPart(stamp=10, content="k")],
            gen_data={},
        ),
        HeavyConversationPart(stamp=11, content="l"),
        HeavyConversationPart(stamp=12, content="m"),
        LightConversationPart(stamp=13, content="n"),
    ]

    class MockLLM(LLM):
        def _query_impl(self, instructions, messages, final_instr=None):
            yield "Image summary: " + ", ".join(str(p.content) for p in messages)

        def _data_impl(self, i, p, a):  # type: ignore
            return {}

        def _formatted_impl(self, instructions, messages, format) -> str:
            return ""

    summariser = SummariseHeavy(
        summariser_factories=[partial(FrameSummariser, llm=MockLLM())],
        max_heavy=6,
        min_heavy=2,
    )
    for part in initial:
        summariser.notify(part)
    assert summariser.should_apply()

    actual = summariser.apply(initial)
    assert actual == expected


def test_keep_last_n_frames_strategy():
    initial = [
        Frame(stamp=0, content="a"),
        Frame(stamp=1, content="b"),
        Frame(stamp=2, content="c"),
        Turn(stamp=3, content="d", author="T"),
        Frame(stamp=4, content="e"),
        Frame(stamp=5, content="f"),
        Turn(stamp=6, content="g", author="T"),
        Frame(stamp=7, content="h"),
        Turn(stamp=8, content="i", author="T"),
        Turn(stamp=9, content="j", author="T"),
        Frame(stamp=10, content="k"),
        Turn(stamp=11, content="l", author="T"),
        Frame(stamp=12, content="m"),
        Turn(stamp=13, content="n", author="T"),
        Frame(stamp=14, content="o"),
    ]

    expected = [
        ConversationSummary(
            stamp=10,
            content="Summary: a, b, c, d, e, f, g, h, i, j, k",
            children=[
                Frame(stamp=0, content="a"),
                Frame(stamp=1, content="b"),
                Frame(stamp=2, content="c"),
                Turn(stamp=3, content="d", author="T"),
                Frame(stamp=4, content="e"),
                Frame(stamp=5, content="f"),
                Turn(stamp=6, content="g", author="T"),
                Frame(stamp=7, content="h"),
                Turn(stamp=8, content="i", author="T"),
                Turn(stamp=9, content="j", author="T"),
                Frame(stamp=10, content="k"),
            ],
            gen_data={},
        ),
        Turn(stamp=11, content="l", author="T"),
        Frame(stamp=12, content="m"),
        Turn(stamp=13, content="n", author="T"),
        Frame(stamp=14, content="o"),
    ]

    class MockLLM(LLM):
        def _query_impl(self, instructions, messages, final_instr=None):
            yield "Summary: " + ", ".join(str(p.content) for p in messages)

        def _data_impl(self, i, p, a):  # type: ignore
            return {}

        def _formatted_impl(self, instructions, messages, format) -> str:
            return ""

    strategy = KeepLastNStrategy(
        summariser_factories=[partial(MixedSummariser, llm=MockLLM())],
        threshold=5,
        keep=2,
    )
    for part in initial:
        strategy.notify(part)
    assert strategy.should_apply()

    actual = strategy.apply(initial)
    assert actual == expected
