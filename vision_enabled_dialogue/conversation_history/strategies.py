# Copyright 2025 Giulio Antonio Abbo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from collections.abc import Callable

from .parts import ConversationPart
from .summarisers import Summariser

SummariserFactory = Callable[[], Summariser]
"""Factory type for creating new summariser instances."""


class GrowthStrategy(ABC):
    """Base class for strategies that compact conversation history.

    Strategies are notified of new conversation parts and can decide when to
    apply themselves to compact the history.

    Once a strategy decides to apply, it will be given the current list of
    conversation parts and should return a new list of parts that represents the
    compacted history. After these parts are returned, the strategy should be
    disposed and a new instance should be created for future notifications.
    """

    summariser_factories: list[SummariserFactory]
    """Methods to create new summariser instances for the strategy when needed."""

    def __init__(self, summariser_factories: list[SummariserFactory]):
        self.summariser_factories = summariser_factories

    def notify_multiple(self, parts: list[ConversationPart]) -> None:
        """Notify the strategy about multiple new parts.

        :param parts: The parts that were added.
        """
        for part in parts:
            self.notify(part)

    @abstractmethod
    def notify(self, part: ConversationPart) -> bool:
        """Notify the strategy about one new part.

        :param part: The part that was added.
        :returns: Whether the strategy should be applied.
        """

    @abstractmethod
    def should_apply(self) -> bool:
        """Return whether the strategy is ready to run."""

    @abstractmethod
    def apply(self, parts: list[ConversationPart]) -> list[ConversationPart]:
        """Return a compacted list of conversation parts.

        :param parts: The current list of conversation parts.
        :returns: A new list of conversation parts that represents the compacted
            history.
        """


class KeepLastNStrategy(GrowthStrategy):
    """Summarise older content while keeping the last important frames.

    Summarisation is triggered when the number of `ConversationPart` with
    `is_important` set to `True` exceeds a threshold. Summarisation will keep
    the last N important parts and summarise the rest using the provided
    summarisers.

    For example, this can be used to keep the last N turns of the conversation
    while summarising older content to save memory and improve performance.
    """

    threshold: int
    """Summarise when the important parts exceed this threshold."""
    keep: int
    """Keep this many important parts in the history."""

    _counter: int

    def __init__(
        self,
        summariser_factories: list[SummariserFactory],
        threshold: int = 6,
        keep: int = 4,
    ):
        """Initialize the strategy with a threshold and number of parts to keep.

        :param summariser_factories: Functions that return summariser instances.
        :param threshold: The number of important parts that triggers
            summarisation.
        :param keep: The number of important parts to keep in the history.
        """
        super().__init__(summariser_factories)
        assert threshold > keep
        self.threshold = threshold
        self.keep = keep
        self._counter = 0

    def notify(self, part: ConversationPart) -> bool:
        if part.is_important:
            self._counter += 1
        return self.should_apply()

    def should_apply(self) -> bool:
        return self._counter > self.threshold

    def apply(self, parts: list[ConversationPart]) -> list[ConversationPart]:
        last_to_keep = [p for p in parts if p.is_important][-self.keep]
        index = parts.index(last_to_keep)
        to_summarise = parts[:index]

        result = []
        current_summariser = None
        for part in to_summarise:
            if current_summariser is not None:
                if current_summariser.can_append(part):
                    current_summariser.append(part)
                    continue
                else:
                    result.append(current_summariser.summarise())
                    current_summariser = None

            # summariser is None
            current_summariser = next(
                s() for s in self.summariser_factories if s().can_append(part)
            )
            current_summariser.append(part)

        if current_summariser is not None:
            result.append(current_summariser.summarise())
        result.extend(parts[index:])

        self._counter = self.keep
        return result


class SummariseHeavy(GrowthStrategy):
    """Summarise heavy parts once their count exceeds the limit.

    Summarisation is triggered when the number of `ConversationPart` with
    `is_heavy` set to `True` exceeds `max_heavy`. Summarisation will keep at
    least `min_heavy` heavy parts and summarise the rest, without mixing them
    with non-heavy parts.

    For example, this can be used to summarise older frames in a video while
    keeping the last few frames for context.

    This strategy reflects what reported in: _Giulio Antonio Abbo and Tony
    Belpaeme. 2025. I Was Blind but Now I See: Implementing Vision-Enabled
    Dialogue in Social Robots. In Proceedings of the 2025 ACM/IEEE International
    Conference on Human-Robot Interaction (HRI '25). IEEE Press, 1176-1180._
    """

    max_heavy: int
    """Summarise when the heavy parts exceed this threshold."""
    min_heavy: int
    """Keep at least this many heavy parts in the history."""

    _heavy_count: int

    def __init__(
        self,
        summariser_factories: list[SummariserFactory],
        max_heavy: int = 4,
        min_heavy: int = 3,
    ):
        """Initialize the strategy with a threshold and number of parts to keep.

        :param summariser_factories: Functions that return summariser instances.
        :param max_heavy: The number of heavy parts that triggers summarisation.
        :param min_heavy: The number of heavy parts to keep in the history.
        """
        super().__init__(summariser_factories)
        assert max_heavy >= min_heavy
        self.max_heavy = max_heavy
        self.min_heavy = min_heavy
        self._heavy_count = 0

    def notify(self, part: ConversationPart) -> bool:
        if part.is_heavy:
            self._heavy_count += 1
        return self.should_apply()

    def should_apply(self) -> bool:
        return self._heavy_count > self.max_heavy

    def apply(self, parts: list[ConversationPart]) -> list[ConversationPart]:
        available = self._heavy_count - self.min_heavy
        result = []
        current_summariser: Summariser | None = None
        for part in parts:
            if not part.is_heavy:
                if current_summariser is not None and current_summariser.has_parts():
                    result.append(current_summariser.summarise())
                    current_summariser = None
                result.append(part)
                continue
            if available > 0:
                if current_summariser is None:
                    current_summariser = next(
                        s() for s in self.summariser_factories if s().can_append(part)
                    )
                current_summariser.append(part)
                available -= 1
                continue

            if current_summariser is not None and current_summariser.has_parts():
                result.append(current_summariser.summarise())
                current_summariser = None
            result.append(part)
        self._heavy_count = self.min_heavy
        return result
