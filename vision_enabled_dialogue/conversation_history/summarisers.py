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

from ..llm import LLM
from .parts import (
    ConversationPart,
    ConversationSummary,
    Frame,
    FrameSummary,
    HiddenPart,
)


class Summariser(ABC):
    """Base class for buffered parts that produce one summary part.

    A summary of a `ConversationPart` is a new `ConversationPart` that
    represents the content of the buffered parts in a more compact form. The
    summariser should be able to determine whether a new part can be added to
    the buffer, and when requested, produce a summary part for the buffered
    content.

    The summariser should be disposed after producing a summary, and a new
    instance should be created for future parts.
    """

    _buffer: list

    def __init__(self):
        self._buffer = []

    @abstractmethod
    def can_append(self, part: ConversationPart) -> bool:
        """Return whether the part can be added to the current buffer.

        :param part: The part to check.
        :returns: Whether the part can be added to the buffer.
        """

    def append(self, part: ConversationPart) -> None:
        """Append a compatible part to the buffer.

        :param part: The part to append.
        :raises ValueError: If the part is not compatible with the buffer."""
        if not self.can_append(part):
            raise ValueError("Incompatible part")
        self._buffer.append(part)

    @abstractmethod
    def summarise(self) -> ConversationPart:
        """Return a summary part for the buffered content.

        Dispose of the summariser after calling this method."""

    def has_parts(self) -> bool:
        """Return whether the buffer contains any parts."""
        return len(self._buffer) > 0


class FrameSummariser(Summariser):
    """Summarise buffered frames into a frame summary.

    Only `Frame` parts can be added to the buffer, and the summary will be a
    `FrameSummary` part that contains the buffered frames and a summary of their
    content.
    """

    DEFAULT_INSTRUCTION = "These are frames from a video. Summarise what's happening in the video in one sentence. The frames are preceded by a context to help you summarise the video. Summarise only the frames, not the context. The images can be repeating, this is normal, do not point this out in the description. Respond with only the summary in one sentence. This is very important. Do not include warnings or other messages."

    llm: LLM
    """The language model used to generate the summary."""
    instruction: str
    """The instruction to provide to the language model for summarisation."""

    def __init__(self, llm: LLM, instruction: str | None = None):
        super().__init__()
        self.llm = llm
        self.instruction = instruction or self.DEFAULT_INSTRUCTION

    def can_append(self, part: ConversationPart) -> bool:
        return isinstance(part, Frame)

    def summarise(self) -> FrameSummary:
        future, data = self.llm.query_async(self.instruction, self._buffer)
        return FrameSummary(
            stamp=self._buffer[-1].stamp if self._buffer else 0.0,
            content=future,
            children=self._buffer,
            gen_data=data,
        )


class MixedSummariser(Summariser):
    """Summarise mixed conversation parts into a text summary."""

    DEFAULT_INSTRUCTION = "Summarise the following conversation in one paragraph."

    llm: LLM
    """The language model used to generate the summary."""
    instruction: str
    """The instruction to provide to the language model for summarisation."""

    def __init__(self, llm: LLM, instruction: str | None = None):
        super().__init__()
        self.llm = llm
        self.instruction = instruction or self.DEFAULT_INSTRUCTION

    def can_append(self, part: ConversationPart) -> bool:
        return True

    def summarise(self) -> ConversationSummary:
        future, data = self.llm.query_async(self.instruction, self._buffer)
        return ConversationSummary(
            stamp=self._buffer[-1].stamp if self._buffer else 0.0,
            content=future,
            children=self._buffer,
            gen_data=data,
        )


class HidingSummariser(Summariser):
    """Group parts and hides them from the conversation history."""

    _buffer: list[ConversationPart]

    def can_append(self, part: ConversationPart) -> bool:
        return True

    def summarise(self) -> HiddenPart:
        return HiddenPart(
            stamp=self._buffer[-1].stamp if self._buffer else 0.0,
            children=self._buffer,
        )
