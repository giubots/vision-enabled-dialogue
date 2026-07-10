# Copyright 2025 Giulio Antonio Abbo
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from abc import ABC
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConversationPart(ABC):
    """Base record for items stored in the conversation history."""

    stamp: float
    content: Any
    author: str | None = field(init=False, default=None)
    children: Sequence["ConversationPart"] | None = field(init=False, default=None)
    gen_data: Any | None = field(init=False, default=None)
    is_heavy: bool = field(init=False, default=False)
    is_hidden: bool = field(init=False, default=False)
    is_important: bool = field(init=False, default=False)

    def to_openai(self) -> Sequence[dict[str, str]]:
        """Convert to OpenAI format.

        The returned value <val> can be used as input for the OpenAI API.
        ```
            client.responses.create(
                model="gpt-4.1",
                input=[{"role": "user", "content": [<val>, ...]}]
            )
        ```
        """
        return []

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dictionary for JSON serialization.

        In addition to the dataclass fields, this includes a "_type" field with
        the class name.
        """
        as_dict = asdict(self)
        as_dict["_type"] = self.__class__.__name__
        if isinstance(self.content, Future):
            as_dict["content"] = self.content.result()
        return as_dict


@dataclass(frozen=True)
class Thought(ConversationPart):
    """A hidden assistant thought recorded as important text."""

    content: str
    is_important: bool = field(init=False, default=True)

    def to_openai(self) -> Sequence[dict]:
        return [
            {
                "type": "input_text",
                "text": f"THOUGHT: {self.content}",
            }
        ]


@dataclass(frozen=True)
class Turn(ConversationPart):
    """A spoken conversation turn with a known author."""

    content: str
    author: str
    is_important: bool = field(init=False, default=True)

    def to_openai(self) -> Sequence[dict]:
        return [
            {
                "type": "input_text",
                "text": f"{self.author.upper()}: {self.content}",
            }
        ]


@dataclass(frozen=True)
class GeneratedTurn(Turn):
    """A generated assistant turn with attached generation metadata."""

    gen_data: Any


@dataclass(frozen=True)
class Frame(ConversationPart):
    """A camera frame stored as base64-encoded JPEG data."""

    content: str
    is_heavy: bool = field(init=False, default=True)

    def to_openai(self) -> Sequence[dict]:
        return [
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{self.content}",
            }
        ]


@dataclass(frozen=True)
class ConversationNode(ConversationPart):
    """A conversation part that groups child parts."""

    children: Sequence[ConversationPart]


@dataclass(frozen=True)
class FrameSummary(ConversationNode):
    """A summary node for a group of frames.

    Note that the content can be a Future[str] if the summary is being generated
    asynchronously.
    """

    content: str | Future[str]
    children: Sequence[Frame]
    gen_data: Any

    def to_openai(self) -> Sequence[dict]:
        if isinstance(self.content, Future) and not self.content.done():
            return [cc for c in self.children for cc in c.to_openai()]
        return [
            {
                "type": "input_text",
                "text": f"VISUAL CONTEXT: {self.content}",
            }
        ]


@dataclass(frozen=True)
class ConversationSummary(ConversationNode):
    """A summary node for a group of conversation parts."""

    content: str | Future[str]
    gen_data: Any

    def to_openai(self) -> Sequence[dict]:
        if isinstance(self.content, Future) and not self.content.done():
            return [cc for c in self.children for cc in c.to_openai()]
        return [
            {
                "type": "input_text",
                "text": f"CONVERSATION SUMMARY: {self.content}",
            }
        ]


@dataclass(frozen=True)
class HiddenPart(ConversationNode):
    """A hidden node that keeps grouped parts out of the transcript."""

    content: None = field(init=False, default=None)
    is_hidden: bool = field(init=False, default=True)
