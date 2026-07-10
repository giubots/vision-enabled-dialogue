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

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass
class GenData:
    """Metadata captured for a generation request."""

    details: str
    """Details about the generation like model and temperature."""
    instructions: str
    """The system instructions used for the generation."""
    prompt: list[dict]
    """The conversation messages used for the generation."""
    final_instruction: str | None
    """The final user instruction used for the generation, if any."""
    previous_data: "GenData | None"
    """Previous generation data, if any, to allow chaining of generations."""


class OpenAIFormattable(ABC):
    """Interface for objects that can be converted to OpenAI input."""

    @abstractmethod
    def to_openai(self) -> list[dict]:
        """Convert to OpenAI input format."""


class LLM(ABC):
    """Base class for streaming and structured language-model queries."""

    _logger: logging.Logger

    def __init__(self, logger: logging.Logger | None = None):
        self._logger = logger or logging.getLogger(__name__)

    def query(
        self,
        instructions: str,
        messages: list[OpenAIFormattable],
        final_instruction: str | None = None,
    ) -> tuple[str, GenData]:
        """Run a blocking query and return the full text plus metadata.

        :param instructions: The system instructions.
        :param messages: The conversation messages.
        :param final_instruction: Optional final user instruction.
        :returns: The generated text and metadata.
        """
        text_s = self._query_impl(instructions, messages, final_instruction)
        return "".join(text_s), self._data_impl(
            instructions, messages, final_instruction
        )

    def query_async(
        self,
        instructions: str,
        messages: list[OpenAIFormattable],
        final_instruction: str | None = None,
    ) -> tuple[Future[str], GenData]:
        """Run a query in a worker thread and return a future plus metadata.

        :param instructions: The system instructions.
        :param messages: The conversation messages.
        :param final_instruction: Optional final user instruction.
        :returns: The future of the generated text and generation metadata.
        """

        def my_eq(self: Future[str], other: object) -> bool:
            return self.result() == other

        def my_repr(self: Future[str]) -> str:
            return f"'{self.result()}'"

        def wrapper() -> str:
            text_s = self._query_impl(instructions, messages, final_instruction)
            return "".join(text_s)

        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(wrapper)
            future.__class__.__eq__ = my_eq
            future.__class__.__repr__ = my_repr
        return future, self._data_impl(instructions, messages, final_instruction)

    def query_gen(
        self,
        instructions: str,
        messages: list[OpenAIFormattable],
        final_instruction: str | None = None,
    ) -> tuple[Generator[str], GenData]:
        """Run a query and return the text stream plus metadata.

        :param instructions: The system instructions.
        :param messages: The conversation messages.
        :param final_instruction: Optional final user instruction.
        :returns: The streamed text chunks and generation metadata.
        """
        text_s = self._query_impl(instructions, messages, final_instruction)
        return (text_s, self._data_impl(instructions, messages, final_instruction))

    def formatted(
        self,
        instructions: str,
        messages: list[OpenAIFormattable],
        format_t: type[T],
    ) -> tuple[T, GenData]:
        """Run a structured query and parse the response.

        :param instructions: The system instructions.
        :param messages: The conversation messages.
        :param format_t: The target response type.
        :returns: The parsed response and generation metadata.
        """
        return (
            self._formatted_impl(instructions, messages, format_t),
            self._data_impl(instructions, messages, None),
        )

    @abstractmethod
    def _data_impl(
        self,
        i: str,
        p: list[OpenAIFormattable],
        a: str | None,
    ) -> GenData:
        pass

    @abstractmethod
    def _query_impl(
        self,
        instructions: str,
        messages: list[OpenAIFormattable],
        final_instruction: str | None = None,
    ) -> Generator[str]:
        self._logger.debug(
            f"Query started with instructions: {instructions} and final_instruction: {final_instruction}. Messages: {[m.to_openai() for m in messages]}"
        )

    @abstractmethod
    def _formatted_impl(
        self,
        instructions: str,
        messages: list[OpenAIFormattable],
        format_t: type[T],
    ) -> T:
        pass
