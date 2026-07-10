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

import re
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from time import time

from pydantic import BaseModel

from .conversation_history import ConversationHistory
from .llm import LLM, GenData


@dataclass
class GeneratedResponse:
    """A generated response and its associated metadata."""

    stamp: float
    """Seconds since epoch when the response was generated."""
    response: Generator[str]
    """The generated response as a stream of text."""
    gen_data: GenData
    """Metadata about the response generation."""


class ResponseGenerator(ABC):
    """Base class for dialogue response generators."""

    @abstractmethod
    def get_response(
        self,
        history: ConversationHistory,
        inst: str | None = None,
        p_data: GenData | None = None,
    ) -> GeneratedResponse:
        """Return a generated response for the current history.

        :param history: The conversation history to generate a response for.
        :param inst: Optional final steering instructions.
        :param p_data: Optional previous generation data to chain generations.
        :return: A GeneratedResponse containing the response and metadata.
        """


class LLMResponseGenerator(ResponseGenerator):
    """Generate responses with a text-only language model."""

    DEFAULT_INSTRUCTIONS = "You are impersonating a friendly kid. Have a nice conversation and try to be curious! It is important that you keep your answers short and to the point. DO NOT INCLUDE EMOTICONS OR SMILEYS IN YOUR ANSWERS."

    llm: LLM
    """The language model used to generate the response."""
    instructions: str
    """The instructions to use for the response generation."""

    def __init__(self, llm: LLM, instructions: str | None = None):
        """Initialize the generator with optional instructions.

        :param llm: The language model to use for response generation.
        :param instructions: Optional instructions to override the default
            instructions.
        """
        self.llm = llm
        self.instructions = instructions or self.DEFAULT_INSTRUCTIONS

    def get_response(
        self,
        history: ConversationHistory,
        inst: str | None = None,
        p_data: GenData | None = None,
    ) -> GeneratedResponse:
        stamp = time()
        res, data = self.llm.query_gen(self.instructions, history.parts, inst)
        if p_data is not None:
            data.previous_data = p_data
        return GeneratedResponse(stamp, res, data)


class VLMResponseGenerator(ResponseGenerator):
    """Generate responses with a vision-language model."""

    DEFAULT_INSTRUCTIONS = "You are impersonating a friendly kid. In this conversation, what you see is represented by the images. For example, the images will show you the environment you are in and possibly the person you are talking to. Try to start the conversation by saying something about the person you are talking to if there is one, based on accessories, clothes, etc. If there is no person, try to say something about the environment, but do not describe the environment! Have a nice conversation and try to be curious! It is important that you keep your answers short and to the point. DO NOT INCLUDE EMOTICONS OR SMILEYS IN YOUR ANSWERS."

    vlm: LLM
    """The vision-language model used to generate the response."""
    instructions: str
    """The instructions to use for the response generation."""

    def __init__(self, vlm: LLM, instructions: str | None = None):
        """Initialize the generator with optional instructions.

        :param vlm: The vision-language model to use for response generation.
        :param instructions: Optional instructions to override the default
            instructions.
        """
        self.vlm = vlm
        self.instructions = instructions or self.DEFAULT_INSTRUCTIONS

    def get_response(
        self,
        history: ConversationHistory,
        inst: str | None = None,
        p_data: GenData | None = None,
    ) -> GeneratedResponse:
        stamp = time()
        res, data = self.vlm.query_gen(self.instructions, history.parts, inst)
        if p_data is not None:
            data.previous_data = p_data
        return GeneratedResponse(stamp, res, data)


class DualResponseGenerator(ResponseGenerator):
    """Choose between text and vision generation for a response.

    Calling `force_vlm()` will force the next response to use the vision
    generator.
    """

    DEFAULT_INSTRUCTIONS = "You are in charge of choosing the best model for the current conversation. You can choose between the Vision Language Model (VLM) and the Language Model (LM). The VLM is a model that can understand images and text, and can generate text and image descriptions. The LM is a model that can only understand text and generate text. If the provided text is about colors, what something looks like, holding objects, appearance, the environment and so on, you will choose VLM. If the user asks how does something look like, what am I holding, and similar, then use a VLM. If the text is about abstract things, conversational, and in all those cases where it cannot benefit from an image input, use the LM. The LM is faster and cheaper, and should be the preferred choice, unless the input could benefit from using image input. Does this input need a VLM?"

    llm_generator: LLMResponseGenerator
    """The text-only language model generator."""
    vlm_generator: VLMResponseGenerator
    """The vision-language model generator."""
    chooser: LLM
    """A fast model that chooses which generator toq use."""
    instructions: str
    """The instructions to use for the chooser model."""

    _force_vlm: bool

    def __init__(
        self, llm: LLM, vlm: LLM, chooser: LLM, instructions: str | None = None
    ):
        """Initialize the dual generator with the two models and a chooser.

        :param llm: The text-only language model to use for response generation.
        :param vlm: The vision-language model to use for response generation.
        :param chooser: A fast model that chooses which generator to use.
        :param instructions: Optional instructions to override the default
            instructions for the chooser model.
        """
        self.llm_generator = LLMResponseGenerator(llm)
        self.vlm_generator = VLMResponseGenerator(vlm)
        self.chooser = chooser
        self.instructions = instructions or self.DEFAULT_INSTRUCTIONS
        self._force_vlm = False

    def force_vlm(self):
        """Force the next response to use the vision generator."""
        self._force_vlm = True

    def get_response(
        self,
        history: ConversationHistory,
        inst: str | None = None,
        p_data: GenData | None = None,
    ) -> GeneratedResponse:
        if self._force_vlm:
            return self.vlm_generator.get_response(history, inst, p_data)

        res, data = self.chooser.formatted(
            self.instructions,
            history.parts,
            format_t=self._Response,
        )
        if p_data is not None:
            data.previous_data = p_data

        if res.needs_vlm:
            return self.vlm_generator.get_response(history, inst, data)
        return self.llm_generator.get_response(history, inst, data)

    class _Response(BaseModel):
        needs_vlm: bool


def sentence_chunk(generator: Generator[str]) -> Generator[str]:
    """Split a streamed response into sentence-sized chunks."""
    stop_regex: str = r"^(.*[.?!\n])(.*)$"
    buffer = ""
    for text in generator:
        buffer += text
        match = re.match(stop_regex, buffer)
        if match:
            yield match.group(1).strip()
            buffer = match.group(2)
    if buffer:
        yield buffer.strip()
