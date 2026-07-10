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

import threading
from collections.abc import Callable, Generator
from concurrent.futures import Future

from .conversation_history import (
    ConversationHistory,
    Frame,
    HidingSummariser,
    SummariseHeavy,
    Thought,
    Turn,
)
from .conversation_history.parts import GeneratedTurn
from .llm import GPT
from .response_gen import ResponseGenerator, VLMResponseGenerator, sentence_chunk

InterruptableSpeechAction = Callable[[Generator[str]], Future[str]]
"""Show a stream of text and returning what was actually spoken."""


class DialogueManager:
    """Manage conversation history, steering, and spoken responses."""

    speak: InterruptableSpeechAction
    """Called to speak a generated response."""
    history: ConversationHistory
    """The conversation history."""
    response_generator: ResponseGenerator
    """The generator for creating responses."""
    persistent_steering: bool
    """Whether to persist steering instructions across turns."""

    _next_steering: str | None
    _steering_lock = threading.Lock()

    def __init__(
        self,
        speak: InterruptableSpeechAction,
        history: ConversationHistory | None = None,
        response_generator: ResponseGenerator | None = None,
        persistent_steering: bool = False,
    ):
        """Creates a new dialogue manager with a given speech action.

        To add turns or frames to the conversation history, use the `add_turn`
        and `add_frame`. Inner thoughts, not part of the conversation, can be
        added with `add_thought`.

        Additional steering instructions for the next generated response can be
        added with `add_steering`. These are kept for one turn only, unless
        `persistent_steering` is set to True. When generating a response with
        `generate_response`, the provided steering will override any stored
        steering. If no steering is provided, the stored steering will be used.
        If no stored steering is available, no steering will be used. Steering
        instructions are never stored in the conversation history, but they are
        part of the metadata of the generated response.

        If no conversation history is provided, a new one will be created with a
        default summarisation strategy that summarises heavy turns with a hiding
        summariser, keeping exactly 4 heavy turns in the history. If no response
        generator is provided, a new one will be created using a the default GPT
        vision model.

        :param speak: A function to speak a generated response.
        :param history: Optional conversation history to use.
        :param response_generator: Optional response generator to use.
        :param persistent_steering: Whether to persist steering instructions
            across turns.
        """
        self.speak = speak
        self.history = history or ConversationHistory(
            strategy_factories=[
                lambda: SummariseHeavy(
                    summariser_factories=[HidingSummariser],
                    max_heavy=4,
                    min_heavy=4,
                ),
            ]
        )
        self.response_generator = response_generator or VLMResponseGenerator(GPT())
        self.persistent_steering = persistent_steering

        self._next_steering = None
        self._steering_lock = threading.Lock()

    def add_frame(self, frame_base64: str, stamp: float) -> None:
        """Append a camera frame to the conversation history.

        :param frame_base64: The frame encoded as base64 JPEG data.
        :param stamp: The frame timestamp.
        """
        self.history.append(Frame(stamp, frame_base64))

    def add_turn(self, text: str, stamp: float, speaker: str = "human") -> None:
        """Append a spoken turn to the conversation history.

        :param text: The spoken text.
        :param stamp: The turn timestamp.
        :param speaker: The speaker label.
        """
        self.history.append(Turn(stamp, text, speaker))

    def add_thought(self, text: str, stamp: float) -> None:
        """Append an internal thought to the conversation history.

        :param text: The thought text.
        :param stamp: The thought timestamp.
        """
        self.history.append(Thought(stamp, text))

    def add_steering(self, steering: str) -> None:
        """Store steering for the next generated response.

        :param steering: The steering instruction.
        """
        with self._steering_lock:
            self._next_steering = steering

    def generate_response(
        self,
        speaker: str = "assistant",
        steering: str | None = None,
        response_generator: ResponseGenerator | None = None,
    ) -> None:
        """Generate a response and outputs it with the `speak` action.

        Additional steering instructions for the next generated response can be
        added with the `steering` parameter. If no steering is provided, the
        stored steering will be used, if any. Steering instructions are never
        stored in the conversation history, but they are part of the metadata of
        the generated response.

        :param speaker: The speaker label for the generated turn.
        :param steering: Optional steering instruction for this turn.
        :param response_generator: Optional response generator override.
        """
        if steering is None:
            with self._steering_lock:
                steering = self._next_steering
                if not self.persistent_steering:
                    self._next_steering = None
        if response_generator is None:
            response_generator = self.response_generator
        response = response_generator.get_response(self.history, steering)

        def on_done(future: Future[str]) -> None:
            spoken = future.result()
            self.history.append(
                GeneratedTurn(response.stamp, spoken, speaker, response.gen_data)
            )

        self.speak(sentence_chunk(response.response)).add_done_callback(on_done)
