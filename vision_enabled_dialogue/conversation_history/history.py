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

import json
import logging
from collections.abc import Callable
from threading import Condition, Thread

from .parts import ConversationPart
from .strategies import GrowthStrategy

StrategyFactory = Callable[[], GrowthStrategy]
"""Factory type for instantiating new growth strategies."""


class ConversationHistory:
    """Store conversation parts and compact them with background strategies.

    This notifies on updates and full updates. The strategy classes are
    instantiated whenever the history is (re)set.
    """

    strategy_factories: list[StrategyFactory]
    """Methods to create new growth strategies for the history when needed."""
    on_update: Callable[[ConversationPart], None]
    """Called when a new part is added to the history."""
    on_full_update: Callable[[list[ConversationPart]], None]
    """Called when the full history is replaced, with the new content."""

    _condition: Condition
    _worker_thread: Thread
    _content: list[ConversationPart]
    _strategies: list[GrowthStrategy]
    _logger: logging.Logger

    def __init__(
        self,
        initial_content: list[ConversationPart] | None = None,
        strategy_factories: list[StrategyFactory] | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize the conversation history with optional initial content.

        :param initial_content: The initial conversation parts.
        :param strategy_factories: Functions that return growth strategies.
        :param logger: Optional logger for debug output.
        """
        self.strategy_factories = strategy_factories or []
        self.on_update = lambda _: None
        self.on_full_update = lambda _: None

        self._condition = Condition()
        self._worker_thread = Thread(target=self._apply_strategies, daemon=True)
        self._content = []
        self._strategies = []
        self._logger = logger or logging.getLogger(__name__)

        self._worker_thread.start()
        self.set(initial_content or [], True)

    @property
    def parts(self) -> list[ConversationPart]:
        """Return a copy of the current conversation parts."""
        with self._condition:
            return self._content.copy()

    def set(self, content: list[ConversationPart], notify_append: bool = False) -> None:
        """Replace the conversation history with new content.

        :param content: The new conversation parts.
        :param notify_append: Whether to emit append callbacks for each part.
        """
        self._logger.debug(f"Setting conversation history with {len(content)} parts")
        with self._condition:
            self._content = []
            self._strategies = [s() for s in self.strategy_factories]
            for part in content:
                self.append(part, notify=notify_append)
            self.on_full_update(self._content)
        self._logger.debug("Finished setting conversation history")

    def append(self, part: ConversationPart, notify: bool = True) -> None:
        """Append a conversation part and notify any active strategies.

        :param part: The part to append.
        :param notify: Whether to emit the update callback.
        """
        self._logger.debug(f"Appending part to conversation history: {part}")
        with self._condition:
            self._content.append(part)
            for strategy in self._strategies:
                if strategy.notify(part):
                    self._logger.debug(f"Strategy {strategy} notified")
                    self._condition.notify()
            if notify:
                self.on_update(part)

    def save(self, path: str) -> None:
        """Write the current conversation history to a JSON file.

        :param path: The file path to write.
        """
        self._logger.debug(f"Saving conversation history to {path}")
        with self._condition:
            with open(path, "w", encoding="utf-8") as f:
                contents = {"contents": [part.to_dict() for part in self._content]}
                json.dump(contents, f, indent=2)
        self._logger.debug(f"Finished saving conversation history to {path}")

    def _apply_strategies(self) -> None:
        self._logger.debug("Starting apply strategies")
        with self._condition:
            while True:
                to_apply = next((s for s in self._strategies if s.should_apply()), None)
                if to_apply is None:
                    self._condition.wait()
                    continue
                self._logger.debug(f"Applying strategy {to_apply}")
                new_content = to_apply.apply(self._content)
                self.set(new_content)
