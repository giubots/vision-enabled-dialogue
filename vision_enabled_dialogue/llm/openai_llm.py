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
import os
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from openai import OpenAI, Stream
from openai.types.responses import ResponseStreamEvent

from .llm import LLM, GenData, OpenAIFormattable, T


class GPT(LLM):
    """OpenAI client that streams responses from the configured model.

    This is compatible with local Ollama API endpoints and the OpenAI API.
    """

    model: str
    """The model to use for generation."""
    temperature: float
    """The temperature to use for generation."""
    tools_desc: list[dict]
    """The description of the tools available for function calls."""
    tools_impl: dict[str, Callable]
    """The implementation of the tools available for function calls."""

    _client: OpenAI
    _executor: ThreadPoolExecutor

    def __init__(
        self,
        model: str = "gemma3:27b",
        temperature: float = 0.7,
        tools_desc: list[dict] | None = None,
        tools_impl: dict[str, Callable] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize the GPT client with model settings and optional tools.

        :param model: The model to use for generation.
        :param temperature: The temperature to use for generation.
        :param tools_desc: The description of the tools available for function
            calls.
        :param tools_impl: The implementation of the tools available for
            function calls.
        :param base_url: The base URL for the OpenAI API, or None to use the
            default or environment variable; use "" to disable the base URL.
        :param api_key: The API key for the OpenAI API.
        :param logger: Optional logger for debug output."""
        super().__init__(logger)
        self.model = model
        self.temperature = temperature
        self.tools_desc = tools_desc or []
        self.tools_impl = tools_impl or {}

        api_key = api_key or os.environ.get("OPENAI_API_KEY", "ollama")
        client_params = {'api_key': api_key}
        if base_url is None:
            base_url = os.environ.get("OPENAI_API_BASE_URL", "http://localhost:11434/v1")
        if base_url:
            client_params['base_url'] = base_url
        self._client = OpenAI(**client_params)
        self._executor = ThreadPoolExecutor(max_workers=3)

    def _data_impl(
        self,
        i: str,
        p: list[OpenAIFormattable],
        a: str | None,
    ) -> GenData:
        super()._data_impl(i, p, a)
        return GenData(
            f"Model: {self.model}\nTemperature: {self.temperature}",
            i,
            [mm for m in p for mm in m.to_openai()],
            a,
            None,
        )

    def _query_impl(
        self,
        instructions: str,
        messages: list[OpenAIFormattable],
        final_instruction: str | None = None,
    ) -> Generator[str]:
        super()._query_impl(instructions, messages, final_instruction)
        m_input = [
            {"role": "user", "content": [mm for m in messages for mm in m.to_openai()]}
        ]
        if final_instruction is not None:
            m_input.append({"role": "user", "content": final_instruction})

        stream: Stream[ResponseStreamEvent] = self._client.responses.create(
            model=self.model,
            temperature=self.temperature,
            instructions=instructions,
            input=m_input,  # type: ignore
            stream=True,
            store=False,
        )

        def stream_handler(stream: Stream[ResponseStreamEvent]) -> Generator[str]:
            for event in stream:
                match event.type:
                    case "response.failed" | "response.incomplete" | "error":
                        self._logger.warning(f"Received error event: {event}")
                    case "response.output_text.delta":
                        self._logger.debug(f"Received text delta: {event.delta}")
                        yield event.delta
                    case "response.output_item.done" if (
                        event.item.type == "function_call"
                    ):
                        self._function_call(event.item.name, event.item.arguments)

        return stream_handler(stream)

    def _function_call(self, name: str, args: str):
        if name not in self.tools_impl:
            self._logger.warning(f"Function call to unknown tool: {name}")
            return
        try:
            args_dict = json.loads(args)
        except json.JSONDecodeError as e:
            self._logger.warning(
                f"Failed to decode function call arguments: {args}. Error: {e}"
            )
            return
        with self._executor as executor:
            self._logger.info(f"Executing tool: {name} with arguments: {args_dict}")
            executor.submit(self.tools_impl[name], **args_dict)

    def _formatted_impl(
        self,
        instructions: str,
        messages: list[OpenAIFormattable],
        format_t: type[T],
    ) -> T:
        super()._formatted_impl(instructions, messages, format_t)
        formatted_messages = [mm for m in messages for mm in m.to_openai()]

        response = self._client.responses.parse(
            model=self.model,
            temperature=0,  # FIXME: this is not in the data
            instructions=instructions,
            input=[{"role": "user", "content": formatted_messages}],  # type: ignore
            store=False,
            text_format=format_t,
        )

        return response.output_parsed  # type: ignore
