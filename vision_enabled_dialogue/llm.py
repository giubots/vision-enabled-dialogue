import time

from openai import OpenAI

from vision_enabled_dialogue.messages import Message


class LLM:
    debug: bool

    def __init__(self, debug: bool = False):
        self.debug = debug

    def query(self, messages: list[Message]) -> str:
        raise NotImplementedError()


class GPT(LLM):
    client: OpenAI
    model: str

    def __init__(self, debug: bool = False, model: str = "gpt-4o-mini-2024-07-18"):
        super().__init__(debug)
        self.client = OpenAI()
        self.model = model

    def query(self, messages: list[Message]) -> str:
        if self.debug:
            print("\033[94m")
            print(*messages, sep="\n")
            print("\033[0m")
        prompt = [m.gpt_format() for m in messages]
        params = {
            "model": self.model,
            "messages": prompt,
            "max_tokens": 200,
        }
        result = self.client.chat.completions.create(**params)
        return result.choices[0].message.content  # type: ignore


class Mock(LLM):
    def query(self, messages) -> str:
        time.sleep(2)
        return "This is a mock answer."


class MockChooser(LLM):
    def query(self, messages: list) -> str:
        return "VLM"
