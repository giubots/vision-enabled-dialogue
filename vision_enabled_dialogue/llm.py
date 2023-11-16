import time
from openai import OpenAI


class LLM:
    def query(self, prompt: list) -> str:
        raise NotImplementedError()


class GPT4(LLM):
    client: OpenAI

    def __init__(self):
        self.client = OpenAI()

    def query(self, prompt) -> str:
        params = {
            "model": "gpt-4-vision-preview",
            "messages": prompt,
            "max_tokens": 200,
        }
        result = self.client.chat.completions.create(**params)
        return result.choices[0].message.content  # type: ignore


class Mock(LLM):
    def query(self, prompt) -> str:
        time.sleep(2)
        return "This is a mock answer."
