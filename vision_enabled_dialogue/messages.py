class Message:
    content: str

    def is_frame(self):
        return False

    def is_user(self):
        return False

    def is_system(self):
        return False

    def is_assistant(self):
        return False

    def is_f_summary(self):
        return False

    def gpt_format(self):
        raise NotImplementedError

    def __str__(self) -> str:
        return self.content


class FrameMessage(Message):
    def __init__(self, content):
        self.content = content

    def is_frame(self):
        return True

    def gpt_format(self):
        return {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + self.content},
                }
            ],
        }

    def __str__(self) -> str:
        return "FrameMessage"


class SystemMessage(Message):
    def __init__(self, content):
        self.content = content

    def is_system(self):
        return True

    def gpt_format(self):
        return {"role": "system", "content": self.content}

    def __str__(self) -> str:
        return "Answer: " + self.content


class UserMessage(Message):
    def __init__(self, content):
        self.content = content

    def is_user(self):
        return True

    def gpt_format(self):
        return {
            "role": "user",
            "content": [{"type": "text", "text": self.content}],
        }


class AssistantMessage(Message):
    def __init__(self, content):
        self.content = content

    def is_assistant(self):
        return True

    def gpt_format(self):
        return {
            "role": "assistant",
            "content": [{"type": "text", "text": self.content}],
        }


class FSummaryMessage(Message):
    def __init__(self, content):
        self.content = "Image summary: " + content

    def is_f_summary(self):
        return True

    def gpt_format(self):
        return {
            "role": "user",
            "content": [{"type": "text", "text": self.content}],
        }
