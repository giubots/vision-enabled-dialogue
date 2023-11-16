from threading import Lock
from typing import List, Tuple

from vision_enabled_dialogue.llm import LLM
from vision_enabled_dialogue.messages import (
    Message,
    FrameMessage,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    FSummaryMessage,
)


class Conversation:
    """Handles the conversation and summarisation of frames."""

    llm: LLM
    """The LLM that generates the responses and summaries."""
    fr_buff_size: int
    """The number of frames to buffer before summarising."""
    fr_recap: int
    """The maximum number of frames to summarise."""

    _fr_count: int
    _messages: List[Message]
    _lock: Lock

    def __init__(self, llm: LLM, fr_buff_size: int = 4, fr_recap: int = 3):
        self.llm = llm
        self.fr_buff_size = fr_buff_size
        self.fr_recap = fr_recap
        self._fr_count = 0
        self._messages = []
        self._lock = Lock()

    def add_text(self, text) -> str:
        """Add a user message and return the response."""

        with self._lock:
            self._messages.append(UserMessage(text))

            prompt = [
                SystemMessage(
                    "You are impersonating a friendly kid. "
                    "In this conversation, what you see is represented by the images. "
                    "For example, the images will show you the environment you are in and possibly the person you are talking to. "
                    "Try to start the conversation by saying something about the person you are talking to if there is one, based on accessories, clothes, etc. "
                    "If there is no person, try to say something about the environment, but do not describe the environment! "
                    "Have a nice conversation and try to be curious! "
                    "It is important that you keep your answers short and to the point. "
                    "DO NOT INCLUDE EMOTICONS OR SMILEYS IN YOUR ANSWERS. ",
                ),
                *self._messages,
            ]
            prompt = [m.gpt_format() for m in prompt]
            answer = self.llm.query(prompt)

            assistantMessage = AssistantMessage(answer)
            self._messages.append(assistantMessage)
        return answer

    def add_frame(self, frame):
        """Add a frame to the conversation, summarise if necessary."""

        with self._lock:
            self._messages.append(FrameMessage(frame))
            self._fr_count += 1

            if self._fr_count >= self.fr_buff_size:
                new_messages, removed = self.get_fr_summary()
                self._messages = new_messages
                self._fr_count -= removed
                self.last = removed

    def get_fr_summary(self) -> Tuple[List[Message], int]:
        """Summarise the frames and return the new messages and the number of frames removed."""

        # Assuming frames > fr_buff_size > fr_recap
        # Assuming called with lock
        first_fr = None
        i = None
        for i, m in enumerate(self._messages):
            if m.is_frame():
                if first_fr is None:
                    first_fr = i
            if first_fr is not None and (m.is_user() or i - first_fr >= self.fr_recap):
                break
        before = self._messages[:first_fr]
        to_summarise = self._messages[first_fr:i]
        after = self._messages[i:]

        prompt = [
            SystemMessage(
                "These are frames from a video. Summarise what's happening in the video in one sentence. "
                "The frames are preceded by a context to help you summarise the video. "
                "Summarise only the frames, not the context."
                "The images can be repeating, this is normal, do not point this out in the description."
                "Respond with only the summary in one sentence. This is very important. "
                "Do not include warnings or other messages."
            ).gpt_format(),
            *[b.gpt_format() for b in before],
            *[s.gpt_format() for s in to_summarise],
        ]
        summary = self.llm.query(prompt)

        messages = [
            *before,
            FSummaryMessage(summary),
            *after,
        ]

        return messages, i - first_fr  # type: ignore

    def __str__(self) -> str:
        with self._lock:
            return "\n".join([str(m) for m in self._messages])
