from threading import Lock
from typing import Literal, Tuple

from vision_enabled_dialogue.llm import LLM
from vision_enabled_dialogue.messages import (
    AssistantMessage,
    FrameMessage,
    FSummaryMessage,
    Message,
    SystemMessage,
    UserMessage,
)


class Conversation:
    """Handles the conversation and summarisation of frames."""

    vlm: LLM
    """The LLM that generates the responses and summaries."""
    llm: LLM
    """LLM used when the VLM is not needed."""
    model_chooser: LLM
    """The LLM that chooses whether to use the VLM or the LLM."""
    fr_buff_size: int
    """The number of frames to buffer before summarising."""
    fr_recap: int
    """The maximum number of frames to summarise."""

    _fr_count: int
    _messages: list[Message]
    _lock: Lock
    _force_vlm: bool
    _last_frame: FrameMessage
    _add_behaviour: callable

    def __init__(
        self,
        vlm: LLM,
        llm: LLM,
        model_chooser: LLM,
        fr_buff_size: int = 4,
        fr_recap: int = 3,
        add_behaviour: Literal["keep_history", "keep_latest"] = "keep_history",
    ):
        """Initialise the conversation with the LLMs and the frame summarisation settings.

        The conversation can be configured to keep a history of frames (add_behaviour="keep_history") or only the latest frame (add_behaviour="keep_latest").
        Keeping a history will summarise the frames when necessary, while keeping only the latest frame will not summarise the frames and ignore fr_buff_size and fr_recap.

        :param vlm: The LLM that generates the responses and summaries.
        :param llm: LLM used when the VLM is not needed.
        :param model_chooser: The LLM that chooses whether to use the VLM or the LLM.
        :param fr_buff_size: The number of frames to buffer before summarising.
        :param fr_recap: The maximum number of frames to summarise.
        :param add_behaviour: The behaviour to use when adding frames.
        """
        self.vlm = vlm
        self.llm = llm
        self.model_chooser = model_chooser
        if add_behaviour == "keep_history":
            self._add_behaviour = self._add_behaviour_keep_history
        elif add_behaviour == "keep_latest":
            self._add_behaviour = self._add_behaviour_keep_latest
        else:
            raise ValueError("Invalid add_behaviour")
        self.fr_buff_size = fr_buff_size
        self.fr_recap = fr_recap
        self._fr_count = 0
        self._messages = []
        self._lock = Lock()
        self._force_vlm = False
        self._last_frame = None

    def add_text(self, text: str, model_choice: Literal["VLM", "LM"] = None) -> str:
        """Add a user message and return the response."""
        message = UserMessage(text)

        if self._force_vlm:
            print("Forcing VLM")
            model_choice = "VLM"
            self._force_vlm = False
        elif model_choice is None:
            while model_choice not in ["VLM", "LM"]:
                print("Prompting for model choice")
                model_choice_prompt = [
                    SystemMessage(
                        "You are in charge of choosing the best model for the current conversation. "
                        "You can choose between the Vision Language Model (VLM) and the Language Model (LM). "
                        "The VLM is a model that can understand images and text, and can generate text and image descriptions. "
                        "The LM is a model that can only understand text and generate text. "
                        "If the provided text is about colors, what something looks like, holding objects, appearance, the environment and so on, you will choose VLM. "
                        "If the user asks how does something look like, what am I holding, and similar, then use a VLM. "
                        "If the text is about abstract things, conversational, and in all those cases where it cannot benefit from an image input, use the LM. "
                        "The LM is faster and cheaper, and should be the preferred choice, unless the input could benefit from using image input. "
                        "Answer with 'VLM' or 'LM', based on the following prompt."
                    ),
                    message,
                ]
                model_choice = self.model_chooser.query(model_choice_prompt)
        print("Using model:", model_choice)

        with self._lock:
            self._messages.append(message)

            if model_choice == "VLM":
                prompt: list[Message] = [
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
                if self._last_frame is not None:
                    prompt.append(self._last_frame)
                answer = self.vlm.query(prompt)

            else:
                prompt: list[Message] = [
                    SystemMessage(
                        "You are impersonating a friendly kid. "
                        "Have a nice conversation and try to be curious! "
                        "It is important that you keep your answers short and to the point. "
                        "DO NOT INCLUDE EMOTICONS OR SMILEYS IN YOUR ANSWERS. ",
                    ),
                    *[m for m in self._messages if not m.is_frame()],
                ]
                answer = self.llm.query(prompt)

            self._messages.append(AssistantMessage(answer))
        return answer

    def add_frame(self, frame):
        """Add a frame to the conversation, summarise if necessary."""

        with self._lock:
            self._add_behaviour(frame)

    def force_vlm(self):
        """Force the next message to be processed by the VLM."""
        self._force_vlm = True

    def _add_behaviour_keep_history(self, string64: str):
        # Call with lock
        self._messages.append(FrameMessage(string64))
        self._fr_count += 1
        if self._fr_count >= self.fr_buff_size:
            new_messages, removed = self.get_fr_summary()
            self._messages = new_messages
            self._fr_count -= removed

    def _add_behaviour_keep_latest(self, string64: str):
        # Call with lock
        self._last_frame = FrameMessage(string64)

    def get_fr_summary(self) -> Tuple[list[Message], int]:
        """Summarise the frames and return the new messages and the number of frames removed."""

        # Assuming frames > fr_buff_size > fr_recap
        # Assuming called with lock
        first_fr = None
        i = None
        for i, m in enumerate(self._messages):
            if m.is_frame() and first_fr is None:
                first_fr = i
            if first_fr is not None and (not m.is_frame() or i - first_fr >= self.fr_recap):
                break
        before = self._messages[:first_fr]
        to_summarise = self._messages[first_fr:i]
        after = self._messages[i:]

        prompt = [
            SystemMessage(
                "These are frames from a video. Summarise what's happening in the video in one sentence. "
                "The frames are preceded by a context to help you summarise the video. "
                "Summarise only the frames, not the context. "
                "The images can be repeating, this is normal, do not point this out in the description. "
                "Respond with only the summary in one sentence. This is very important. "
                "Do not include warnings or other messages."
            ),
            *before,
            *to_summarise,
        ]
        summary = self.vlm.query(prompt)

        messages = [
            *before,
            FSummaryMessage(summary),
            *after,
        ]

        return messages, i - first_fr  # type: ignore

    def get_conv_summary(self) -> str:
        """Use the LLM to summarise the conversation into a paragraph and return it."""

        prompt = [
            SystemMessage("Summarise the following conversation in one paragraph. "),
            *[m for m in self._messages if not m.is_frame()],
        ]
        summary = self.llm.query(prompt)
        return summary

    def __str__(self) -> str:
        with self._lock:
            return "\n".join([str(m) for m in self._messages])
