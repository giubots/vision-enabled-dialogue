import argparse
import base64
import logging
import threading
import time
from collections.abc import Iterable
from concurrent.futures import Future

import cv2

from vision_enabled_dialogue import DialogueManager
from vision_enabled_dialogue.conversation_history import (
    ConversationHistory,
    HidingSummariser,
    KeepLastNStrategy,
    MixedSummariser,
    SummariseHeavy,
)
from vision_enabled_dialogue.llm import GPT
from vision_enabled_dialogue.response_gen import VLMResponseGenerator


def send_cam(on_frame, stop, period=5):
    """Sends an image from the webcam to the on_frame callback, once every 'period' seconds."""
    vid = cv2.VideoCapture(0)
    while not stop.is_set():
        _, frame = vid.read()
        _, buffer = cv2.imencode(".jpg", frame)
        string64 = base64.b64encode(buffer).decode("utf-8")  # type: ignore
        on_frame(string64, time.time())
        print("Frame sent")
        time.sleep(period)


def dialogue_interactive(dm: DialogueManager):
    """Start an interactive dialogue with the user."""
    try:
        while True:
            text = input("You: ")
            if text:
                dm.add_turn(text, time.time())
                answer = dm.generate_response()
                print("AI: ", answer)
    except KeyboardInterrupt:
        pass


def speak(text_generator: Iterable[str]) -> Future[str]:
    """A simple speak function that prints the text to the console.

    Note that this implementation is synchronous, so it will block until the
    text is fully printed. In a real application, you would want to implement
    this function to speak the text asynchronously and return a Future that
    resolves when the speaking is done.
    """
    spoken_text = []
    for text in text_generator:
        spoken_text.append(text)
        print(text, end="")
    print()
    future = Future()
    future.set_result(" ".join(spoken_text))
    return future


if __name__ == "__main__":

    # Setup arguments
    parser = argparse.ArgumentParser(description="vision_enabled_dialogue")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Whether the conversation is multimodal (with video) or text-only.",
    )
    args = parser.parse_args()

    # Setup dialogue manager
    dialogue_manager = DialogueManager(speak=speak)

    # A more in-depth example (to be customised before use)
    conv_summariser_llm = GPT()  # Use a fast model here
    response_gen = VLMResponseGenerator(
        instructions="Your instructions here",
        vlm=GPT(),  # use a vision-enabled model here
    )
    advanced = DialogueManager(
        speak=speak,
        history=ConversationHistory(
            strategy_factories=[
                lambda: SummariseHeavy(
                    max_heavy=4,
                    min_heavy=4,
                    summariser_factories=[HidingSummariser],
                ),
                lambda: KeepLastNStrategy(
                    keep=10,
                    threshold=20,
                    summariser_factories=[
                        lambda: MixedSummariser(llm=conv_summariser_llm)
                    ],
                ),
            ]
        ),
        response_generator=response_gen,
        persistent_steering=False,
    )

    # Start video capture if enabled
    if not args.no_video:
        print("Video mode enabled")
        stopped = threading.Event()
        t_args = (dialogue_manager.add_frame, stopped)
        thread = threading.Thread(target=send_cam, args=t_args)
        thread.start()

    # Start dialogue
    dialogue_interactive(dialogue_manager)

    # Dialogue exited, stop video capture
    if not args.no_video:
        stopped.set()
        print("Waiting for thread to stop")
        thread.join()
