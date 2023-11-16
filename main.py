import json
import threading
import base64
import time
import cv2
import argparse


from vision_enabled_dialogue.conversation import Conversation
from vision_enabled_dialogue.llm import GPT4


def send_vid(on_frame, stopped, source, period=5):
    """Sends the 'source' video frame by frame to the on_frame callback, once every 'period' seconds."""
    vid = cv2.VideoCapture(source)
    while vid.isOpened():
        success, frame = vid.read()
        if stopped.is_set() or not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)

        string64 = base64.b64encode(buffer).decode("utf-8")  # type: ignore
        on_frame(string64)
        print("Frame sent")
        time.sleep(period)
    vid.release()


def send_cam(on_frame, stopped, period=5):
    """Sends an image from the webcam to the on_frame callback, once every 'period' seconds."""
    vid = cv2.VideoCapture(0)
    while not stopped.is_set():
        _, frame = vid.read()
        _, buffer = cv2.imencode(".jpg", frame)
        string64 = base64.b64encode(buffer).decode("utf-8")  # type: ignore
        on_frame(string64)
        print("Frame sent")
        time.sleep(period)


def dialogue_interactive(on_message):
    """Start an interactive dialogue with the user."""
    try:
        while True:
            text = input("You: ")
            if text:
                answer = on_message(text)
                print("AI: ", answer)
    except KeyboardInterrupt:
        pass


def dialogue_script(on_message, script, delay=3):
    """Start a dialogue with the user using a script."""
    for text in script:
        time.sleep(delay)
        print("Script: ", text)
        answer = on_message(text)
        print("AI: ", answer)


if __name__ == "__main__":
    # Setup arguments
    parser = argparse.ArgumentParser(description="vision_enabled_dialogue")
    parser.add_argument(
        "--video",
        type=str,
        dest="source_path",
        help="Source of video (path to video file)",
    )
    parser.add_argument(
        "--script",
        type=str,
        dest="script_path",
        help="Script to use for dialogue (path to json array of strings)",
    )
    args = parser.parse_args()

    conversation = Conversation(GPT4())

    # Start video capture
    stopped = threading.Event()
    if args.source_path:
        print("Using video source", args.source_path)
        t_args = (conversation.add_frame, stopped, args.source_path)
        thread = threading.Thread(target=send_vid, args=t_args)
    else:
        t_args = (conversation.add_frame, stopped)
        thread = threading.Thread(target=send_cam, args=t_args)
    thread.start()

    # Start dialogue
    if args.script_path:
        print("Using script", args.script_path)
        with open(args.script_path, "r") as f:
            script = json.load(f)
        dialogue_script(conversation.add_text, script)
    else:
        dialogue_interactive(conversation.add_text)

    # Dialogue exited, stop video capture
    stopped.set()
    print("Waiting for thread to stop")
    thread.join()
