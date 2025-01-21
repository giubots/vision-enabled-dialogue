import base64
import threading

import zmq
from furhat_remote_api import FurhatRemoteAPI

from vision_enabled_dialogue.conversation import Conversation
from vision_enabled_dialogue.llm import GPT

FURHAT_IP = "YOUR-FURHAT-IP-HERE"


def send_furhat(on_frame, stopped, detection_period=50):
    print("Starting video capture")
    context = zmq.Context()
    fh_socket = context.socket(zmq.SUB)
    fh_socket.connect(f"tcp://{FURHAT_IP}:3000")
    fh_socket.subscribe("")

    # Only read the last message to avoid lagging behind the stream.
    fh_socket.setsockopt(zmq.RCVHWM, 1)
    fh_socket.setsockopt(zmq.CONFLATE, 1)

    iterations = 0
    while not stopped.is_set():
        string = fh_socket.recv()
        magicnumber = string[0:3]

        # check if we have a JPEG image (starts with ffd8ff)
        if magicnumber == b"\xff\xd8\xff":
            if iterations % detection_period == 0:
                string64 = base64.b64encode(string).decode("utf-8")
                print("Frame received")
                on_frame(string64)
            iterations += 1
    print("Stopping video capture")


def dialogue_furhat(conversation: Conversation):
    def conv_io(conversation: Conversation):
        furhat = FurhatRemoteAPI(FURHAT_IP)
        furhat.set_voice(name="Matthew")

        try:
            while True:
                furhat.attend(user="CLOSEST")
                furhat.set_led(red=0, green=0, blue=255)
                result = furhat.listen().message  # type: ignore
                furhat.set_led(red=0, green=0, blue=0)
                if result:
                    print("You:", result)
                    furhat.gesture(name="GazeAway")
                    furhat.attend(location="1,-1,5")
                    answer = conversation.add_text(result)
                    print("AI:", answer)
                    furhat.attend(user="CLOSEST")
                    furhat.say(text=answer, blocking=True)
                else:
                    print("No text detected")
        except Exception as e:
            print(e)
        finally:
            furhat.attend(user=None)

    t = threading.Thread(target=conv_io, args=(conversation,))
    t.daemon = True
    t.start()

    try:
        while True:
            input("")
            print("Forcing VLM on next question")
            conversation.force_vlm()
    except KeyboardInterrupt:
        print("Stopping dialogue")


if __name__ == "__main__":
    try:
        conversation = Conversation(
            vlm=GPT(model="gpt-4o-mini-2024-07-18"),
            llm=GPT(model="gpt-3.5-turbo-0125"),
            model_chooser=GPT(model="gpt-3.5-turbo-0125"),
        )

        # Start video capture
        stopped = threading.Event()
        t_args = (conversation.add_frame, stopped)
        thread = threading.Thread(target=send_furhat, args=t_args)
        thread.start()

        # Start dialogue
        dialogue_furhat(conversation)
    except Exception as e:
        print(e)
    finally:
        # Dialogue exited, stop video capture
        stopped.set()
        print("Waiting for thread to stop")
        thread.join()
