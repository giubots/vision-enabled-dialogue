import threading
import base64

from furhat_remote_api import FurhatRemoteAPI
import zmq

from vision_enabled_dialogue.conversation import Conversation
from vision_enabled_dialogue.llm import GPT4


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


def dialogue_furhat(on_message):
    print("Starting dialogue")
    furhat = FurhatRemoteAPI(FURHAT_IP)
    furhat.set_voice(name="Matthew")
    
    try:
        while True:
            furhat.attend(user="CLOSEST")
            result = furhat.listen().message  # type: ignore
            if result:
                print(result)
                furhat.gesture(name="GazeAway")
                furhat.attend(location="0,10,0")
                answer = on_message(result)
                furhat.attend(user="CLOSEST")
                furhat.say(text=answer, blocking=True)
            else:
                print("No text detected")
    except (Exception, KeyboardInterrupt) as e:
        print(e)
    print("Stopping dialogue")


if __name__ == "__main__":
    try:
        conversation = Conversation(GPT4())

        # Start video capture
        stopped = threading.Event()
        t_args = (conversation.add_frame, stopped)
        thread = threading.Thread(target=send_furhat, args=t_args)
        thread.start()

        # Start dialogue
        dialogue_furhat(conversation.add_text)
    except Exception as e:
        print(e)
    finally:
        # Dialogue exited, stop video capture
        stopped.set()
        print("Waiting for thread to stop")
        thread.join()
