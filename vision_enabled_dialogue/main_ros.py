import base64

import rclpy
from hri_msgs.msg import LiveSpeech
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.subscription import Subscription
from sensor_msgs.msg import Image
from tts_msgs.action import TTS

from vision_enabled_dialogue.conversation import Conversation
from vision_enabled_dialogue.llm import GPT


class VisionEnabledDialogue(Node):
    conversation: Conversation
    text_sub: Subscription
    cam_sub: Subscription
    text_client: ActionClient

    def __init__(self):
        super().__init__("vision_enabled_dialogue")
        self.conversation = Conversation(
            vlm=GPT(model="gpt-4o-mini-2024-07-18"),
            llm=GPT(model="gpt-3.5-turbo-0125"),
            model_chooser=GPT(model="gpt-3.5-turbo-0125"),
            add_behaviour="keep_latest",
        )
        self.text_sub = self.create_subscription(
            LiveSpeech,
            "/humans/voices/anonymous_speaker/speech",
            self.send_text,
            10,
        )
        self.cam_sub = self.create_subscription(
            Image,
            "/camera1/image_raw",
            self.send_cam,
            10,
        )
        self.text_client = ActionClient(self, TTS, "/tts_engine/tts")

    def send_text(self, msg):
        text = msg.final
        print("You: ", text)
        answer = self.conversation.add_text(text)
        print("AI: ", answer)
        tts_msg = TTS.Goal()
        tts_msg.input = answer
        self.text_client.wait_for_server()
        self.text_client.send_goal_async(tts_msg)

    def send_cam(self, msg):
        string64 = base64.b64encode(msg.data).decode("utf-8")  # type: ignore
        self.conversation.add_frame(string64)
        print("Frame sent")

    def force_vlm(self):
        self.conversation.force_vlm()


def main(args=None):
    rclpy.init(args=args)
    vision_enabled_dialogue = VisionEnabledDialogue()
    try:
        rclpy.spin(vision_enabled_dialogue)
    except KeyboardInterrupt:
        pass
    vision_enabled_dialogue.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
