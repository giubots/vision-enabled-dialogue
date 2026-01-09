import base64

import cv2
import rclpy
from cv_bridge import CvBridge
from hri_msgs.msg import LiveSpeech
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from rclpy.subscription import Subscription
from sensor_msgs.msg import Image, CompressedImage
from vision_enabled_dialogue.conversation import Conversation
from vision_enabled_dialogue.llm import GPT

from tts_msgs.action import TTS


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
        self.cam_sub = self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self.send_cam_compressed,
            qos_profile=QoSPresetProfiles.SENSOR_DATA.value,
            callback_group=MutuallyExclusiveCallbackGroup(),
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

    def send_cam_compressed(self, msg):
        image = CvBridge().compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        _, img_encoded = cv2.imencode(".png", image)
        string64 = base64.b64encode(img_encoded).decode("utf-8")
        self.conversation.add_frame(string64)

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
