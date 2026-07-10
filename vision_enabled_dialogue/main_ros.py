# Copyright 2025 Giulio Antonio Abbo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ROS 2 entry point for the vision-enabled dialogue node."""

import base64
import json
import typing
from collections import deque
from collections.abc import Iterable
from concurrent.futures import Future

import cv2
import rclpy
from cv_bridge import CvBridge
from hri_msgs.msg import LiveSpeech
from hri_valawai_msgs.msg import ConversationUpdate, TranscriptUpdate
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.publisher import Publisher
from rclpy.qos import QoSPresetProfiles
from rclpy.subscription import Subscription
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String
from tts_msgs.action import TTS

from vision_enabled_dialogue.conversation_history import (
    ConversationHistory,
    HidingSummariser,
    SummariseHeavy,
)
from vision_enabled_dialogue.conversation_history.parts import ConversationPart
from vision_enabled_dialogue.dialogue_manager import DialogueManager
from vision_enabled_dialogue.llm import GPT
from vision_enabled_dialogue.response_gen import ResponseGenerator, VLMResponseGenerator


class VisionEnabledDialogue(Node):
    """Run the ROS 2 dialogue node.

    ROS parameters:

    - `persistent_steering` (bool, default: False): Whether steering commands persist until replaced.

    Subscribed topics:

    - `/humans/voices/anonymous_speaker/speech` (hri_msgs/msg/LiveSpeech): Final speech turns.
    - `dialogue/steering` (std_msgs/msg/String): Steering commands for the next response.
    - `/camera/image_raw` (sensor_msgs/msg/Image): Raw camera frames.
    - `/camera/image_raw/compressed` (sensor_msgs/msg/CompressedImage): Compressed camera frames.

    Published topics:

    - `/conversation` (hri_valawai_msgs/msg/ConversationUpdate): Incremental and full conversation updates.
    - `/transcript` (hri_valawai_msgs/msg/TranscriptUpdate): Important utterances for the transcript.

    Action clients:

    - `/tts_engine/tts` (tts_msgs/action/TTS): Text-to-speech requests for generated responses.
    """

    persistent_steering_p: Parameter
    conv_pub: Publisher
    transcript_pub: Publisher
    text_sub: Subscription
    steer_sub: Subscription
    cam_sub: Subscription
    cam_sub_compressed: Subscription
    text_client: ActionClient

    _generator = ResponseGenerator
    _dm: DialogueManager

    def __init__(self):
        super().__init__("vision_enabled_dialogue")
        self.persistent_steering_p = self.declare_parameter(
            "persistent_steering",
            False,
        )
        self.conv_pub = self.create_publisher(
            ConversationUpdate,
            "/conversation",
            qos_profile=QoSPresetProfiles.SERVICES_DEFAULT.value,
        )
        self.transcript_pub = self.create_publisher(
            TranscriptUpdate,
            "/transcript",
            qos_profile=QoSPresetProfiles.SERVICES_DEFAULT.value,
        )
        self.text_sub = self.create_subscription(
            LiveSpeech,
            "/humans/voices/anonymous_speaker/speech",
            self._on_speech,
            qos_profile=QoSPresetProfiles.SERVICES_DEFAULT.value,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.steer_sub = self.create_subscription(
            String,
            "dialogue/steering",
            self._on_steering,
            qos_profile=QoSPresetProfiles.SERVICES_DEFAULT.value,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.cam_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._on_image,
            qos_profile=QoSPresetProfiles.SENSOR_DATA.value,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.cam_sub_compressed = self.create_subscription(
            CompressedImage,
            "/camera/image_raw/compressed",
            self._on_compressed_image,
            qos_profile=QoSPresetProfiles.SENSOR_DATA.value,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.text_client = ActionClient(self, TTS, "/tts_engine/tts")

        history = ConversationHistory(
            strategy_factories=[
                lambda: SummariseHeavy(
                    summariser_factories=[HidingSummariser],
                    max_heavy=5,
                    min_heavy=1,
                ),
            ],
            logger=self.get_logger(),
        )
        history.on_update = self._on_conversation_update
        history.on_full_update = self._on_full_conversation_update
        self._generator = VLMResponseGenerator(GPT(logger=self.get_logger()))
        persistent_steering: bool = self.persistent_steering_p.value
        self._dm = DialogueManager(
            self._speak,
            history,
            self._generator,
            persistent_steering,
        )
        self.get_logger().info("Initialisation completed")

    def _on_conversation_update(self, part: ConversationPart):
        self.conv_pub.publish(
            ConversationUpdate(
                stamp=self.get_clock().now().to_msg(),
                incremental=True,
                parts=json.dumps([part.to_dict()]),
            )
        )
        self.get_logger().debug(
            f"Published conversation update: {part.__class__.__name__}"
        )
        if not part.is_important:
            return
        self.transcript_pub.publish(
            TranscriptUpdate(
                stamp=self.get_clock().now().to_msg(),
                author_id=part.author,
                content=part.content,
            )
        )
        self.get_logger().debug(f"Published transcript update: {part.content}")

    def _on_full_conversation_update(self, parts: list[ConversationPart]):
        self.conv_pub.publish(
            ConversationUpdate(
                stamp=self.get_clock().now().to_msg(),
                incremental=False,
                parts=json.dumps([part.to_dict() for part in parts]),
            )
        )
        self.get_logger().debug("Published full conversation update")

    def _on_speech(self, msg: LiveSpeech):
        stamp = msg.header.stamp
        speech = msg.final
        if not speech:
            return
        self._dm.add_turn(speech, stamp.sec + stamp.nanosec / 1e9)
        self._dm.generate_response()

    def _on_steering(self, msg: String):
        self._dm.add_steering(msg.data)
        self.get_logger().debug(f"Received steering command: {msg.data}")

    def _on_image(self, msg: Image):
        string64 = base64.b64encode(msg.data).decode("utf-8")
        self._dm.add_frame(
            string64, msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        )

    def _on_compressed_image(self, msg: CompressedImage):
        image = CvBridge().compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        _, img_encoded = cv2.imencode(".jpg", image)
        string64 = base64.b64encode(img_encoded).decode("utf-8")
        self._dm.add_frame(
            string64, msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        )

    def _speak(self, speech_g: Iterable[str]) -> Future[str]:
        string_f = Future[str]()
        received_txt = []
        pending_count = 0
        backup_list = deque()  # Used when the spoken text is not returned

        def _tts_goal_response_callback(future: rclpy.Future) -> None:
            nonlocal pending_count
            goal_handle = typing.cast(ClientGoalHandle, future.result())
            if not goal_handle.accepted:
                self.get_logger().warning("TTS rejected")
                return
            self.get_logger().debug("TTS accepted")
            pending_count += 1
            goal_handle.get_result_async().add_done_callback(_tts_result_callback)

        def _tts_feedback_callback(feedback) -> None:
            self.get_logger().debug(f"TTS feedback: {feedback.feedback.word}")

        def _tts_result_callback(future) -> None:
            nonlocal pending_count
            error_msg: str = future.result().result.error_msg
            text: str = future.result().result.text.strip()
            if text != "":
                received_txt.append(text)
            elif error_msg != "":
                self.get_logger().warning(f"TTS error: {error_msg}")
            else:
                self.get_logger().warning("TTS returned neither text nor error")

            pending_count -= 1
            if pending_count == 0:
                string_f.set_result(" ".join(received_txt or backup_list))

        for text in speech_g:
            self.text_client.wait_for_server()
            msg = TTS.Goal(input=text)
            goal_f = self.text_client.send_goal_async(msg, _tts_feedback_callback)
            goal_f.add_done_callback(_tts_goal_response_callback)
            backup_list.append(text)
        return string_f


def main(args=None):
    """Start the ROS 2 dialogue node."""

    rclpy.init(args=args)
    node = VisionEnabledDialogue()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
