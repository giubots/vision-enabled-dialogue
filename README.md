# VISION-ENABLED DIALOGUE

This repository contains the code for a vision-enabled dialogue system. The system integrates dialogue and visual inputs, leveraging the latest advancements in Large Language Models to create a more contextually aware and immersive dialogue experience. The system summarises the images in the prompt to reduce its length.

For this implementation, we use GPT-4. You can use the system as a standalone application with a webcam or integrate it with a Furhat robot.
This blog post explains the system in more detail: [Implementing Vision-Powered Chit-Chats with Robots: A GPT-4 Adventure](https://dev.to/giubots/implementing-vision-powered-chit-chats-with-robots-a-gpt-4-adventure-5fhg).

## Paper Reference

Please refer to the paper for a detailed explanation of the system:

- Title: "I Was Blind but Now I See: Implementing Vision-Enabled Dialogue in Social Robots"
- Authors: Giulio Antonio Abbo and Tony Belpaeme
- To be presented as a Late Breaking Report at the 2025 HRI conference.
- [Link to the arXiv preprint](https://doi.org/10.48550/arXiv.2311.08957) (November 2023)

## Setup

- Clone the repository:
 ```shell
  git clone https://github.com/giubots/vision-enabled-dialogue.git
  cd vision-enabled-dialogue
 ```

### Use it as a standalone application

- Install dependencies (we are using Python 3.12):

 ```shell
  pip install -r requirements.txt
 ```

- Run the standalone application (WINDOWS):

 ```powershell
  $env:OPENAI_API_KEY = 'YOUR-KEY-HERE'; python .\main.py
 ```

- Run the standalone application (LINUX):

 ```shell
  OPENAI_API_KEY='YOUR-KEY-HERE' python main.py
 ```

- By default, the application uses the webcam as input. You can also specify a video file as input, which will be fed frame-by-frame:

 ```shell
  python main.py --video=./video.mp4
 ```

- You can also specify a script file (JSON array of strings) to use as input instead of stdin:

 ```shell
   python main.py --script=./script.json
 ```

### Use it with a Furhat robot

_Note: the Furhat SDK does not currently support the video feed, only the actual robot does._

- Install dependencies:

 ```shell
  pip install -r requirements-furhat.txt
 ```

- Enable the video feed: [instructions](https://docs.furhat.io/users/#external-feeds).
- Start the API server: [instructions](https://docs.furhat.io/remote-api/#run-the-server-on-the-robot).
- Set the `FURHAT_IP` variable in `main_furhat.py` to the IP address of your Furhat robot.

- Run the Furhat implementation (WINDOWS):

 ```powershell
  $env:OPENAI_API_KEY = 'YOUR-KEY-HERE'; python .\main_furhat.py
 ```

- Run the Furhat implementation (LINUX):

 ```shell
  OPENAI_API_KEY='YOUR-KEY-HERE' python main_furhat.py
 ```

### Use it with ROS2

You can treat this repo as a ROS2 package. Note that this implementation is still in its very early stages. The easiest way is to run this in the included ROS2 Devcontainer. Alternatively, you can clone this repo in your workspace's src folder. The commands below build, source and run the node.

```shell
# In your workspace folder
colcon build --symlink-install
source ./install/setup.bash
cd src/vision_enabled_dialogue/
pip install -r requirements.txt --break-system-packages
OPENAI_API_KEY='YOUR-KEY-HERE' ros2 run vision_enabled_dialogue main
```

For the frames topic, this node listens to `/camera1/image_raw`. To test it you can use [usb_cam](https://ros-drivers.github.io/usb_cam/main/).
In the commands below, change 'jazzy' with your ROS2 version if supported.

```shell
sudo apt-get install ros-jazzy-usb-cam
source /opt/ros/jazzy/setup.bash
ros2 run usb_cam usb_cam_node_exe --ros-args --remap __ns:=/camera1 -p framerate:=1.0 -p pixel_format:="raw_mjpeg"
```

For the dialogue topic, we rely on the ROS4HRI [rqt_chat](https://github.com/pal-robotics/rqt_chat) tool and its topics. You can install it by cloning the repo and its dependencies in the src folder.

```shell
cd ws/src
git clone https://github.com/pal-robotics/rqt_chat.git
git clone https://github.com/ros4hri/hri_msgs.git
git clone https://github.com/ros4hri/hri_actions_msgs.git
git clone https://github.com/pal-robotics/pal_tts_msgs.git
cd ..
sudo rosdep update && sudo rosdep install --from-paths src --ignore-src -y 
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source ./install/setup.bash
rqt -s rqt_chat --force-discover
```

## Usage

Launch the application, you can start interacting with it (by talking to the Furhat, writing to the console, or sending messages to the right ROS2 topic).
The application will use the available images (through Furhat's camera, webcam, or images published on the right ROS2 topic) to enhance the interaction.
In the Furhat and standalone setups, sending an empty message will force the system to use a VLM on the next interaction.

The models used and the strategy for handling the images can be customised in code.
Note that the delay to get an answer from the LLM can vary between 1 and 10 seconds.
