# VISION-ENABLED DIALOGUE

This repository contains the code for a vision-enabled dialogue system. The system integrates dialogue and visual inputs, leveraging the latest advancements in Large Language Models to create a more contextually aware and immersive dialogue experience. The system summarises the images in the prompt to reduce its length.

For this implementation we use GPT-4. You can use the system as a standalone application with a webcam or integrate it with a Furhat robot.

## Paper Reference

Please refer to the paper for a detailed explanation of the system:

- Title: "I Was Blind but Now I See: Implementing Vision-Enabled Dialogue in Social Robots"
- Authors: Giulio Antonio Abbo and Tony Belpaeme
- Date: November 2023
- [Link to the arXiv preprint](https://doi.org/10.48550/arXiv.2311.08957)

## Setup and Usage

- Clone the repository:
  ```shell
  git clone https://github.com/giubots/vision-enabled-dialogue.git
  cd vision-enabled-dialogue
  ```

### Use it as a standalone application

- Install dependencies:

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

## NOTE

The delay to get an answer from the LLM can vary between 1 and 10 seconds.
