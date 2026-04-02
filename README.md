\# YOLOv8 Object Detection Drone in AirSim



A drone that flies autonomously in Microsoft AirSim while detecting objects like cars, people and trees in real time using YOLOv8 on GPU.



\## Demo

Drone detects cars, people and obstacles while flying autonomously!



\## What It Does

\- Drone flies autonomously in AirSim neighborhood

\- YOLOv8 detects objects from drone front camera in real time

\- Obstacle avoidance using depth camera

\- Stuck detection and height control system

\- Runs on NVIDIA GPU for fast detection



\## Tech Stack

\- Python 3.10

\- YOLOv8 (Ultralytics)

\- Microsoft AirSim

\- PyTorch 2.7.1 with CUDA 11.8

\- OpenCV

\- NVIDIA RTX 3050 GPU



\## Project Structure

src/

\- test\_yolo.py        YOLOv8 webcam test

\- airsim\_yolo.py      Drone detection flight



\## How It Works

1\. Drone takes off and flies into neighborhood

2\. Front camera captures RGB frames

3\. YOLOv8 detects objects in each frame on GPU

4\. Depth camera measures obstacle distances

5\. Drone avoids obstacles and continues detecting



\## Objects Detected

\- Cars

\- People

\- Trees

\- Buildings



\## Setup

pip install -r requirements.txt



\## Run

python src/airsim\_yolo.py



\## Author

Yogesh E S

