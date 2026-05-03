# 🎯 YOLOv8 Drone Detection

![Status](https://img.shields.io/badge/Status-Stable-brightgreen)
![Framework](https://img.shields.io/badge/Framework-Ultralytics%20YOLOv8-blue)
![Dataset](https://img.shields.io/badge/Data-Custom%20UAV%20Dataset-yellow)
![inference](https://img.shields.io/badge/Inference-Real--Time-red)


![Drone Detection Demo](detection_demo.gif)

# YOLOv8 Real-Time Drone Detection in AirSim

Real-time UAV detection pipeline using YOLOv8n, pulling live frames from Microsoft AirSim and running inference to detect and track drones with bounding boxes. Built to explore how computer vision can support counter-drone or multi-drone coordination use cases in simulation before moving to hardware.

![Detection Demo](detection_demo.gif)

## What this does

The pipeline connects to AirSim via its Python API, captures frames from the drone's front camera, runs YOLOv8 inference on each frame, and renders bounding boxes with class labels and confidence scores in real time. `check.py` handles the AirSim connection and frame loop; the model and inference logic sit in `src/`.

## Why YOLOv8n

I started with YOLOv8s but it dropped below real-time at 640×640 on my test machine. YOLOv8n at 320×320 runs at ~28 FPS in AirSim with detection confidence above 0.7 for most drone profiles — good enough for the use case.

## Stack

| Component | Detail |
|---|---|
| Model | YOLOv8n (Ultralytics) |
| Simulator | Microsoft AirSim (UE4) |
| Language | Python 3.x |
| Libraries | ultralytics, airsim, opencv-python, numpy |

## Setup

```bash
pip install -r requirements.txt
# Start AirSim in any environment with a drone
python check.py
```

The script auto-connects to AirSim on localhost. Tested on AirSim Neighborhood environment.

## Sample output

Detection frames at 10-frame intervals are saved in the repo root (`frame_0000.png` through `frame_0090.png`) so you can see what the model detects without running AirSim yourself.

## Performance

| Resolution | FPS | Avg confidence |
|---|---|---|
| 640×640 | ~11 | 0.81 |
| 320×320 | ~28 | 0.74 |
| 224×224 | ~35 | 0.68 |

320×320 is the sweet spot for real-time use.

## What I learned

YOLOv8 detects drones well in clear conditions but confidence drops significantly in AirSim's dusk lighting. Fine-tuning on rendered low-light frames brought it back above 0.7. The custom UAV dataset used for fine-tuning covers 4 drone profiles: quadrotor, fixed-wing, hexacopter, nano.

## Status

Stable. Next: test detection-triggered autonomous interception via DroneKit commands.
