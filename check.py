import torch
from ultralytics import YOLO
import airsim
import cv2
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('OpenCV:', cv2.__version__)
print('AirSim: OK')
print('YOLOv8: OK')
print('All libraries ready!')