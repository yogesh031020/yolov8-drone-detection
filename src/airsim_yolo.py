import airsim
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

print("Loading YOLOv8 on GPU...")
model = YOLO("yolov8n.pt")
model.to("cuda")
print("Model loaded!")

print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()
time.sleep(2)
client.enableApiControl(True)
client.armDisarm(True)
time.sleep(2)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(3)

print("Going to flight height...")
client.moveByVelocityAsync(0, 0, -3, 3).join()
time.sleep(2)

print("Moving into neighborhood...")
client.moveByVelocityAsync(3, 0, 0, 2).join()
time.sleep(1)

print("Starting YOLOv8 detection flight!")
print("Press ESC to stop")
print("-" * 45)

ACTIONS = {
    0: (3,  0,  0, 0.5),
    1: (0, -2,  0, 0.5),
    2: (0,  2,  0, 0.5),
    3: (0,  0, -1, 0.5),
    4: (-2, 0,  0, 0.5),
    5: (0,  0,  1, 0.5),
}
ACTION_NAMES = ["Forward", "Left", "Right", "Up", "Back", "Down"]
TARGET_HEIGHT = 8.0
last_actions = []
step = 0

def safe_min(arr):
    v = arr[arr < 10000]
    return float(v.min()) if len(v) > 0 else 999.0

while True:
    try:
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current_height = -pos.z_val

        rgb_responses = client.simGetImages([
            airsim.ImageRequest(
                "front_center",
                airsim.ImageType.Scene,
                False, False)
        ])

        depth_responses = client.simGetImages([
            airsim.ImageRequest(
                "front_center",
                airsim.ImageType.DepthPlanar,
                True)
        ])

        rgb_r = rgb_responses[0]
        rgb_array = np.frombuffer(
            rgb_r.image_data_uint8, dtype=np.uint8)
        rgb_image = rgb_array.reshape(
            rgb_r.height, rgb_r.width, 3)

        depth_r = depth_responses[0]
        depth = np.array(
            depth_r.image_data_float, dtype=np.float32)
        depth = depth.reshape(depth_r.height, depth_r.width)
        h = depth_r.height
        w = depth_r.width

        results = model(rgb_image, verbose=False)
        annotated = results[0].plot()
        boxes = results[0].boxes
        names = results[0].names
        detected = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = names[cls_id]
                detected.append(label + " " + str(round(conf * 100)) + "%")

        valid_depth = depth[depth < 10000]
        min_dist = float(valid_depth.min()) if len(valid_depth) > 0 else 999.0

        stuck = False
        if len(last_actions) >= 6:
            if set(last_actions[-6:]) == {1, 2}:
                stuck = True

        if stuck:
            action_id = 3
            last_actions = []
            print("STUCK - Going Up!")
        elif current_height > TARGET_HEIGHT + 3:
            action_id = 5
            print("TOO HIGH - Coming Down!")
        elif min_dist < 0.5:
            action_id = 4
            print("CRITICAL - BACKING UP!")
        elif min_dist < 5.0:
            left_min = safe_min(depth[:, :w//3])
            mid_min = safe_min(depth[:, w//3:2*w//3])
            right_min = safe_min(depth[:, 2*w//3:])
            top_min = safe_min(depth[:h//2, :])
            if mid_min > 4.0:
                action_id = 0
            elif top_min > 6.0 and current_height < TARGET_HEIGHT + 3:
                action_id = 3
            elif left_min > right_min:
                action_id = 1
            else:
                action_id = 2
            print("OBSTACLE OVERRIDE -> " + ACTION_NAMES[action_id])
        else:
            action_id = 0

        last_actions.append(action_id)
        if len(last_actions) > 10:
            last_actions.pop(0)

        cv2.putText(annotated,
            "Action: " + ACTION_NAMES[action_id],
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2)
        cv2.putText(annotated,
            "Height: " + str(round(current_height, 1)) + "m",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1)
        cv2.putText(annotated,
            "Min Dist: " + str(round(min_dist, 2)) + "m",
            (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 1)
        cv2.putText(annotated,
            "Objects: " + str(len(detected)),
            (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 1)
        cv2.putText(annotated,
            "Step: " + str(step),
            (10, 135), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (200, 200, 200), 1)

        cv2.imshow("YOLOv8 Drone Detection", annotated)

        vx, vy, vz, dur = ACTIONS[action_id]
        client.moveByVelocityAsync(vx, vy, vz, dur)

        print("Step " + str(step).zfill(3) +
              " | " + ACTION_NAMES[action_id].ljust(8) +
              " | Dist: " + str(round(min_dist, 2)) + "m" +
              " | Height: " + str(round(current_height, 1)) + "m" +
              " | Detected: " + str(detected[:3]))

        step += 1

        if cv2.waitKey(1) == 27:
            print("Stopped!")
            break

        time.sleep(0.2)

    except KeyboardInterrupt:
        print("Interrupted!")
        break

    except Exception as e:
        print("Error: " + str(e))
        continue

cv2.destroyAllWindows()
print("Landing...")
client.landAsync().join()
client.armDisarm(False)
print("Mission Complete!")