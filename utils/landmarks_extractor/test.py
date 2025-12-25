import json
from pathlib import Path
import numpy as np
import argparse
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent

BASE_IMAGE_DIR = BASE_DIR / "shared_artifacts/images/hagrid_30k"

RAW_LANDMARKS_PATH = BASE_IMAGE_DIR / "hagrid_30k_landmarks_raw.json"
PROCESSED_LANDMARKS_PATH = BASE_IMAGE_DIR / "hagrid_30k_landmarks_processed.json"


def normalize_landmarks(landmarks, handedness):
    landmarks = np.array(landmarks)[:, :2]  # use only x,y

    wrist = landmarks[0]
    landmarks = landmarks - wrist

    mcp_index = 9
    scale = np.linalg.norm(landmarks[mcp_index])
    if scale > 0:
        landmarks = landmarks / scale

    if handedness == "Left":
        landmarks[:, 0] = -landmarks[:, 0]

    return landmarks.tolist()

def normalize_rotation(landmarks):
    landmarks = np.array(landmarks)

    # Reference vector: from wrist (now at origin) to middle finger MCP
    reference_vector = landmarks[9]  # Middle finger MCP
    
    # Current angle of reference vector
    current_angle = np.arctan2(reference_vector[1], reference_vector[0])
    
    # Target angle (pointing up in image coordinates = -90 degrees = -pi/2)
    # Note: In image coordinates, Y increases downward, so "up" is negative Y
    target_angle = -np.pi / 2
    
    # Calculate rotation needed
    rotation_angle = target_angle - current_angle
    
    # Apply rotation
    rotated_landmarks = rotate_landmarks(landmarks, rotation_angle)
    
    return rotated_landmarks

def rotate_landmarks(landmarks, angle):
    landmarks = np.array(landmarks)
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    center = landmarks.mean(axis=0)
    landmarks = landmarks - center
    landmarks = (landmarks @ R.T) + center
    return landmarks.tolist()

def process_landmarks(raw_data):
    hand_sizes = []  # collect for global scale check

    for i, entry in enumerate(raw_data):
        landmarks = entry["landmarks"]
        handedness = entry["handedness"]
        gesture = entry["gesture"]
        image_path = entry["image_path"]

        x = np.array([p[0] for p in landmarks], dtype=np.float32)
        y = np.array([p[1] for p in landmarks], dtype=np.float32)

        # if image_path != "stop/9ad4d55d-28c4-4001-8215-6ee22ec044d2.jpg":
        #     continue

        # plt.figure(figsize=(6,6))
        # plt.scatter(x, y)

        # plt.xlim(-10, 10)
        # plt.ylim(-10, 10)
        # plt.xlabel("X (normalized)")
        # plt.ylabel("Y (normalized)")
        # plt.title("Normalized landmarks")

        # plt.gca().invert_yaxis()

        # plt.savefig('landmarks_plot.png')
        # print(image_path)

        # return

        tag = f"[{gesture} | {handedness} | {image_path}]"

        # 1. Wrist at origin
        assert abs(x[0]) < 1e-4, f"{tag} wrist x not zero: {x[0]}"
        assert abs(y[0]) < 1e-4, f"{tag} wrist y not zero: {y[0]}"

        # 2. Range check
        assert x.min() > -3 and x.max() < 3, f"{tag} x out of range: min={x.min()}; max={x.max()}"
        assert y.min() > -3 and y.max() <= 0, f"{tag} y out of range: min={y.min()}; max={y.max()}"

        # 3. Fingers mostly above wrist (Y negative)
        tip_ids = [4, 8, 12, 16, 20]
        num_down = sum(y[i] > 0 for i in tip_ids)
        assert num_down <= 1, f"{tag} too many fingertips below wrist"

        # 4. Collect scale (index fingertip distance)
        hand_size = np.linalg.norm([x[9], y[9]])
        hand_sizes.append(hand_size)

    # 5. Global scale consistency check
    hand_sizes = np.array(hand_sizes)
    assert hand_sizes.std() < 0.5, f"Global scale inconsistency (std={hand_sizes.std():.2f})"

    print(f"✔ Dataset passed validation ({len(raw_data)} samples)")

def main(args):
    # Step 2: Load raw and process if requested
    if args.process:
        with open(PROCESSED_LANDMARKS_PATH) as f:
            raw_data = json.load(f)
        
        process_landmarks(raw_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and/or process hand landmarks")
    parser.add_argument("--process", action="store_true", help="Load raw landmarks and process them")
    parser.add_argument("--augment", action="store_true", help="Apply augmentation during processing")
    args = parser.parse_args()
    args.process = True

    main(args)
