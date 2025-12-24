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
    cos_a = np.cos(rotation_angle)
    sin_a = np.sin(rotation_angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    rotated_landmarks = (rotation_matrix @ landmarks.T).T
    
    return rotated_landmarks.tolist()

def rotate_landmarks(landmarks, angle_deg):
    landmarks = np.array(landmarks)
    angle = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    center = landmarks.mean(axis=0)
    landmarks = landmarks - center
    landmarks = (landmarks @ R.T) + center
    return landmarks.tolist()

def process_landmarks(raw_data):
    for entry in raw_data:
        landmarks = entry["hand_landmarks"]
        handedness = entry["handedness"]
        gesture = entry["gesture"]
        image_path = entry["image_path"]


        lr = rotate_landmarks(landmarks, 90)

        ln = normalize_landmarks(lr, handedness)

        rl = normalize_rotation(lr)


        x = [point[0] for point in ln]
        y = [point[1] for point in ln]

        print(x)
        print(y)

        plt.figure(figsize=(6,6))
        plt.scatter(x, y)

        plt.xlim(-2, 2)
        plt.ylim(-3, 0.5)
        plt.xlabel("X (normalized)")
        plt.ylabel("Y (normalized)")
        plt.title("Normalized landmarks")

        plt.gca().invert_yaxis()

        plt.savefig('landmarks_plot.png')
        print(image_path)

        return


def main(args):
    # Step 2: Load raw and process if requested
    if args.process:
        with open(RAW_LANDMARKS_PATH) as f:
            raw_data = json.load(f)
        
        processed = process_landmarks(raw_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and/or process hand landmarks")
    parser.add_argument("--process", action="store_true", help="Load raw landmarks and process them")
    parser.add_argument("--augment", action="store_true", help="Apply augmentation during processing")
    args = parser.parse_args()
    args.process = True

    main(args)
