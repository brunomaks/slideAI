import mediapipe as mp
import cv2
import os
import tqdm
import json
from pathlib import Path
import numpy as np
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent.parent

BASE_IMAGE_DIR = BASE_DIR / "shared_artifacts/images/hagrid_30k"

DETECTOR_PATH = BASE_DIR / 'shared_artifacts/models/hand_landmarker.task'

RAW_LANDMARKS_PATH = BASE_IMAGE_DIR / "hagrid_30k_landmarks_raw.json"
PROCESSED_LANDMARKS_PATH = BASE_IMAGE_DIR / "hagrid_30k_landmarks_processed.json"

mp_tasks = mp.tasks
BaseOptions = mp_tasks.BaseOptions
VisionRunningMode = mp_tasks.vision.RunningMode
HandLandmarkerOptions = mp_tasks.vision.HandLandmarkerOptions
HandLandmarker = mp_tasks.vision.HandLandmarker

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(DETECTOR_PATH)),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE
)

def extract_landmarks(image_path, landmarker):
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = landmarker.detect(mp_image)
    return results

def save_raw_landmarks():
    dataset = []
    skipped_count = 0
    with HandLandmarker.create_from_options(options) as landmarker:
        for gesture_folder in os.listdir(BASE_IMAGE_DIR):
            gesture_path = BASE_IMAGE_DIR / gesture_folder
            if not gesture_path.is_dir():
                continue
            for file in tqdm.tqdm(os.listdir(gesture_path), desc=f"Extracting {gesture_folder}"):
                image_path = gesture_path / file
                results = extract_landmarks(image_path, landmarker)

                if not results.hand_landmarks:
                    skipped_count += 1
                    continue

                raw_result = {
                    "gesture": gesture_folder,
                    "image_path": str(image_path.relative_to(BASE_IMAGE_DIR)),
                    "handedness": results.handedness[0][0].category_name,
                    "hand_landmarks": [[lm.x, lm.y, lm.z] for lm in results.hand_landmarks[0]]
                }

                dataset.append(raw_result)

    print(f"Skipped {skipped_count} images because no landmarks were detected")
    RAW_LANDMARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_LANDMARKS_PATH, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Raw landmark data saved to {RAW_LANDMARKS_PATH}")

def normalize_landmarks(landmarks, handedness, rotate_angle_deg=0):
    landmarks = np.array(landmarks)[:, :2]  # use only x,y

    wrist = landmarks[0]
    landmarks = landmarks - wrist

    mcp_index = 9
    scale = np.linalg.norm(landmarks[mcp_index])
    if scale > 0:
        landmarks = landmarks / scale

    if handedness == "Left":
        landmarks[:, 0] = -landmarks[:, 0]

    if rotate_angle_deg != 0:
        landmarks = rotate_landmarks(landmarks, rotate_angle_deg)

    return landmarks.tolist()

def rotate_landmarks(landmarks, angle_deg):
    angle = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    center = landmarks.mean(axis=0)
    landmarks = landmarks - center
    landmarks = (landmarks @ R.T) + center
    return landmarks

def process_landmarks(raw_data):
    results = []

    for entry in raw_data:
        landmarks = entry["hand_landmarks"]
        handedness = entry["handedness"]

        normalized = normalize_landmarks(landmarks, handedness)

        results.append({
            "gesture": entry["gesture"],
            "image_path": entry["image_path"],
            "handedness": entry["handedness"],
            "landmarks": normalized,
        })

    return results

def main(args):
    # Step 1: Extract raw landmarks if requested or if raw file doesn't exist
    if args.extract or not RAW_LANDMARKS_PATH.exists():
        save_raw_landmarks()

    # Step 2: Load raw and process if requested
    if args.process:
        with open(RAW_LANDMARKS_PATH) as f:
            raw_data = json.load(f)
        
        processed = process_landmarks(raw_data)
        
        PROCESSED_LANDMARKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROCESSED_LANDMARKS_PATH, "w") as f:
            json.dump(processed, f, indent=2)
        print(f"Processed landmarks saved to {PROCESSED_LANDMARKS_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and/or process hand landmarks")
    parser.add_argument("--extract", action="store_true", help="Run MediaPipe and extract raw landmarks")
    parser.add_argument("--process", action="store_true", help="Load raw landmarks and process them")
    args = parser.parse_args()

    # If no args given, do both
    if not (args.extract or args.process):
        args.extract = True
        args.process = True

    main(args)
