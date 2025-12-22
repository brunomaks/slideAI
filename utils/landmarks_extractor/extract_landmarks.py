import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import os
import argparse
import tqdm
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
DETECTOR_PATH = BASE_DIR / 'hand_landmarker.task'
BASE_IMAGE_DIR = BASE_DIR.parent.parent / "shared_artifacts/images/hagrid_30k"

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarker = mp.tasks.vision.HandLandmarker

options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(DETECTOR_PATH)),
                                num_hands=1,
                                running_mode=VisionRunningMode.IMAGE)

class DatasetOptions:
    def __init__(self, augment=False, augment_gestures=None, augment_count=0, augment_angles=None):
        self.augment = augment
        self.augment_gestures = augment_gestures if augment_gestures else []
        self.augment_count = augment_count
        self.augment_angles = augment_angles if augment_angles else []

dataset_options = DatasetOptions(
    augment=True,
    augment_gestures=['two_up_inverted'],
    augment_count=2,
    augment_angles=[-30, -45, -60, -90]
)


def main():
    # args = parse_args()
    # process_image(args.input, args.output)
    input_path = BASE_IMAGE_DIR
    output_path = BASE_IMAGE_DIR / "hagrid_30k_landmarks.json"
    dataset = build_dataset(input_path, dataset_options)
    save_dataset(dataset, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest = "input", default = "input/", help="Path to input folder")
    parser.add_argument("-o", "--output", dest = "output", default = "output/landmarks_raw.json", help="Path to output .json file")
    return parser.parse_args()

def build_dataset(input_path, dataset_options):
    dataset = []
    skipped_count = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        for gesture_folder in os.listdir(input_path):
            gesture_path = input_path / gesture_folder
            for file in tqdm.tqdm(os.listdir(gesture_path), desc=f"Processing {gesture_folder}"):
                image_path = gesture_path / file
                landmarks = extract_landmarks(image_path, landmarker)
                result = process_landmarks(landmarks, gesture_folder, image_path)
                if result:
                    dataset.append(result)
                else:
                    skipped_count += 1

                if dataset_options.augment:
                    if gesture_folder in dataset_options.augment_gestures:
                        for _ in range(dataset_options.augment_count):
                            angle = random.choice(dataset_options.augment_angles)
                            result = process_landmarks(landmarks, gesture_folder, image_path, angle)
                            if result:
                                dataset.append(result)
                            else:
                                skipped_count += 1 
    
    print(f"Total images processed(and augmented?): {len(dataset) + skipped_count}")
    print(f"Total images skipped (no hand detected): {skipped_count}")
    return dataset

def save_dataset(dataset, output_path):
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"File saved successfully.")
    except Exception as e:
        print(f"Error writing file: {e}")

def extract_landmarks(image_path, landmarker):
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    results = landmarker.detect(mp_image)

    return results


def process_landmarks(input, gesture_name, image_path, rotate_angle_deg=0):
    if not input.hand_landmarks:
        return

    landmarks = input.hand_landmarks[0]
    handedness_category = input.handedness[0][0]
    handedness = handedness_category.category_name
    confidence = handedness_category.score

    landmark_list = []
    for lm in landmarks:
        landmark_list.append([lm.x, lm.y])

    normalized_landmarks = normalize_landmarks(landmark_list, handedness, rotate_angle_deg)
    
    # using pathlib for relative paths 
    image_Path = Path(image_path)
    return {
        "gesture": gesture_name,
        "handedness": handedness,
        "confidence": confidence,
        "landmarks": normalized_landmarks,
        "image_path": str(image_Path.relative_to(BASE_IMAGE_DIR))
    }

def normalize_landmarks(landmarks, handedness, rotate_angle_deg=0):
    landmarks = np.array(landmarks)

    # Translate so that wrist is at origin
    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # Scale so that distance between wrist and middle finger MCP is 1
    mcp_index = 9  # Middle finger MCP landmark index
    scale = np.linalg.norm(landmarks[mcp_index]) # euclidean distance from the origin (wrist)
    if scale > 0:
        landmarks = landmarks / scale
    
    # Mirror left hands
    if handedness == "Left":
        landmarks[:, 0]  =  -landmarks[:, 0]

    if rotate_angle_deg != 0:
        landmarks = rotate_landmarks(landmarks, rotate_angle_deg)

    return landmarks.tolist()

# to be able to detect rotated hands
def rotate_landmarks(landmarks, angle_deg):
    angle = np.deg2rad(angle_deg)

    # 2d rotation matrix
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ]) # 2,2

    center = landmarks.mean(axis=0)

    # make center to be the origin
    landmarks = landmarks - center # 21,2

    # perform rotation and translate back
    landmarks = (landmarks @ R.T) + center

    return landmarks


if __name__ == "__main__":
    main()