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


def main():
    # args = parse_args()
    # process_image(args.input, args.output)
    # test_image = '/home/max/projects/slide-ai/shared_artifacts/images/hagrid_30k/0aa54e32-7418-484b-b855-9d6c9d04a46c.jpg'
    # process_image(test_image, "call")
    input_path = BASE_IMAGE_DIR

    dataset = []
    skipped_count = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        for gesture_folder in os.listdir(input_path):
            gesture_path = input_path / gesture_folder
            for file in tqdm.tqdm(os.listdir(gesture_path), desc=f"Processing {gesture_folder}"):
                image_path = gesture_path / file
                result = process_image(image_path, gesture_folder, landmarker)
                if result:
                    dataset.append(result)
                else:
                    skipped_count += 1

    output_path = BASE_IMAGE_DIR / "hagrid_30k_landmarks.json"

    print(f"Saving JSON to: {output_path}")
    print(f"Parent directory exists? {output_path.parent.exists()}")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"File saved successfully.")
    except Exception as e:
        print(f"Error writing file: {e}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest = "input", default = "input/", help="Path to input folder")
    parser.add_argument("-o", "--output", dest = "output", default = "output/landmarks_raw.json", help="Path to output .json file")
    return parser.parse_args()

def process_image(image_path, gesture_name, landmarker):
    image_Path = Path(image_path)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    results = landmarker.detect(mp_image)

    if not results.hand_landmarks:
        return

    landmarks = results.hand_landmarks[0]
    handedness_category = results.handedness[0][0]
    handedness = handedness_category.category_name
    confidence = handedness_category.score

    landmark_list = []
    for lm in landmarks:
        landmark_list.append([lm.x, lm.y])

    normalized_landmarks = normalize_landmarks(landmark_list, handedness)
    
    return {
        "gesture": gesture_name,
        "handedness": handedness,
        "confidence": confidence,
        "landmarks": normalized_landmarks,
        "image_path": str(image_Path.relative_to(BASE_IMAGE_DIR))
    }

def normalize_landmarks(landmarks, handedness):
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

    return landmarks.tolist()

if __name__ == "__main__":
    main()