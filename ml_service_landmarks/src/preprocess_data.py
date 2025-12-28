import cv2
import os
import json
import sqlite3
from pathlib import Path
import numpy as np
import tqdm

DB_PATH = Path.cwd() / "shared_artifacts" / "data" / "landmarks.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

RAW_IMAGES_PATH = os.getenv("RAW_IMAGES_PATH")

LANDMARK_DETECTOR_PATH = os.getenv("LANDMARK_DETECTOR_PATH")

import mediapipe as mp

mp_tasks = mp.tasks
BaseOptions = mp_tasks.BaseOptions
VisionRunningMode = mp_tasks.vision.RunningMode
HandLandmarkerOptions = mp_tasks.vision.HandLandmarkerOptions
HandLandmarker = mp_tasks.vision.HandLandmarker

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(LANDMARK_DETECTOR_PATH)),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE
)

def create_database(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS gestures_raw (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gesture TEXT NOT NULL,
        image_path TEXT NOT NULL,
        handedness TEXT NOT NULL,
        landmarks TEXT NOT NULL CHECK(json_valid(landmarks))
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS gestures_processed (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gesture TEXT NOT NULL,
        image_path TEXT NOT NULL,
        handedness TEXT NOT NULL,
        landmarks TEXT NOT NULL CHECK(json_valid(landmarks))
    )
    """)

    conn.commit()
    conn.close()


create_database(DB_PATH)

def extract_landmarks(image_path, landmarker):
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = landmarker.detect(mp_image)
    return results


def ingest_raw_landmarks():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    skipped = 0
    inserted = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        for gesture_folder in os.listdir(RAW_IMAGES_PATH):
            gesture_path = RAW_IMAGES_PATH / gesture_folder

            if not gesture_path.is_dir():
                continue

            for file in tqdm.tqdm(os.listdir(gesture_path), desc=f"Extracting {gesture_folder}"):
                image_path = gesture_path / file
                results = extract_landmarks(image_path, landmarker)

                if not results.hand_landmarks:
                    skipped += 1
                    continue

                record = {
                    "gesture": gesture_folder,
                    "image_path": str(image_path.relative_to(RAW_IMAGES_PATH)),
                    "handedness": results.handedness[0][0].category_name,
                    "landmarks": [[lm.x, lm.y, lm.z] for lm in results.hand_landmarks[0]]
                }

                cur.execute("""
                INSERT INTO gestures_raw
                (gesture, image_path, handedness, landmarks)
                VALUES (?, ?, ?, ?)
                """, (
                    record["gesture"],
                    record["image_path"],
                    record["handedness"],
                    json.dumps(record["landmarks"])
                ))

                inserted += 1

    conn.commit()
    conn.close()

    print(f"Inserted {inserted} raw samples")
    print(f"Skipped {skipped} images with no detected landmarks")



# make wrist to be at the origin (0,0), scale all the landmarks to similar scale, 
# flip gestures performed with left hand to only consider right hand gestures
def normalize_landmarks(landmarks, handedness):
    landmarks = np.array(landmarks)[:, :2]

    wrist = landmarks[0]
    landmarks = landmarks - wrist

    scale = np.linalg.norm(landmarks[9])
    if scale > 0:
        landmarks = landmarks / scale

    if handedness == "Left":
        landmarks[:, 0] = -landmarks[:, 0]

    return normalize_rotation(landmarks)

# rotate all the landmarks to point in the same direction (down)
def normalize_rotation(landmarks):
    reference_vector = landmarks[9]  # Middle finger MCP
    current_angle = np.arctan2(reference_vector[1], reference_vector[0])
    target_angle = -np.pi / 2
    rotation_angle = target_angle - current_angle
    return rotate_landmarks(landmarks, rotation_angle)

# rotates around the origin (0,0) - wrist!
def rotate_landmarks(landmarks, angle):
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    return landmarks @ R.T


def ingest_normalized_landmarks():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT * FROM gestures_raw
    """).fetchall()

    inserted = 0
    discarded = 0

    hand_sizes = []

    for _, gesture, image_path, handedness, landmarks_json in rows:
        landmarks = json.loads(landmarks_json)
        normalized = normalize_landmarks(landmarks, handedness)

        # 1. Check that wrists live at the origin
        wrist = normalized[0]
        if not np.allclose(wrist, [0, 0], atol=1e-3):
            discarded += 1
            continue

        # 2. Make sure all the landmarks are contained within specific intervals
        # x-interval: (-3, 3), y-interval: (-3, 0)
        xs, ys = normalized[:, 0], normalized[:, 1]
        if min(xs) < -3 or max(xs) > 3 or min(ys) < -3 or max(ys) > 0:
            discarded += 1
            continue

        # 3. Fingers mostly above wrist (y negative)
        tip_ids = [4, 8, 12, 16, 20]  # thumb and fingertips indices (from mediapipe handlandmarker)
        num_down = sum(ys[i] > 0 for i in tip_ids)
        if num_down > 1:
            discarded += 1
            continue

        # 4.1 Collect scale (distance from wrist to middle finger MCP)
        hand_size = np.linalg.norm(normalized[9])
        hand_sizes.append(hand_size)

        cur.execute("""
        INSERT INTO gestures_processed
        (gesture, image_path, handedness, landmarks)
        VALUES (?, ?, ?, ?)
        """, (
            gesture,
            image_path,
            handedness,
            json.dumps(normalized.tolist())
        ))

        inserted += 1

    # 4.2 Global scale consistency check
    if hand_sizes:
        hand_sizes = np.array(hand_sizes)
        std_scale = hand_sizes.std()
        assert std_scale < 0.5, f"Global scale inconsistency (std={std_scale:.2f})"

    conn.commit()
    conn.close()

    print(f"Inserted {inserted} processed samples")
    print(f"Discarded {discarded} invalid samples")
