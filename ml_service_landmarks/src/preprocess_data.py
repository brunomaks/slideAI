from typing import Dict
import cv2
import os
import json
import sqlite3
from pathlib import Path
import numpy as np
import mediapipe as mp

DB_PATH = Path(os.getenv("DATABASE_PATH"))

RAW_IMAGES_PATH = Path(os.getenv("RAW_IMAGES_PATH"))
LANDMARK_DETECTOR_PATH = Path(os.getenv("LANDMARK_DETECTOR_PATH"))


__all__ = [
    "init_database",
    "ingest_raw_landmarks",
    "ingest_normalized_landmarks",
]

def init_database(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _create_database(db_path)

def ingest_raw_landmarks(db_path: Path, landmarker_path: Path, raw_images_path: Path, dataset_version: str) -> Dict[str, int]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    skipped = 0
    inserted = 0

    with _create_landmarker(landmarker_path) as landmarker:
        for gesture_folder in os.listdir(raw_images_path):
            gesture_path = raw_images_path / gesture_folder

            if not gesture_path.is_dir():
                continue

            for file in os.listdir(gesture_path):
                image_path = gesture_path / file
                results = _extract_landmarks(image_path, landmarker)

                if not results.hand_landmarks:
                    skipped += 1
                    continue

                record = {
                    "gesture": gesture_folder,
                    "image_path": str(image_path.relative_to(RAW_IMAGES_PATH)),
                    "handedness": results.handedness[0][0].category_name,
                    "landmarks": [[lm.x, lm.y, lm.z] for lm in results.hand_landmarks[0]]
                }

                try:
                    cur.execute("""
                    INSERT INTO gestures_raw
                    (gesture, image_path, handedness, landmarks, dataset_version)
                    VALUES (?, ?, ?, ?, ?)
                    """, (
                        record["gesture"],
                        record["image_path"],
                        record["handedness"],
                        json.dumps(record["landmarks"]),
                        dataset_version
                    ))
                    inserted += 1
                except sqlite3.IntegrityError:
                    # Duplicate (dataset_version, image_path), skip or handle as needed
                    skipped += 1

    conn.commit()
    conn.close()

    return {
        "inserted": inserted,
        "skipped": skipped,
    }

def ingest_normalized_landmarks(db_path: Path, dataset_version: str) -> Dict[str, int]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    rows = cur.execute("""
        SELECT id, gesture, image_path, handedness, landmarks 
        FROM gestures_raw
        WHERE dataset_version = ?
    """, (dataset_version)).fetchall()

    inserted = 0
    discarded = 0

    hand_sizes = []

    for raw_id, gesture, image_path, handedness, landmarks_json in rows:
        landmarks = json.loads(landmarks_json)
        normalized = _normalize_landmarks(landmarks, handedness)

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

        try:
            cur.execute("""
            INSERT INTO gestures_processed
            (raw_id, gesture, image_path, handedness, landmarks, dataset_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                raw_id,
                gesture,
                image_path,
                handedness,
                json.dumps(normalized.tolist()),
                dataset_version
            ))
            inserted += 1
        except sqlite3.IntegrityError:
            # Duplicate processed record for this dataset_version/image_path
            discarded += 1

    # 4.2 Global scale consistency check
    hand_sizes = np.array(hand_sizes)
    std_scale = hand_sizes.std()
        
    conn.commit()
    conn.close()

    return {
        "inserted": inserted,
        "discarded": discarded,
        "std_scale": std_scale # hand size variation
    }


def _create_landmarker(model_path: Path):
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_hands=1,
        running_mode=VisionRunningMode.IMAGE
    )
    return HandLandmarker.create_from_options(options)


def _create_database(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS gestures_raw (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gesture TEXT NOT NULL,
        image_path TEXT NOT NULL,
        handedness TEXT NOT NULL,
        landmarks TEXT NOT NULL CHECK(json_valid(landmarks)),
        dataset_version TEXT NOT NULL,
        UNIQUE (dataset_version, image_path)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS gestures_processed (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        raw_id INTEGER NOT NULL,
        gesture TEXT NOT NULL,
        image_path TEXT NOT NULL,
        handedness TEXT NOT NULL,
        landmarks TEXT NOT NULL CHECK(json_valid(landmarks)),
        dataset_version TEXT NOT NULL,
        FOREIGN KEY (raw_id) REFERENCES gestures_raw(id),
        UNIQUE (dataset_version, image_path)
    )
    """)

    conn.commit()
    conn.close()

def _extract_landmarks(image_path, landmarker):
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = landmarker.detect(mp_image)
    return results

# make wrist to be at the origin (0,0), scale all the landmarks to similar scale, 
# flip gestures performed with left hand to only consider right hand gestures
def _normalize_landmarks(landmarks, handedness):
    landmarks = np.array(landmarks)[:, :2]

    wrist = landmarks[0]
    landmarks = landmarks - wrist

    scale = np.linalg.norm(landmarks[9])
    if scale > 0:
        landmarks = landmarks / scale

    if handedness == "Left":
        landmarks[:, 0] = -landmarks[:, 0]

    return _normalize_rotation(landmarks)

# rotate all the landmarks to point in the same direction (down)
def _normalize_rotation(landmarks):
    reference_vector = landmarks[9]  # Middle finger MCP
    current_angle = np.arctan2(reference_vector[1], reference_vector[0])
    target_angle = -np.pi / 2
    rotation_angle = target_angle - current_angle
    return _rotate_landmarks(landmarks, rotation_angle)

# rotates around the origin (0,0) - wrist!
def _rotate_landmarks(landmarks, angle):
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    return landmarks @ R.T

