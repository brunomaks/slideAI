# Contributors:
# - Mahmoud

from typing import Dict
import cv2
import json
import sqlite3
from pathlib import Path
import numpy as np
import mediapipe as mp
from collections import defaultdict

# exposed functions
__all__ = [
    "init_database",
    "ingest_raw_landmarks",
    "ingest_normalized_landmarks",
]

def init_database(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    _create_database(db_path)

def ingest_raw_landmarks(db_path: Path, landmarker_path: Path, raw_images_path: Path, dataset_version: str) -> Dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        skipped = 0
        inserted = 0
        total = 0

        with _create_landmarker(landmarker_path) as landmarker:
            for gesture_folder in raw_images_path.iterdir():
                gesture_path = raw_images_path / gesture_folder

                if not gesture_path.is_dir():
                    continue

                for file in gesture_path.iterdir():
                    total += 1
                    image_path = gesture_path / file
                    results = _extract_landmarks(image_path, landmarker)

                    if not results.hand_landmarks:
                        skipped += 1
                        continue

                    gesture = gesture_folder.name
                    image_path = str(image_path.relative_to(raw_images_path))
                    handedness = results.handedness[0][0].category_name
                    landmarks = [[lm.x, lm.y, lm.z] for lm in results.hand_landmarks[0]]

                    try:
                        cur.execute("""
                        INSERT INTO gestures_raw
                        (gesture, image_path, handedness, landmarks, dataset_version)
                        VALUES (?, ?, ?, ?, ?)
                        """, (
                            gesture,
                            image_path,
                            handedness,
                            json.dumps(landmarks),
                            dataset_version
                        ))
                        inserted += 1
                    except sqlite3.IntegrityError:
                        # Duplicate (dataset_version, image_path), skip or handle as needed
                        skipped += 1

        return {
            "total": total,
            "inserted": inserted,
            "skipped": skipped
        }

def ingest_normalized_landmarks(db_path: Path, dataset_version: str) -> Dict[str, int]:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        rows = cur.execute("""
            SELECT id, gesture, image_path, handedness, landmarks 
            FROM gestures_raw
            WHERE dataset_version = ?
        """, (dataset_version,)).fetchall()

        inserted = 0
        discarded = 0
        label_stats = defaultdict(int)

        for raw_id, gesture, image_path, handedness, landmarks_json in rows:

            normalized = _normalize_and_validate_row(landmarks_json, handedness)
            if normalized is None:
                discarded += 1
                continue

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
                label_stats[gesture] += 1
            except sqlite3.IntegrityError:
                # Duplicate processed record for this dataset_version/image_path
                discarded += 1

        return {
            "inserted": inserted,
            "discarded": discarded,
            "label_stats": dict(label_stats)
        }


def _normalize_and_validate_row(landmarks_json: str, handedness: str) -> np.ndarray | None:
    landmarks = np.array(json.loads(landmarks_json))
    normalized = _normalize_landmarks(landmarks, handedness)
    normalized_and_rotated = _normalize_rotation(normalized)

    validation = _validate_landmarks(normalized_and_rotated)
    if not validation:
        return None

    return normalized


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
    with sqlite3.connect(db_path) as conn:
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


def _extract_landmarks(image_path, landmarker):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not read image {image_path}, skipping.")
        return mp.tasks.vision.HandLandmarkerResult(hand_landmarks=[], handedness=[], hand_world_landmarks=[])
    return _extract_landmarks_from_image(image, landmarker)

def _extract_landmarks_from_image(image: np.ndarray, landmarker):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    return landmarker.detect(mp_image)

# make wrist to be at the origin (0,0), scale all the landmarks to similar scale, 
# flip gestures performed with left hand to only consider right hand gestures
def _normalize_landmarks(landmarks: np.ndarray, handedness: str, reference_idx: int = 9) -> np.ndarray:
    landmarks = np.array(landmarks)[:, :2]

    wrist = landmarks[0]
    landmarks = landmarks - wrist

    # distance between wrist and middle finger MCP
    scale = np.linalg.norm(landmarks[reference_idx])
    if scale > 0:
        landmarks = landmarks / scale

    if handedness == "Left":
        landmarks[:, 0] = -landmarks[:, 0]

    return landmarks

# rotate all the landmarks to roughly point in the same direction (down)
# vector wrist - middle finger MCP must point down 
def _normalize_rotation(landmarks: np.ndarray, reference_idx: int = 9) -> np.ndarray:
    reference_vector = landmarks[reference_idx]  # Middle finger MCP

    # figure out the angle between positive x axis and the MCP vector 
    current_angle = np.arctan2(reference_vector[1], reference_vector[0])

    target_angle = -np.pi / 2
    rotation_angle = target_angle - current_angle
    return _rotate_landmarks(landmarks, rotation_angle)

# rotates around the origin (0,0) - wrist!
def _rotate_landmarks(landmarks: np.ndarray, angle: float) -> np.ndarray:
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    return landmarks @ R.T

# validators
def _validate_landmarks(landmarks: np.ndarray) -> bool:
    return (
        _wrist_at_origin(landmarks)
        and _landmarks_within_bounds(landmarks)
    )

def _wrist_at_origin(landmarks: np.ndarray) -> bool:
    return np.allclose(landmarks[0], [0, 0], atol=1e-3)

# helps to discard anomalies when a resting hand is detected instead of the actual gesture
def _landmarks_within_bounds(landmarks: np.ndarray, x_bounds=(-3, 3), y_bounds=(-3, 0)) -> bool:
    xs, ys = landmarks[:, 0], landmarks[:, 1]
    if xs.min() < x_bounds[0] or xs.max() > x_bounds[1]:
        return False
    if ys.min() < y_bounds[0] or ys.max() > y_bounds[1]:
        return False
    return True