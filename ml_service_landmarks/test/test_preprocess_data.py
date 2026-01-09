# Contributors:
# - Maksym
# - Yaroslav

import pytest
import numpy as np
import sqlite3
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

from src.preprocess_data import (
    _normalize_landmarks,
    _normalize_rotation,
    _rotate_landmarks,
    _wrist_at_origin,
    _landmarks_within_bounds,
    _create_database,
    _extract_landmarks,
    _extract_landmarks_from_image,
    ingest_normalized_landmarks
)

@pytest.fixture
def valid_landmarks():
    """Store valid NORMALIZED 21-point hand landmarks"""
    landmarks = np.array([
      [
        0.0,
        0.0
      ],
      [
        0.43829571228890685,
        -0.273930645226044
      ],
      [
        0.6582884598497839,
        -0.6873584187742646
      ],
      [
        0.7094585986831587,
        -1.0312338847454419
      ],
      [
        0.821258875271017,
        -1.2759600154318045
      ],
      [
        0.2861715014925578,
        -1.0157110773953162
      ],
      [
        0.21039870861873486,
        -1.4347630253849344
      ],
      [
        0.12588486364766063,
        -1.6764234513537457
      ],
      [
        0.038586364064052324,
        -1.8690685034477106
      ],
      [
        -3.066758932213058e-17,
        -1.0
      ],
      [
        -0.09229413411087613,
        -1.4622222661944007
      ],
      [
        -0.17862767354157558,
        -1.7410896344932956
      ],
      [
        -0.262765670715618,
        -1.9643760784656734
      ],
      [
        -0.2696959278551583,
        -0.9193915555859782
      ],
      [
        -0.35019086760523516,
        -1.350082283672182
      ],
      [
        -0.4053733892478007,
        -1.6120586423039613
      ],
      [
        -0.45116841404366226,
        -1.825282883619543
      ],
      [
        -0.5429482884407795,
        -0.7766802011452668
      ],
      [
        -0.61628016488821,
        -1.1027995064667497
      ],
      [
        -0.6485405959618097,
        -1.3178966346630554
      ],
      [
        -0.6699313925583495,
        -1.5001135631900127
      ]
    ])
    return landmarks


@pytest.fixture
def scaled_landmarks(valid_landmarks):
    """Valid landmarks scaled by the factor of two"""
    return valid_landmarks * 2.0


@pytest.fixture
def temp_db_path():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix='.sqlite', delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def mock_landmarker():
    """Mock MediaPipe hand landmarker."""
    landmarker = Mock()
    result = Mock()
    result.hand_landmarks = []
    result.handedness = []
    landmarker.detect.return_value = result
    return landmarker


@pytest.fixture
def synthetic_image():
    """Create an empty test image (all zeros)"""
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestNormalizeLandmarks:
    """Test _normalize_landmarks function."""
    
    def test_wrist_at_origin(self, valid_landmarks):
        """1. Wrist (landmark 0) should be at origin after normalization."""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        assert np.allclose(normalized[0], [0, 0], atol=1e-6)
    
    def test_scale_invariance(self, valid_landmarks, scaled_landmarks):
        """2. Normalization should be scale-invariant."""
        norm1 = _normalize_landmarks(valid_landmarks, handedness='Right')
        norm2 = _normalize_landmarks(scaled_landmarks, handedness='Right')
        assert np.allclose(norm1, norm2, atol=1e-6)
    
    # this test might seem redundant but it protects agains future changes
    def test_scale_reference_is_landmark_9(self, valid_landmarks):
        """3. Landmark 9 (middle finger MCP) should be at unit distance."""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        distance = np.linalg.norm(normalized[9])
        assert np.isclose(distance, 1.0, atol=1e-6)
    
    def test_zero_scale_safety(self):
        """4. Should handle case where landmark 9 coincides with wrist."""
        # All landmarks at wrist position
        landmarks = np.zeros((21, 2))
        normalized = _normalize_landmarks(landmarks, handedness='Right')
        
        # Should not crash and output should not be nan/ or +inf/-inf
        assert normalized.shape == (21, 2)
        assert np.isfinite(normalized).all()
    
    def test_left_hand_mirroring(self, valid_landmarks):
        """5. Left hand should mirror x-coordinates, preserve y."""
        right = _normalize_landmarks(valid_landmarks, handedness='Right')
        left = _normalize_landmarks(valid_landmarks, handedness='Left')
        
        # X-coordinates should be flipped
        assert np.allclose(right[:, 0], -left[:, 0], atol=1e-6)
        # Y-coordinates should be unchanged
        assert np.allclose(right[:, 1], left[:, 1], atol=1e-6)
    
    def test_output_shape(self, valid_landmarks):
        """6. For valid input, output should always be (21, 2)."""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        assert normalized.shape == (21, 2)
    
    def test_no_nans_or_infs(self, valid_landmarks):
        """7. For valid input, output should never contain NaN or inf values."""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        assert np.isfinite(normalized).all()


class TestNormalizeRotation:
    """Test the _normalize_rotation function."""
    
    def test_middle_finger_points_down(self, valid_landmarks):
        """8. Middle finger MCP (landmark 9) should point down"""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        rotated = _normalize_rotation(normalized)
        
        angle = np.arctan2(rotated[9, 1], rotated[9, 0])
        assert np.isclose(angle, -np.pi/2, atol=0.01)
    
    def test_rotation_preserves_distances(self, valid_landmarks):
        """9. Rotation should preserve distances between landmarks"""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        
        # Computing the distance between two vectors is same as finding the length of their difference
        # Calculate distances before rotation
        distances_before = []
        for i in range(21):
            for j in range(i+1, 21):
                dist = np.linalg.norm(normalized[i] - normalized[j])
                distances_before.append(dist)
        
        # Rotate and recalculate
        rotated = _normalize_rotation(normalized)
        distances_after = []
        for i in range(21):
            for j in range(i+1, 21):
                dist = np.linalg.norm(rotated[i] - rotated[j])
                distances_after.append(dist)
        
        assert np.allclose(distances_before, distances_after, atol=1e-6)
    
    def test_rotation_preserves_origin(self, valid_landmarks):
        """10. Wrist should stay at [0, 0] after rotation"""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        rotated = _normalize_rotation(normalized)
        assert np.allclose(rotated[0], [0, 0], atol=1e-6)


class TestRotateLandmarks:
    """Test the _rotate_landmarks helper function."""
    
    def test_identity_rotation(self, valid_landmarks):
        """11. Zero-degree rotation should return input unchanged."""
        rotated = _rotate_landmarks(valid_landmarks, angle=0)
        assert np.allclose(rotated, valid_landmarks, atol=1e-6)
    
    def test_90_degree_rotation(self):
        """12. 90-degree rotation should produce known output."""
        landmarks = np.array([[1.0, 0.0], [0.0, 1.0]])
        rotated = _rotate_landmarks(landmarks, angle=np.pi/2)
        
        expected = np.array([[0.0, 1.0], [-1.0, 0.0]])
        assert np.allclose(rotated, expected, atol=1e-6)

    def test_180_degree_rotation(self):
        """13. 180-degree rotation should produce known output"""
        landmarks = np.array([[1.0, 0.0], [0.0, 1.0]])
        rotated = _rotate_landmarks(landmarks, angle=np.pi)
        
        expected = np.array([[-1.0, 0.0], [0.0, -1.0]])
        assert np.allclose(rotated, expected, atol=1e-6)

    def test_360_degree_rotation(self, valid_landmarks):
        """14. 360-degree rotation should produce exact input."""
        rotated = _rotate_landmarks(valid_landmarks, angle=2 * np.pi)
        assert np.allclose(rotated, valid_landmarks, atol=1e-6)

    def test_negative_90_degree_rotation(self):
        """15. Negative 90-degrees rotation should produce known output."""
        landmarks = np.array([[1.0, 0.0], [0.0, 1.0]])
        rotated = _rotate_landmarks(landmarks, angle=-np.pi/2)
        
        expected = np.array([[0.0, -1.0], [1.0, 0.0]])
        assert np.allclose(rotated, expected, atol=1e-6)

    def test_positive_negative_rotation_inverse(self):
        """16. Rotating by positive and then negative angle should return original landmarks"""
        landmarks = np.array([[1.0, 2.0], [-3.0, 4.0], [0.5, -0.5]])
        
        angle = np.pi / 4  # 45 degrees
        rotated_pos = _rotate_landmarks(landmarks, angle)
        rotated_back = _rotate_landmarks(rotated_pos, -angle)
        
        assert np.allclose(rotated_back, landmarks, atol=1e-6)

    def test_negative_90_equals_270_rotation(self):
        """17. Rotating by positive has the same effect as rotating by the corresponding negative angle"""
        landmarks = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        rotated_neg90 = _rotate_landmarks(landmarks, angle=-np.pi/2)
        rotated_270 = _rotate_landmarks(landmarks, angle=3*np.pi/2)
        
        assert np.allclose(rotated_neg90, rotated_270, atol=1e-6)

class TestWristAtOrigin:
    """Test the wrist_at_origin validator."""
    
    def test_passes_when_wrist_at_origin(self):
        """18. Should pass when wrist is at origin."""
        landmarks = np.zeros((21, 2))
        assert _wrist_at_origin(landmarks) is True
    
    def test_fails_when_wrist_offset(self):
        """19. Should fail when wrist is slightly offset."""
        landmarks = np.zeros((21, 2))
        landmarks[0] = [0.002, 0.002]  # Beyond tolerance
        assert _wrist_at_origin(landmarks) is False
    
    def test_respects_tolerance(self):
        """20. Should respect tolerance of 1e-3."""
        landmarks = np.zeros((21, 2))
        landmarks[0] = [0.0009, 0.0009]  # Within tolerance
        assert _wrist_at_origin(landmarks) is True

class TestLandmarksWithinBounds:
    """Test the landmarks_within_bounds validator."""
    
    def test_passes_for_landmarks_inside_bounds(self, valid_landmarks):
        """21. Should pass for landmarks inside bounds."""
        landmarks = np.clip(valid_landmarks, -2, 0) * 1.5
        assert _landmarks_within_bounds(landmarks) is True
    
    def test_fails_when_x_less_than_minus_3(self):
        """22. Should fail when x < -3"""
        landmarks = np.zeros((21, 2))
        landmarks[5, 0] = -3.1
        assert _landmarks_within_bounds(landmarks) is False
    
    def test_fails_when_x_greater_than_3(self):
        """23. Should fail when x > 3"""
        landmarks = np.zeros((21, 2))
        landmarks[5, 0] = 3.1
        assert _landmarks_within_bounds(landmarks) is False
    
    def test_fails_when_y_less_than_minus_3(self):
        """24. Should fail when y < -3"""
        landmarks = np.zeros((21, 2))
        landmarks[5, 1] = -3.1
        assert _landmarks_within_bounds(landmarks) is False
    
    def test_fails_when_y_greater_than_0(self):
        """25. Should fail when y > 0"""
        landmarks = np.zeros((21, 2))
        landmarks[5, 1] = 0.1
        assert _landmarks_within_bounds(landmarks) is False

    def test_passes_when_on_boundary(self):
        """26. Should pass when at exact boundaries"""
        landmarks = np.zeros((21, 2))
        landmarks[0, :] = [-3, -3]  # bottom-left corner
        landmarks[1, :] = [3, 0]    # top-right corner
        assert _landmarks_within_bounds(landmarks) is True

class TestDatabaseSchema:
    def test_database_schema_created(self, temp_db_path):
        """27. Should detect schema drift"""
        _create_database(temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()

        cur.execute("""
            SELECT name FROM sqlite_master WHERE type='table'
        """)
        tables = {row[0] for row in cur.fetchall()}

        assert "gestures_raw" in tables
        assert "gestures_processed" in tables

        cur.execute("PRAGMA table_info(gestures_raw)")
        raw_cols = {row[1] for row in cur.fetchall()}

        assert {
            "id", "gesture", "image_path",
            "handedness", "landmarks", "dataset_version"
        }.issubset(raw_cols)

        conn.close()

    def test_raw_landmarks_round_trip(self, temp_db_path, valid_landmarks):
        """28. Should pass the process of inserting and retrieving raw 3D landamrks"""
        _create_database(temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()

        landmarks_3d = np.hstack([valid_landmarks, np.zeros((21, 1))])

        cur.execute("""
            INSERT INTO gestures_raw
            (gesture, image_path, handedness, landmarks, dataset_version)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "test",
            "img.png",
            "Right",
            json.dumps(landmarks_3d.tolist()),
            "v1"
        ))

        cur.execute("SELECT landmarks FROM gestures_raw")
        stored = json.loads(cur.fetchone()[0])

        arr = np.array(stored)
        assert arr.shape == (21, 3)
        assert np.isfinite(arr).all()

        conn.close()

    def test_raw_unique_constraint_enforced(self, temp_db_path, valid_landmarks):
        """29. Should pass if unique constraint is enforced"""
        _create_database(temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()

        landmarks_3d = np.hstack([valid_landmarks, np.zeros((21, 1))])

        row = (
            "gesture",
            "img.png",
            "Right",
            json.dumps(landmarks_3d.tolist()),
            "v1"
        )

        cur.execute("""
            INSERT INTO gestures_raw
            (gesture, image_path, handedness, landmarks, dataset_version)
            VALUES (?, ?, ?, ?, ?)
        """, row)

        with pytest.raises(sqlite3.IntegrityError):
            cur.execute("""
                INSERT INTO gestures_raw
                (gesture, image_path, handedness, landmarks, dataset_version)
                VALUES (?, ?, ?, ?, ?)
            """, row)

        conn.close()

    def test_same_image_path_allowed_across_versions(self, temp_db_path, valid_landmarks):
        """30. Should pass because having the same image in two different dataset versions is valid"""
        _create_database(temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()

        landmarks_3d = np.hstack([valid_landmarks, np.zeros((21, 1))])

        for version in ("v1", "v2"):
            cur.execute("""
                INSERT INTO gestures_raw
                (gesture, image_path, handedness, landmarks, dataset_version)
                VALUES (?, ?, ?, ?, ?)
            """, (
                "gesture",
                "img.png",
                "Right",
                json.dumps(landmarks_3d.tolist()),
                version
            ))

        cur.execute("SELECT COUNT(*) FROM gestures_raw")
        assert cur.fetchone()[0] == 2

        conn.close()

    def test_processed_requires_existing_raw(self, temp_db_path, valid_landmarks):
        """31. Should raise exception because its not possible to insert a row into processed table without having corresponding raw table row"""
        _create_database(temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        conn.execute("PRAGMA foreign_keys = ON") # sqlite does not enforce FK by default
        cur = conn.cursor()

        with pytest.raises(sqlite3.IntegrityError):
            cur.execute("""
                INSERT INTO gestures_processed
                (raw_id, gesture, image_path, handedness, landmarks, dataset_version)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                9999999,
                "gesture",
                "img.png",
                "Right",
                json.dumps(valid_landmarks.tolist()),
                "v1"
            ))
            conn.commit() 

        conn.close()

class TestIngestNormalizedLandmarks:
    def test_ingest_normalized_landmarks_happy_path(self, temp_db_path, valid_landmarks):
        """32. Should normalize, and insert landmarsk correctly"""
        _create_database(temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()

        landmarks_3d = np.hstack([valid_landmarks, np.zeros((21, 1))])

        cur.execute("""
            INSERT INTO gestures_raw
            (gesture, image_path, handedness, landmarks, dataset_version)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "gesture",
            "img.png",
            "Right",
            json.dumps(landmarks_3d.tolist()),
            "v1"
        ))

        conn.commit()
        conn.close()

        stats = ingest_normalized_landmarks(temp_db_path, "v1")

        assert stats["inserted"] == 1
        assert stats["discarded"] == 0
        assert stats["label_stats"] == {"gesture": 1}

    def test_invalid_normalized_data_is_discarded(self, temp_db_path):
        """33. Should discard invalid normalized landmarks"""
        _create_database(temp_db_path)

        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()

        # Construct raw landmarks that will violate bounds AFTER normalization
        # Wrist at (0,0), middle finger MCP at (1, 0) results in scale = 1
        # Another point far outside allowed bounds
        bad_landmarks = np.zeros((21, 3))
        bad_landmarks[9, :2] = [1.0, 0.0]       # reference point
        bad_landmarks[5, :2] = [10.0, -10.0]    # violates bounds after normalization


        cur.execute("""
            INSERT INTO gestures_raw
            (gesture, image_path, handedness, landmarks, dataset_version)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "gesture",
            "bad.png",
            "Right",
            json.dumps(bad_landmarks.tolist()),
            "v1"
        ))

        conn.commit()
        conn.close()

        stats = ingest_normalized_landmarks(temp_db_path, "v1")

        assert stats["inserted"] == 0
        assert stats["discarded"] == 1


    def test_processed_landmarks_are_2d_not_3d(self, temp_db_path, valid_landmarks):
        """34. Should insert normalized 2d landmarks insead of original raw 3d ones"""
        _create_database(temp_db_path)

        raw_landmarks_3d = np.hstack([valid_landmarks, np.zeros((21, 1))])

        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO gestures_raw
            (gesture, image_path, handedness, landmarks, dataset_version)
            VALUES (?, ?, ?, ?, ?)
        """, (
            "gesture",
            "img.png",
            "Right",
            json.dumps(raw_landmarks_3d.tolist()),
            "v1"
        ))

        conn.commit()
        conn.close()

        stats = ingest_normalized_landmarks(temp_db_path, "v1")
        assert stats["inserted"] == 1

        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()

        cur.execute("""
            SELECT landmarks FROM gestures_processed
        """)
        stored = json.loads(cur.fetchone()[0])
        arr = np.array(stored)

        assert arr.shape == (21, 2)
        assert arr.ndim == 2
        assert arr.shape[1] == 2

        conn.close()

class TestMockLandmarker:
    def test_extract_landmarks_image_not_found_does_not_call_detect(self, mock_landmarker):
        """35. Should not return any landmarks as the image does not exist"""
        result = _extract_landmarks("missing.png", mock_landmarker)

        assert result.hand_landmarks == []
        assert result.handedness == []

        mock_landmarker.detect.assert_not_called()

    def test_extract_landmarks_from_image_calls_detect(self, mock_landmarker, synthetic_image):
        """36. Should return default landmarks for an image"""
        result = _extract_landmarks_from_image(synthetic_image, mock_landmarker)

        mock_landmarker.detect.assert_called_once()
        assert result is mock_landmarker.detect.return_value