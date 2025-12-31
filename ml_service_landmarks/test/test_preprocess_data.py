"""
Comprehensive test suite for hand landmark processing pipeline.

Organized by test tiers and functional areas:
- Tier 1: Pure, deterministic, must-have tests
- Tier 2: High-leverage regression prevention
- Tier 3: Integration and confidence tests
"""

import pytest
import numpy as np
import sqlite3
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple

# Import your actual module - adjust the import path as needed
# from your_module import (
#     _normalize_landmarks,
#     _normalize_rotation,
#     _rotate_landmarks,
#     wrist_at_origin,
#     landmarks_within_bounds,
#     fingers_above_wrist,
#     validate_landmarks,
#     _normalize_and_validate_row,
#     _create_database,
#     ingest_raw_landmarks,
#     ingest_normalized_landmarks,
# )


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_landmarks():
    """Generate valid 21-point hand landmarks."""
    # Create a realistic hand shape
    landmarks = np.array([
        [0.0, 0.0],      # 0: Wrist
        [0.1, -0.3],     # 1-4: Thumb
        [0.2, -0.5],
        [0.25, -0.7],
        [0.3, -0.9],
        [-0.1, -0.4],    # 5-8: Index
        [-0.15, -0.8],
        [-0.15, -1.1],
        [-0.15, -1.3],
        [0.0, -0.5],     # 9-12: Middle (reference point)
        [0.0, -1.0],
        [0.0, -1.4],
        [0.0, -1.6],
        [0.1, -0.5],     # 13-16: Ring
        [0.15, -0.9],
        [0.15, -1.2],
        [0.15, -1.4],
        [0.2, -0.4],     # 17-20: Pinky
        [0.25, -0.7],
        [0.25, -0.9],
        [0.25, -1.0],
    ])
    return landmarks


@pytest.fixture
def scaled_landmarks(valid_landmarks):
    """Same landmarks scaled by 2x."""
    return valid_landmarks * 2.0


@pytest.fixture
def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    conn = sqlite3.connect(':memory:')
    yield conn
    conn.close()


@pytest.fixture
def temp_db_path():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
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
    """Create a synthetic test image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ============================================================================
# TIER 1: GEOMETRY & NORMALIZATION TESTS
# ============================================================================

class TestNormalizeLandmarks:
    """Test the _normalize_landmarks function."""
    
    def test_wrist_at_origin(self, valid_landmarks):
        """1. Wrist (landmark 0) should be at origin after normalization."""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        assert np.allclose(normalized[0], [0, 0], atol=1e-6)
    
    def test_scale_invariance(self, valid_landmarks, scaled_landmarks):
        """2. Normalization should be scale-invariant."""
        norm1 = _normalize_landmarks(valid_landmarks, handedness='Right')
        norm2 = _normalize_landmarks(scaled_landmarks, handedness='Right')
        assert np.allclose(norm1, norm2, atol=1e-6)
    
    def test_scale_reference_is_landmark_9(self, valid_landmarks):
        """3. Landmark 9 (middle finger base) should be at unit distance."""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        distance = np.linalg.norm(normalized[9])
        assert np.isclose(distance, 1.0, atol=1e-6)
    
    def test_zero_scale_safety(self):
        """4. Should handle case where landmark 9 coincides with wrist."""
        # All landmarks at wrist position
        landmarks = np.zeros((21, 2))
        normalized = _normalize_landmarks(landmarks, handedness='Right')
        
        # Should not crash and output should be finite
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
        """6. Output should always be (21, 2)."""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        assert normalized.shape == (21, 2)
    
    def test_no_nans_or_infs(self, valid_landmarks):
        """7. Output should never contain NaN or inf values."""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        assert np.isfinite(normalized).all()


class TestNormalizeRotation:
    """Test the _normalize_rotation function."""
    
    def test_middle_finger_points_down(self, valid_landmarks):
        """8. Middle finger (landmark 9) should point down (angle ≈ -π/2)."""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        rotated = _normalize_rotation(normalized)
        
        angle = np.arctan2(rotated[9, 1], rotated[9, 0])
        assert np.isclose(angle, -np.pi/2, atol=0.1)
    
    def test_rotation_preserves_distances(self, valid_landmarks):
        """9. Rotation should preserve distances between landmarks."""
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        
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
        """10. Wrist should stay at [0, 0] after rotation."""
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


# ============================================================================
# TIER 1: VALIDATION LOGIC TESTS
# ============================================================================

class TestWristAtOrigin:
    """Test the wrist_at_origin validator."""
    
    def test_passes_when_wrist_at_origin(self):
        """13. Should pass when wrist is at origin."""
        landmarks = np.zeros((21, 2))
        assert wrist_at_origin(landmarks) is True
    
    def test_fails_when_wrist_offset(self):
        """14. Should fail when wrist is slightly offset."""
        landmarks = np.zeros((21, 2))
        landmarks[0] = [0.002, 0.002]  # Beyond tolerance
        assert wrist_at_origin(landmarks) is False
    
    def test_respects_tolerance(self):
        """15. Should respect tolerance of 1e-3."""
        landmarks = np.zeros((21, 2))
        landmarks[0] = [0.0009, 0.0009]  # Within tolerance
        assert wrist_at_origin(landmarks) is True


class TestLandmarksWithinBounds:
    """Test the landmarks_within_bounds validator."""
    
    def test_passes_for_landmarks_inside_bounds(self, valid_landmarks):
        """16. Should pass for landmarks inside bounds."""
        # Scale to fit within [-3, 3] x [-3, 0]
        landmarks = valid_landmarks * 1.5
        assert landmarks_within_bounds(landmarks) is True
    
    def test_fails_when_x_less_than_minus_3(self):
        """17. Should fail when x < -3."""
        landmarks = np.zeros((21, 2))
        landmarks[5, 0] = -3.1
        assert landmarks_within_bounds(landmarks) is False
    
    def test_fails_when_x_greater_than_3(self):
        """18. Should fail when x > 3."""
        landmarks = np.zeros((21, 2))
        landmarks[5, 0] = 3.1
        assert landmarks_within_bounds(landmarks) is False
    
    def test_fails_when_y_less_than_minus_3(self):
        """19. Should fail when y < -3."""
        landmarks = np.zeros((21, 2))
        landmarks[5, 1] = -3.1
        assert landmarks_within_bounds(landmarks) is False
    
    def test_fails_when_y_greater_than_0(self):
        """20. Should fail when y > 0."""
        landmarks = np.zeros((21, 2))
        landmarks[5, 1] = 0.1
        assert landmarks_within_bounds(landmarks) is False


class TestFingersAboveWrist:
    """Test the fingers_above_wrist validator."""
    
    def test_passes_when_all_fingertips_above_wrist(self):
        """21. Should pass when all fingertips are above wrist."""
        landmarks = np.zeros((21, 2))
        # Set fingertips (4, 8, 12, 16, 20) below wrist (y < 0)
        for tip in [4, 8, 12, 16, 20]:
            landmarks[tip, 1] = -1.0
        assert fingers_above_wrist(landmarks) is True
    
    def test_passes_when_exactly_max_down_fingers_down(self):
        """22. Should pass when exactly max_down fingers are down."""
        landmarks = np.zeros((21, 2))
        # Put 2 fingers down (y > 0)
        landmarks[4, 1] = 0.1
        landmarks[8, 1] = 0.1
        # Others up (y < 0)
        for tip in [12, 16, 20]:
            landmarks[tip, 1] = -1.0
        assert fingers_above_wrist(landmarks, max_down=2) is True
    
    def test_fails_when_too_many_fingers_down(self):
        """23. Should fail when max_down + 1 fingers are down."""
        landmarks = np.zeros((21, 2))
        # Put 3 fingers down
        for tip in [4, 8, 12]:
            landmarks[tip, 1] = 0.1
        # Others up
        for tip in [16, 20]:
            landmarks[tip, 1] = -1.0
        assert fingers_above_wrist(landmarks, max_down=2) is False
    
    def test_correct_fingertip_indices_used(self):
        """24. Should use correct fingertip indices [4, 8, 12, 16, 20]."""
        landmarks = np.zeros((21, 2))
        # Put non-fingertip landmarks down
        landmarks[1, 1] = 0.1
        landmarks[2, 1] = 0.1
        # All fingertips up
        for tip in [4, 8, 12, 16, 20]:
            landmarks[tip, 1] = -1.0
        assert fingers_above_wrist(landmarks) is True


class TestValidateLandmarks:
    """Test the validate_landmarks function."""
    
    def test_passes_when_all_validators_pass(self, valid_landmarks):
        """25. Should pass when all validators pass."""
        # Normalize to ensure validators pass
        normalized = _normalize_landmarks(valid_landmarks, handedness='Right')
        rotated = _normalize_rotation(normalized)
        assert validate_landmarks(rotated) is True
    
    def test_fails_when_any_validator_fails(self):
        """26. Should fail when any validator fails."""
        landmarks = np.zeros((21, 2))
        landmarks[0] = [1.0, 1.0]  # Wrist not at origin
        assert validate_landmarks(landmarks) is False
    
    def test_short_circuit_behavior(self):
        """27. Should short-circuit on first failure."""
        landmarks = np.zeros((21, 2))
        landmarks[0] = [1.0, 1.0]  # Wrist not at origin
        
        # Mock validators to track calls
        with patch('your_module.wrist_at_origin', return_value=False) as mock_wrist:
            with patch('your_module.landmarks_within_bounds') as mock_bounds:
                validate_landmarks(landmarks)
                mock_wrist.assert_called_once()
                # Second validator should not be called due to short-circuit
                mock_bounds.assert_not_called()


# ============================================================================
# TIER 1-2: ROW-LEVEL PROCESSING TESTS
# ============================================================================

class TestNormalizeAndValidateRow:
    """Test the _normalize_and_validate_row function."""
    
    def test_valid_row_returns_normalized_landmarks(self, valid_landmarks):
        """28. Valid row should return normalized landmarks."""
        row = {
            'landmarks_json': json.dumps(valid_landmarks.tolist()),
            'handedness': 'Right'
        }
        result = _normalize_and_validate_row(row)
        assert result is not None
        assert result.shape == (21, 2)
    
    def test_invalid_row_returns_none(self):
        """29. Invalid row should return None."""
        # Create landmarks that will fail validation
        invalid = np.zeros((21, 2))
        invalid[0] = [10.0, 10.0]  # Wrist far from origin
        
        row = {
            'landmarks_json': json.dumps(invalid.tolist()),
            'handedness': 'Right'
        }
        result = _normalize_and_validate_row(row)
        assert result is None
    
    def test_malformed_json_raises_error(self):
        """30. Malformed JSON should raise appropriate error."""
        row = {
            'landmarks_json': 'not valid json{',
            'handedness': 'Right'
        }
        with pytest.raises(json.JSONDecodeError):
            _normalize_and_validate_row(row)
    
    def test_output_is_finite(self, valid_landmarks):
        """31. Output should be finite."""
        row = {
            'landmarks_json': json.dumps(valid_landmarks.tolist()),
            'handedness': 'Right'
        }
        result = _normalize_and_validate_row(row)
        assert np.isfinite(result).all()
    
    def test_output_satisfies_validation_rules(self, valid_landmarks):
        """32. Output should satisfy all validation rules."""
        row = {
            'landmarks_json': json.dumps(valid_landmarks.tolist()),
            'handedness': 'Right'
        }
        result = _normalize_and_validate_row(row)
        assert validate_landmarks(result) is True


# ============================================================================
# TIER 2: DATABASE SCHEMA TESTS
# ============================================================================

class TestCreateDatabase:
    """Test the _create_database function."""
    
    def test_tables_are_created(self, temp_db_path):
        """33. Should create all required tables."""
        _create_database(temp_db_path)
        
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Check raw_landmarks table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='raw_landmarks'
        """)
        assert cursor.fetchone() is not None
        
        # Check processed_landmarks table
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='processed_landmarks'
        """)
        assert cursor.fetchone() is not None
        
        conn.close()
    
    def test_unique_constraint_exists(self, temp_db_path):
        """34. Should enforce UNIQUE constraint on (dataset_version, image_path)."""
        _create_database(temp_db_path)
        
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Try to insert duplicate
        cursor.execute("""
            INSERT INTO raw_landmarks (dataset_version, image_path, handedness, landmarks_json)
            VALUES (?, ?, ?, ?)
        """, ('v1', 'test.jpg', 'Right', '[]'))
        
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO raw_landmarks (dataset_version, image_path, handedness, landmarks_json)
                VALUES (?, ?, ?, ?)
            """, ('v1', 'test.jpg', 'Right', '[]'))
        
        conn.close()
    
    def test_foreign_key_constraint_exists(self, temp_db_path):
        """35. Should enforce foreign key constraint."""
        _create_database(temp_db_path)
        
        conn = sqlite3.connect(temp_db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()
        
        # Try to insert processed landmark without raw landmark
        with pytest.raises(sqlite3.IntegrityError):
            cursor.execute("""
                INSERT INTO processed_landmarks (raw_id, landmarks_json)
                VALUES (?, ?)
            """, (99999, '[]'))
        
        conn.close()
    
    def test_json_validity_constraint_enforced(self, temp_db_path):
        """36. Should enforce JSON validity constraint if applicable."""
        _create_database(temp_db_path)
        
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # This test depends on whether you have CHECK constraints
        # If you do, invalid JSON should fail
        try:
            cursor.execute("""
                INSERT INTO raw_landmarks (dataset_version, image_path, handedness, landmarks_json)
                VALUES (?, ?, ?, ?)
            """, ('v1', 'test.jpg', 'Right', 'not json{'))
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # Expected if CHECK constraint exists
        
        conn.close()


# ============================================================================
# TIER 2: RAW INGESTION TESTS
# ============================================================================

class TestIngestRawLandmarks:
    """Test the ingest_raw_landmarks function."""
    
    def test_inserts_one_valid_image(self, temp_db_path, mock_landmarker, synthetic_image):
        """37. Should insert one valid image."""
        _create_database(temp_db_path)
        
        # Mock successful detection
        mock_hand = Mock()
        mock_hand.landmark = [Mock(x=0.5, y=0.5) for _ in range(21)]
        mock_landmarker.detect.return_value.hand_landmarks = [mock_hand]
        mock_landmarker.detect.return_value.handedness = [Mock(category_name='Right')]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / 'test.jpg'
            # Save synthetic image
            import cv2
            cv2.imwrite(str(img_path), synthetic_image)
            
            inserted, skipped = ingest_raw_landmarks(
                temp_db_path, tmpdir, 'v1', mock_landmarker
            )
        
        assert inserted == 1
        assert skipped == 0
    
    def test_skips_image_with_no_detected_hand(self, temp_db_path, mock_landmarker, synthetic_image):
        """38. Should skip image with no detected hand."""
        _create_database(temp_db_path)
        
        # Mock no detection
        mock_landmarker.detect.return_value.hand_landmarks = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / 'test.jpg'
            import cv2
            cv2.imwrite(str(img_path), synthetic_image)
            
            inserted, skipped = ingest_raw_landmarks(
                temp_db_path, tmpdir, 'v1', mock_landmarker
            )
        
        assert inserted == 0
        assert skipped == 1
    
    def test_skips_unreadable_image(self, temp_db_path, mock_landmarker):
        """39. Should skip unreadable image."""
        _create_database(temp_db_path)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create corrupt image file
            img_path = Path(tmpdir) / 'corrupt.jpg'
            img_path.write_bytes(b'not an image')
            
            inserted, skipped = ingest_raw_landmarks(
                temp_db_path, tmpdir, 'v1', mock_landmarker
            )
        
        assert inserted == 0
        assert skipped >= 1
    
    def test_correctly_stores_relative_image_path(self, temp_db_path, mock_landmarker, synthetic_image):
        """40. Should store relative image path."""
        _create_database(temp_db_path)
        
        mock_hand = Mock()
        mock_hand.landmark = [Mock(x=0.5, y=0.5) for _ in range(21)]
        mock_landmarker.detect.return_value.hand_landmarks = [mock_hand]
        mock_landmarker.detect.return_value.handedness = [Mock(category_name='Right')]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / 'subdir' / 'test.jpg'
            img_path.parent.mkdir(parents=True, exist_ok=True)
            import cv2
            cv2.imwrite(str(img_path), synthetic_image)
            
            ingest_raw_landmarks(temp_db_path, tmpdir, 'v1', mock_landmarker)
            
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT image_path FROM raw_landmarks")
            path = cursor.fetchone()[0]
            conn.close()
            
            assert path == 'subdir/test.jpg'
    
    def test_correctly_stores_handedness(self, temp_db_path, mock_landmarker, synthetic_image):
        """41. Should correctly store handedness."""
        _create_database(temp_db_path)
        
        mock_hand = Mock()
        mock_hand.landmark = [Mock(x=0.5, y=0.5) for _ in range(21)]
        mock_landmarker.detect.return_value.hand_landmarks = [mock_hand]
        mock_landmarker.detect.return_value.handedness = [Mock(category_name='Left')]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / 'test.jpg'
            import cv2
            cv2.imwrite(str(img_path), synthetic_image)
            
            ingest_raw_landmarks(temp_db_path, tmpdir, 'v1', mock_landmarker)
            
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT handedness FROM raw_landmarks")
            handedness = cursor.fetchone()[0]
            conn.close()
            
            assert handedness == 'Left'
    
    def test_correctly_serializes_landmarks_json(self, temp_db_path, mock_landmarker, synthetic_image):
        """42. Should correctly serialize landmarks as JSON."""
        _create_database(temp_db_path)
        
        mock_hand = Mock()
        mock_hand.landmark = [Mock(x=float(i), y=float(i*2)) for i in range(21)]
        mock_landmarker.detect.return_value.hand_landmarks = [mock_hand]
        mock_landmarker.detect.return_value.handedness = [Mock(category_name='Right')]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / 'test.jpg'
            import cv2
            cv2.imwrite(str(img_path), synthetic_image)
            
            ingest_raw_landmarks(temp_db_path, tmpdir, 'v1', mock_landmarker)
            
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT landmarks_json FROM raw_landmarks")
            landmarks_json = cursor.fetchone()[0]
            conn.close()
            
            landmarks = json.loads(landmarks_json)
            assert len(landmarks) == 21
            assert landmarks[0] == [0.0, 0.0]
    
    def test_duplicate_image_is_skipped(self, temp_db_path, mock_landmarker, synthetic_image):
        """43. Should skip duplicate images."""
        _create_database(temp_db_path)
        
        mock_hand = Mock()
        mock_hand.landmark = [Mock(x=0.5, y=0.5) for _ in range(21)]
        mock_landmarker.detect.return_value.hand_landmarks = [mock_hand]
        mock_landmarker.detect.return_value.handedness = [Mock(category_name='Right')]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / 'test.jpg'
            import cv2
            cv2.imwrite(str(img_path), synthetic_image)
            
            # First ingest
            inserted1, skipped1 = ingest_raw_landmarks(
                temp_db_path, tmpdir, 'v1', mock_landmarker
            )
            
            # Second ingest (duplicate)
            inserted2, skipped2 = ingest_raw_landmarks(
                temp_db_path, tmpdir, 'v1', mock_landmarker
            )
        
        assert inserted1 == 1
        assert inserted2 == 0
    
    def test_insert_skip_counters_are_correct(self, temp_db_path, mock_landmarker, synthetic_image):
        """44. Insert/skip counters should be accurate."""
        _create_database(temp_db_path)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 3 valid images
            for i in range(3):
                img_path = Path(tmpdir) / f'test{i}.jpg'
                import cv2
                cv2.imwrite(str(img_path), synthetic_image)
            
            # Mock: first 2 succeed, last fails
            def mock_detect(image):
                if mock_detect.call_count <= 2:
                    mock_hand = Mock()
                    mock_hand.landmark = [Mock(x=0.5, y=0.5) for _ in range(21)]
                    result = Mock()
                    result.hand_landmarks = [mock_hand]
                    result.handedness = [Mock(category_name='Right')]
                    return result
                else:
                    result = Mock()
                    result.hand_landmarks = []
                    result.handedness = []
                    return result
            
            mock_detect.call_count = 0
            
            def side_effect(img):
                mock_detect.call_count += 1
                return mock_detect(img)
            
            mock_landmarker.detect.side_effect = side_effect
            
            inserted, skipped = ingest_raw_landmarks(
                temp_db_path, tmpdir, 'v1', mock_landmarker
            )
            
            assert inserted == 2
            assert skipped == 1
    
    def test_dataset_version_isolation(self, temp_db_path, mock_landmarker, synthetic_image):
        """45. Same image path with different dataset_version should be allowed."""
        _create_database(temp_db_path)
        
        mock_hand = Mock()
        mock_hand.landmark = [Mock(x=0.5, y=0.5) for _ in range(21)]
        mock_landmarker.detect.return_value.hand_landmarks = [mock_hand]
        mock_landmarker.detect.return_value.handedness = [Mock(category_name='Right')]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / 'test.jpg'
            import cv2
            cv2.imwrite(str(img_path), synthetic_image)
            
            # Ingest with v1
            inserted1, _ = ingest_raw_landmarks(
                temp_db_path, tmpdir, 'v1', mock