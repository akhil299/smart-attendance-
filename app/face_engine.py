"""
face_engine.py - Face Detection, Recognition & Registration
============================================================
Handles:
  - Face encoding and storage
  - Face matching during attendance
  - Registration pipeline (multi-capture)
"""

import os
import cv2
import dlib
import pickle
import numpy as np
from datetime import datetime
from app.config import (
    FACES_DIR, ENCODINGS_PATH, FACE_RECOGNITION_TOLERANCE,
    REGISTRATION_CAPTURES, FACE_DETECTION_SCALE, FACE_DETECTION_NEIGHBORS
)

# ─── Load dlib models ─────────────────────────────────────────────────────────
_detector = dlib.get_frontal_face_detector()

# dlib shape predictor - 68 landmarks
_PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "shape_predictor_68_face_landmarks.dat")
_RECOGNIZER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "dlib_face_recognition_resnet_model_v1.dat")

_predictor = None
_face_rec_model = None

def _load_models():
    """Lazy-load dlib models (downloaded separately)."""
    global _predictor, _face_rec_model
    if _predictor is None:
        if not os.path.exists(_PREDICTOR_PATH):
            raise FileNotFoundError(
                f"Missing: {_PREDICTOR_PATH}\n"
                "Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            )
        _predictor = dlib.shape_predictor(_PREDICTOR_PATH)
    if _face_rec_model is None:
        if not os.path.exists(_RECOGNIZER_PATH):
            raise FileNotFoundError(
                f"Missing: {_RECOGNIZER_PATH}\n"
                "Download from: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
            )
        _face_rec_model = dlib.face_recognition_model_v1(_RECOGNIZER_PATH)
    return _predictor, _face_rec_model


def get_face_encoding(image_rgb: np.ndarray):
    """
    Returns 128-d face encoding for the largest face in the image.
    Returns None if no face found.
    Falls back to Cascade Classifier if dlib fails.
    """
    predictor, face_rec_model = _load_models()
    
    # Handle both RGB and BGR input
    if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
        return None, None
    
    # Try dlib with upsample
    dets = _detector(image_rgb, 1)
    
    # Fallback to OpenCV Cascade if dlib doesn't find faces
    if len(dets) == 0:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if cascade_path and cv2.os.path.exists(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            # More sensitive cascade parameters
            dets_cascade = cascade.detectMultiScale(gray, 1.05, 4, minSize=(70, 70), maxSize=(300, 300))
            
            if len(dets_cascade) > 0:
                # Convert to dlib rectangles
                dets = [dlib.rectangle(int(x), int(y), int(x+w), int(y+h)) 
                        for x, y, w, h in dets_cascade]
    
    if len(dets) == 0:
        return None, None

    # Pick largest face
    det = max(dets, key=lambda d: d.width() * d.height())
    try:
        shape = predictor(image_rgb, det)
        encoding = np.array(face_rec_model.compute_face_descriptor(image_rgb, shape))
        return encoding, det
    except Exception as e:
        return None, None


def get_all_faces(image_rgb: np.ndarray):
    """
    Detect all faces and return list of (encoding, rect) tuples.
    Fast cascade-first detection for real-time registration.
    """
    # Handle invalid input
    if image_rgb is None or len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
        return []
    
    try:
        predictor, face_rec_model = _load_models()
    except FileNotFoundError:
        return []
    
    results = []
    dets = []
    
    # Try fast cascade first (more reliable for registration)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if cascade_path and cv2.os.path.exists(cascade_path):
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        cascade_dets = cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,      # Smaller = more thorough but slower
            minNeighbors=4,        # Lower = more detections, less strict
            minSize=(70, 70),      # Minimum face size
            maxSize=(300, 300)     # Maximum face size
        )
        
        if len(cascade_dets) > 0:
            dets = [dlib.rectangle(int(x), int(y), int(x+w), int(y+h)) 
                    for x, y, w, h in cascade_dets]
    
    # Try dlib if cascade fails
    if len(dets) == 0:
        dets = _detector(image_rgb, 1)
    
    # Encode all faces
    for det in dets:
        try:
            shape = predictor(image_rgb, det)
            encoding = np.array(face_rec_model.compute_face_descriptor(image_rgb, shape))
            results.append((encoding, det))
        except Exception as e:
            continue
    
    return results


def load_encodings() -> dict:
    """Load saved face encodings from disk."""
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_encodings(encodings: dict):
    """Save face encodings to disk."""
    os.makedirs(os.path.dirname(ENCODINGS_PATH), exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(encodings, f)


def register_student(student_id: str, name: str, frames: list) -> bool:
    """
    Register a new student from a list of captured frames.
    Computes average encoding across multiple captures.

    Args:
        student_id: Unique student ID
        name: Student's full name
        frames: List of BGR frames captured during registration

    Returns:
        True if registration successful, False otherwise
    """
    encodings_list = []

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        enc, _ = get_face_encoding(rgb)
        if enc is not None:
            encodings_list.append(enc)

    if len(encodings_list) < 2:
        return False  # Not enough valid captures

    # Average encoding = more robust representation
    avg_encoding = np.mean(encodings_list, axis=0)

    # Save face image
    os.makedirs(FACES_DIR, exist_ok=True)
    face_img_path = os.path.join(FACES_DIR, f"{student_id}.jpg")
    cv2.imwrite(face_img_path, frames[len(frames)//2])  # Save middle frame

    # Update encodings database
    db = load_encodings()
    db[student_id] = {
        "name": name,
        "encoding": avg_encoding,
        "registered_at": datetime.now().isoformat(),
        "face_image": face_img_path
    }
    save_encodings(db)
    return True


def identify_face(encoding: np.ndarray, db: dict = None) -> tuple:
    """
    Match a face encoding against the registered database.

    Returns:
        (student_id, name, confidence) or (None, "Unknown", 0.0)
    """
    if db is None:
        db = load_encodings()

    if not db or encoding is None:
        return None, "Unknown", 0.0

    best_match_id = None
    best_match_name = "Unknown"
    best_distance = float("inf")

    for sid, info in db.items():
        dist = np.linalg.norm(info["encoding"] - encoding)
        if dist < best_distance:
            best_distance = dist
            best_match_id = sid
            best_match_name = info["name"]

    # Convert distance to confidence (0–1)
    confidence = max(0.0, 1.0 - (best_distance / 1.0))

    if best_distance <= FACE_RECOGNITION_TOLERANCE:
        return best_match_id, best_match_name, confidence
    return None, "Unknown", confidence


def get_face_rect_coords(det) -> tuple:
    """Convert dlib rect to (x, y, w, h)."""
    return det.left(), det.top(), det.width(), det.height()


def get_registered_students() -> list:
    """Return list of all registered students."""
    db = load_encodings()
    return [
        {
            "id": sid,
            "name": info["name"],
            "registered_at": info.get("registered_at", "N/A"),
            "face_image": info.get("face_image", "")
        }
        for sid, info in db.items()
    ]


def delete_student(student_id: str) -> bool:
    """Remove a student from the registry."""
    db = load_encodings()
    if student_id in db:
        # Remove face image
        img_path = db[student_id].get("face_image", "")
        if img_path and os.path.exists(img_path):
            os.remove(img_path)
        del db[student_id]
        save_encodings(db)
        return True
    return False
