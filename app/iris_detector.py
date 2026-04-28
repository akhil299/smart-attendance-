"""
iris_detector.py - Iris Detection & Verification
=================================================
Uses Hough Circle Transform on the eye region to detect iris.
Provides additional biometric check beyond face recognition.
"""

import cv2
import numpy as np
import dlib
from scipy.spatial import distance as dist
from app.config import (
    IRIS_MIN_RADIUS, IRIS_MAX_RADIUS,
    LIVENESS_BLINK_THRESHOLD, LIVENESS_BLINK_FRAMES, LIVENESS_REQUIRED_BLINKS
)

# Facial landmark indices for eyes (dlib 68-point model)
LEFT_EYE_IDX  = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))


def eye_aspect_ratio(eye_points: list) -> float:
    """
    Compute Eye Aspect Ratio (EAR) for blink detection.
    EAR drops significantly during a blink.
    
    Eye landmarks order: 0=left_corner, 1=top_left, 2=top_center, 3=top_right, 4=right_corner, 5=bottom_right, etc.
    """
    if not eye_points or len(eye_points) < 6:
        return 1.0  # Return high EAR if landmarks are missing
    
    # Ensure all points are valid
    try:
        # Vertical distances
        A = dist.euclidean(eye_points[1], eye_points[5])  # top_left to bottom_left
        B = dist.euclidean(eye_points[2], eye_points[4])  # top_center to bottom_center
        # Horizontal distance
        C = dist.euclidean(eye_points[0], eye_points[3])  # left_corner to right_corner
        
        # Avoid division by zero
        if C == 0:
            return 1.0
            
        ear = (A + B) / (2.0 * C)
        return max(0.0, min(2.0, ear))  # Clamp between 0 and 2
    except Exception:
        return 1.0


def get_eye_points(shape, eye_indices: list) -> list:
    """Extract (x, y) coordinates for eye landmarks."""
    return [(shape.part(i).x, shape.part(i).y) for i in eye_indices]


def detect_iris(frame_gray: np.ndarray, eye_rect: tuple) -> tuple:
    """
    Detect iris in a given eye region using Hough Circles.

    Args:
        frame_gray: Grayscale frame
        eye_rect: (x, y, w, h) of eye region

    Returns:
        (cx, cy, radius) of detected iris or None
    """
    x, y, w, h = eye_rect
    # Add padding around eye region
    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_gray.shape[1], x + w + pad)
    y2 = min(frame_gray.shape[0], y + h + pad)

    eye_roi = frame_gray[y1:y2, x1:x2]
    if eye_roi.size == 0:
        return None

    # Enhance contrast in eye region
    eye_roi = cv2.equalizeHist(eye_roi)
    eye_roi = cv2.GaussianBlur(eye_roi, (7, 7), 0)

    circles = cv2.HoughCircles(
        eye_roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=25,
        minRadius=IRIS_MIN_RADIUS,
        maxRadius=IRIS_MAX_RADIUS
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Pick circle closest to center of eye region
        center_x = eye_roi.shape[1] // 2
        center_y = eye_roi.shape[0] // 2
        best = min(circles, key=lambda c: abs(c[0] - center_x) + abs(c[1] - center_y))
        # Convert back to full frame coordinates
        return (best[0] + x1, best[1] + y1, best[2])
    return None


def get_eye_rects_from_shape(shape) -> tuple:
    """
    Extract bounding rectangles for both eyes from dlib shape.

    Returns:
        (left_rect, right_rect) where each is (x, y, w, h)
    """
    def points_to_rect(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x = min(xs)
        y = min(ys)
        return (x, y, max(xs) - x, max(ys) - y)

    left_pts = get_eye_points(shape, LEFT_EYE_IDX)
    right_pts = get_eye_points(shape, RIGHT_EYE_IDX)
    return points_to_rect(left_pts), points_to_rect(right_pts)


class LivenessChecker:
    """
    Tracks blink detection over time to verify liveness.
    A real person blinks; a photo/screen does not.
    """

    def __init__(self):
        self.blink_counter = 0
        self.ear_consec_frames = 0
        self.passed = False
        self.total_blinks = 0
        self.frame_count = 0
        self.previous_ear = 1.0

    def reset(self):
        self.blink_counter = 0
        self.ear_consec_frames = 0
        self.passed = False
        self.total_blinks = 0
        self.frame_count = 0
        self.previous_ear = 1.0

    def update(self, left_ear: float, right_ear: float) -> dict:
        """
        Update liveness state with current frame EAR values.

        Returns:
            dict with keys: blinks, passed, ear_avg, status_msg
        """
        self.frame_count += 1
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Detect blink by checking if EAR drops below threshold and recovers
        # This handles both eyes independently
        if avg_ear < LIVENESS_BLINK_THRESHOLD:
            # Eye is closed/closing
            self.ear_consec_frames += 1
        else:
            # Eye is opening - check if we had a blink event
            if self.ear_consec_frames >= LIVENESS_BLINK_FRAMES and self.previous_ear < LIVENESS_BLINK_THRESHOLD:
                # Transition from closed to open = blink detected
                self.total_blinks += 1
            self.ear_consec_frames = 0
        
        self.previous_ear = avg_ear

        if self.total_blinks >= LIVENESS_REQUIRED_BLINKS:
            self.passed = True

        status = "✓ Liveness Confirmed" if self.passed else f"Blink: {self.total_blinks}/{LIVENESS_REQUIRED_BLINKS} (EAR: {avg_ear:.2f})"
        return {
            "blinks": self.total_blinks,
            "passed": self.passed,
            "ear_avg": avg_ear,
            "status_msg": status
        }


class IrisAnalyzer:
    """
    Full iris detection pipeline: detect eyes, compute EAR, find iris.
    """

    def __init__(self, predictor):
        self.predictor = predictor
        self.liveness = LivenessChecker()

    def reset_liveness(self):
        self.liveness.reset()

    def analyze(self, frame_bgr: np.ndarray, face_det) -> dict:
        """
        Run full iris + liveness analysis on a detected face.

        Returns dict with:
            - left_iris, right_iris: (cx, cy, r) or None
            - left_ear, right_ear: float
            - liveness: dict from LivenessChecker
            - landmarks: list of (x,y) for all 68 points
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        try:
            shape = self.predictor(gray, face_det)
        except Exception as e:
            # Fallback if predictor fails
            return {
                "left_iris": None,
                "right_iris": None,
                "left_ear": 1.0,
                "right_ear": 1.0,
                "liveness": self.liveness.update(1.0, 1.0),
                "landmarks": [],
                "left_eye_pts": [],
                "right_eye_pts": []
            }

        try:
            left_pts  = get_eye_points(shape, LEFT_EYE_IDX)
            right_pts = get_eye_points(shape, RIGHT_EYE_IDX)

            # Calculate eye aspect ratio
            left_ear  = eye_aspect_ratio(left_pts)
            right_ear = eye_aspect_ratio(right_pts)

            left_rect, right_rect = get_eye_rects_from_shape(shape)
            left_iris  = detect_iris(gray, left_rect)
            right_iris = detect_iris(gray, right_rect)

            # Update liveness with both eyes' EAR
            liveness_result = self.liveness.update(left_ear, right_ear)

            landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            return {
                "left_iris":  left_iris,
                "right_iris": right_iris,
                "left_ear":   left_ear,
                "right_ear":  right_ear,
                "liveness":   liveness_result,
                "landmarks":  landmarks,
                "left_eye_pts":  left_pts,
                "right_eye_pts": right_pts
            }
        except Exception as e:
            # If analysis fails, don't pass liveness
            return {
                "left_iris": None,
                "right_iris": None,
                "left_ear": 1.0,
                "right_ear": 1.0,
                "liveness": self.liveness.update(1.0, 1.0),
                "landmarks": [],
                "left_eye_pts": [],
                "right_eye_pts": []
            }

    def draw_iris_overlay(self, frame: np.ndarray, analysis: dict) -> np.ndarray:
        """Draw iris circles and eye landmarks on frame."""
        overlay = frame.copy()

        # Draw eye landmarks
        for pt in analysis.get("left_eye_pts", []) + analysis.get("right_eye_pts", []):
            cv2.circle(overlay, pt, 2, (0, 255, 255), -1)

        # Draw iris circles
        for iris, color in [
            (analysis.get("left_iris"), (255, 100, 0)),
            (analysis.get("right_iris"), (255, 100, 0))
        ]:
            if iris:
                cx, cy, r = iris
                cv2.circle(overlay, (cx, cy), r, color, 2)
                cv2.circle(overlay, (cx, cy), 2, (0, 200, 255), -1)

        return overlay
