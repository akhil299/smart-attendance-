"""
camera.py - Camera Feed & Image Processing Utilities
=====================================================
Handles:
  - Webcam capture
  - Low-light enhancement / color grading
  - Frame preprocessing for recognition
"""

import cv2
import numpy as np
from app.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    LOW_LIGHT_THRESHOLD, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID
)


class Camera:
    """Webcam wrapper with auto low-light enhancement."""

    def __init__(self, index: int = 0):
        self.index = index
        self.cap = None
        self.is_open = False

        # CLAHE for low-light enhancement
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_GRID
        )
        self.low_light_mode = False

    def open(self) -> bool:
        """Open webcam."""
        self.cap = cv2.VideoCapture(self.index)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.is_open = True
            return True
        return False

    def read(self) -> tuple:
        """
        Read a frame with auto low-light processing.

        Returns:
            (success, original_frame, processed_frame, is_low_light)
        """
        if not self.is_open or self.cap is None:
            return False, None, None, False

        ret, frame = self.cap.read()
        if not ret:
            return False, None, None, False

        # Check brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        self.low_light_mode = brightness < LOW_LIGHT_THRESHOLD

        if self.low_light_mode:
            processed = self.apply_color_grading(frame)
        else:
            processed = frame.copy()

        return True, frame, processed, self.low_light_mode

    def apply_color_grading(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply color grading for low-light conditions.
        Enhances luminance, reduces noise, warms tone slightly.
        """
        # Convert to LAB for luminance-only enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # CLAHE on luminance channel
        l_enhanced = self._clahe.apply(l)

        # Merge back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Slight denoising
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

        # Warm toning in dark mode (boost red/green slightly)
        enhanced = enhanced.astype(np.float32)
        enhanced[:, :, 2] = np.clip(enhanced[:, :, 2] * 1.05, 0, 255)  # Red
        enhanced[:, :, 1] = np.clip(enhanced[:, :, 1] * 1.02, 0, 255)  # Green
        enhanced = enhanced.astype(np.uint8)

        return enhanced

    def release(self):
        """Release camera resource."""
        if self.cap:
            self.cap.release()
        self.is_open = False


def extract_face_roi(frame: np.ndarray, rect, padding: float = 0.15) -> tuple:
    """
    Extract face ROI from frame with padding.

    Args:
        frame: Full BGR frame
        rect: dlib rectangle
        padding: Fractional padding around face

    Returns:
        (face_bgr, face_gray, (x, y, w, h))
    """
    h_frame, w_frame = frame.shape[:2]
    x = rect.left()
    y = rect.top()
    w = rect.width()
    h = rect.height()

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_frame, x + w + pad_x)
    y2 = min(h_frame, y + h + pad_y)

    face_bgr = frame[y1:y2, x1:x2]
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.size > 0 else None

    return face_bgr, face_gray, (x1, y1, x2 - x1, y2 - y1)


def draw_face_box(frame: np.ndarray, rect, label: str, status: str = "unknown",
                  confidence: float = 0.0) -> np.ndarray:
    """
    Draw styled face bounding box with label.

    status: "present" | "unknown" | "malpractice" | "processing" | "registered"
    """
    color_map = {
        "present":     (45, 178, 100),    # Green
        "unknown":     (200, 80, 80),     # Red
        "malpractice": (0, 60, 200),      # Bright red (BGR)
        "processing":  (200, 160, 0),     # Amber
        "registered":  (160, 100, 230),   # Purple
    }
    color = color_map.get(status, (100, 100, 100))

    x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()

    # Main box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Corner accents
    corner_len = min(w, h) // 5
    thickness = 3
    corners = [
        ((x, y), (x + corner_len, y), (x, y + corner_len)),
        ((x + w, y), (x + w - corner_len, y), (x + w, y + corner_len)),
        ((x, y + h), (x + corner_len, y + h), (x, y + h - corner_len)),
        ((x + w, y + h), (x + w - corner_len, y + h), (x + w, y + h - corner_len)),
    ]
    for corner in corners:
        cv2.line(frame, corner[0], corner[1], color, thickness)
        cv2.line(frame, corner[0], corner[2], color, thickness)

    # Label background
    label_text = f"{label}"
    if confidence > 0:
        label_text += f" ({confidence*100:.0f}%)"

    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    label_y = max(y - 10, text_h + 5)

    cv2.rectangle(frame,
                  (x, label_y - text_h - 6),
                  (x + text_w + 8, label_y + 2),
                  color, -1)
    cv2.putText(frame, label_text,
                (x + 4, label_y - 2),
                cv2.FONT_HERSHEY_DUPLEX, 0.55,
                (255, 255, 255), 1)

    return frame


def draw_hud(frame: np.ndarray, info: dict) -> np.ndarray:
    """
    Draw HUD overlay on the camera feed.

    info keys: mode, fps, low_light, student_count, time
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top bar
    cv2.rectangle(overlay, (0, 0), (w, 36), (13, 13, 25), -1)

    # Mode indicator
    mode_color = {"attendance": (45, 178, 100), "register": (160, 100, 230)}.get(
        info.get("mode", ""), (150, 150, 150)
    )
    mode_text = info.get("mode", "").upper()
    cv2.putText(overlay, f"● {mode_text}", (10, 24),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, mode_color, 1)

    # FPS
    fps_text = f"FPS: {info.get('fps', 0):.0f}"
    cv2.putText(overlay, fps_text, (w - 100, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 161, 178), 1)

    # Low light indicator
    if info.get("low_light"):
        cv2.putText(overlay, "⚡ LOW LIGHT MODE", (w // 2 - 90, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (244, 162, 97), 1)

    # Bottom bar
    cv2.rectangle(overlay, (0, h - 34), (w, h), (13, 13, 25), -1)

    # Student count
    count_text = f"Registered: {info.get('student_count', 0)}"
    cv2.putText(overlay, count_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 161, 178), 1)

    # Time
    time_text = info.get("time", "")
    cv2.putText(overlay, time_text, (w - 130, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 161, 178), 1)

    # Blend overlay
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_for_display(frame: np.ndarray, max_width: int = 640, max_height: int = 480) -> np.ndarray:
    """Resize frame for display while maintaining aspect ratio."""
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    return frame
