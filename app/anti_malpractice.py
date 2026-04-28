"""
anti_malpractice.py - Spoof & Malpractice Detection
=====================================================
Detects:
  1. Photo/Screen spoofing (flat texture, reflections, screen glow)
  2. Phone/Object held in front of camera
  3. Captures evidence snapshot on detection
"""

import cv2
import os
import numpy as np
from datetime import datetime
from app.config import (
    REFLECTION_THRESHOLD, TEXTURE_VARIANCE_THRESHOLD,
    MALPRACTICE_DIR, OBJECT_DETECTION_ENABLED
)


# ─── Spoof Detection ──────────────────────────────────────────────────────────

def analyze_texture(face_roi_gray: np.ndarray) -> dict:
    """
    Analyze face texture to detect flat printed/screen images.
    Real faces have high texture variance; photos/screens are flatter.

    Returns dict with: variance, is_flat, score
    """
    if face_roi_gray is None or face_roi_gray.size == 0:
        return {"variance": 0, "is_flat": True, "score": 0.0}

    # Laplacian variance = sharpness / texture richness
    lap = cv2.Laplacian(face_roi_gray, cv2.CV_64F)
    variance = lap.var()

    # LBP-like micro-texture check
    resized = cv2.resize(face_roi_gray, (64, 64))
    sobel_x = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    texture_score = gradient_mag.mean()

    is_flat = variance < TEXTURE_VARIANCE_THRESHOLD

    return {
        "variance": float(variance),
        "is_flat": is_flat,
        "texture_score": float(texture_score),
        "score": min(1.0, variance / (TEXTURE_VARIANCE_THRESHOLD * 2))
    }


def detect_screen_reflection(frame_bgr: np.ndarray, face_rect: tuple) -> dict:
    """
    Detect bright uniform reflection typical of phone/laptop screens.
    Screens emit blue-shifted light and show rectangular bright regions.

    Returns dict with: has_reflection, bright_ratio, screen_glow
    """
    x, y, w, h = face_rect
    # Check region around face (not just face itself)
    pad = int(max(w, h) * 0.3)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_bgr.shape[1], x + w + pad)
    y2 = min(frame_bgr.shape[0], y + h + pad)

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return {"has_reflection": False, "bright_ratio": 0.0, "screen_glow": False}

    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Check for very bright pixels (screen glow)
    bright_pixels = np.sum(gray_roi > REFLECTION_THRESHOLD)
    total_pixels = gray_roi.size
    bright_ratio = bright_pixels / total_pixels

    # Check blue channel dominance (screens emit cold light)
    b_channel = roi[:, :, 0].mean()
    r_channel = roi[:, :, 2].mean()
    screen_glow = (b_channel > r_channel + 15) and bright_ratio > 0.05

    return {
        "has_reflection": bright_ratio > 0.15,
        "bright_ratio": float(bright_ratio),
        "screen_glow": screen_glow
    }


def detect_moiré_pattern(face_roi_gray: np.ndarray) -> bool:
    """
    Detect moiré patterns that appear when photographing a screen.
    Uses FFT to find regular frequency patterns.
    """
    if face_roi_gray is None or face_roi_gray.size == 0:
        return False

    resized = cv2.resize(face_roi_gray, (128, 128)).astype(np.float32)
    dft = cv2.dft(resized, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    # Normalize and check for high-frequency spikes (moiré)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Mask center (DC component)
    h, w = magnitude.shape
    cx, cy = w // 2, h // 2
    cv2.circle(magnitude, (cx, cy), 20, 0, -1)

    # High peaks in frequency domain = periodic patterns = screen/photo
    _, max_val, _, _ = cv2.minMaxLoc(magnitude)
    return float(max_val) > 200.0


# ─── Object Detection (Phone Detection) ──────────────────────────────────────

def detect_rectangular_objects(frame_bgr: np.ndarray) -> list:
    """
    Detect rectangular objects (phones, tablets, printed photos)
    being held in front of the camera.

    Returns list of (x, y, w, h) rectangles found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Dilate to connect edge gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    suspicious_rects = []
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter by size (large enough to be a phone)
        if area < frame_area * 0.04 or area > frame_area * 0.7:
            continue

        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if roughly rectangular (4-6 corners)
        if 4 <= len(approx) <= 6:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h)
            solidity = area / (w * h)

            # Phone-like: portrait or landscape aspect, solid rectangle
            if 0.4 <= aspect <= 2.5 and solidity > 0.6:
                suspicious_rects.append((x, y, w, h))

    return suspicious_rects


def check_depth_consistency(frame_bgr: np.ndarray, face_rect: tuple) -> dict:
    """
    Check if face region has consistent depth cues.
    A flat photo has uniform color gradient; real faces have 3D shadows.
    """
    x, y, w, h = face_rect
    face_roi = frame_bgr[y:y+h, x:x+w]
    if face_roi.size == 0:
        return {"is_3d": True, "shadow_score": 1.0}

    # Convert to LAB for better luminance analysis
    lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(float)

    # Real faces: gradient from cheeks/nose (lighter) to edges (darker)
    # Photos: more uniform or abnormal gradient
    left_half  = l_channel[:, :w//2].mean()
    right_half = l_channel[:, w//2:].mean()
    top_half   = l_channel[:h//2, :].mean()
    bottom_half= l_channel[h//2:, :].mean()

    # Natural face should have some luminance variation
    lr_diff = abs(left_half - right_half)
    tb_diff = abs(top_half - bottom_half)
    total_variation = lr_diff + tb_diff

    # Also check std deviation of luminance
    l_std = l_channel.std()

    is_3d = l_std > 15.0 and total_variation > 5.0

    return {
        "is_3d": is_3d,
        "shadow_score": min(1.0, l_std / 30.0),
        "luminance_std": float(l_std)
    }


# ─── Main Malpractice Checker ─────────────────────────────────────────────────

class MalpracticeChecker:
    """
    Aggregates all anti-spoofing checks and gives a final verdict.
    """

    def __init__(self):
        self.alerts = []

    def check(self, frame_bgr: np.ndarray, face_rect: tuple, face_roi_gray: np.ndarray) -> dict:
        """
        Run all malpractice checks on current frame.

        Returns:
            {
                "is_malpractice": bool,
                "reason": str,
                "confidence": float,
                "details": dict
            }
        """
        reasons = []
        details = {}

        # 1. Texture Analysis
        texture = analyze_texture(face_roi_gray)
        details["texture"] = texture
        if texture["is_flat"]:
            reasons.append("Flat texture detected (possible photo/printout)")

        # 2. Screen Reflection
        reflection = detect_screen_reflection(frame_bgr, face_rect)
        details["reflection"] = reflection
        if reflection["screen_glow"]:
            reasons.append("Screen glow detected (possible digital display)")

        # 3. Moiré Pattern
        has_moire = detect_moiré_pattern(face_roi_gray)
        details["moire"] = has_moire
        if has_moire:
            reasons.append("Moiré pattern detected (screen/photo reproduction)")

        # 4. Object Detection (Phone)
        if OBJECT_DETECTION_ENABLED:
            rects = detect_rectangular_objects(frame_bgr)
            details["suspicious_objects"] = len(rects)
            details["object_rects"] = rects
            if len(rects) > 0:
                reasons.append(f"Rectangular object detected (possible phone/photo: {len(rects)} object(s))")

        # 5. Depth/3D Check
        depth = check_depth_consistency(frame_bgr, face_rect)
        details["depth"] = depth
        if not depth["is_3d"] and texture["is_flat"]:
            reasons.append("No 3D depth cues detected (likely flat image)")

        # Combine scores
        malpractice_signals = sum([
            texture["is_flat"],
            reflection["screen_glow"],
            has_moire,
            len(details.get("object_rects", [])) > 0,
            not depth["is_3d"]
        ])

        is_malpractice = malpractice_signals >= 2  # Require 2+ signals
        confidence = malpractice_signals / 5.0

        return {
            "is_malpractice": is_malpractice,
            "reason": " | ".join(reasons) if reasons else "Clear",
            "confidence": confidence,
            "details": details,
            "signals_count": malpractice_signals
        }


def save_malpractice_snapshot(frame: np.ndarray, student_id: str = "unknown", reason: str = "") -> str:
    """
    Save a snapshot as evidence when malpractice is detected.

    Returns: path to saved snapshot
    """
    os.makedirs(MALPRACTICE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"malpractice_{student_id}_{timestamp}.jpg"
    filepath = os.path.join(MALPRACTICE_DIR, filename)

    # Add text overlay on snapshot
    annotated = frame.copy()
    cv2.putText(annotated, f"MALPRACTICE DETECTED", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(annotated, f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(annotated, f"Reason: {reason[:60]}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    cv2.imwrite(filepath, annotated)
    return filepath
