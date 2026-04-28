"""
config.py - Central configuration for Smart Attendance System
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "registered_faces")
MALPRACTICE_DIR = os.path.join(DATA_DIR, "malpractice_snapshots")
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
ENCODINGS_PATH = os.path.join(DATA_DIR, "encodings.pkl")

# ─── Google Sheets ────────────────────────────────────────────────────────────
# Replace this with your actual Google Spreadsheet ID
# Found in the URL: https://docs.google.com/spreadsheets/d/<SPREADSHEET_ID>/edit
SPREADSHEET_ID = "1bOnj1Jy0v_Ma8pd_12rS1mtsc7M181CsUsFG6m3VER8"
SHEET_SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive"
]

# ─── Face Recognition ─────────────────────────────────────────────────────────
FACE_RECOGNITION_TOLERANCE = 0.45   # Lower = stricter match (0.0–1.0)
MIN_FACE_CONFIDENCE = 0.90           # Minimum confidence to accept face
REGISTRATION_CAPTURES = 5            # Number of face captures during registration
FACE_DETECTION_SCALE = 1.1
FACE_DETECTION_NEIGHBORS = 5

# ─── Iris Detection ───────────────────────────────────────────────────────────
IRIS_DETECTION_ENABLED = True
IRIS_MIN_RADIUS = 8
IRIS_MAX_RADIUS = 30

# ─── Anti-Malpractice ─────────────────────────────────────────────────────────
LIVENESS_BLINK_THRESHOLD = 0.18      # Eye aspect ratio threshold for blink (lowered from 0.25 for sensitivity)
LIVENESS_BLINK_FRAMES = 1            # Frames eye must be below threshold (lowered from 2)
LIVENESS_REQUIRED_BLINKS = 1         # Blinks required to pass liveness check
LIVENESS_TIMEOUT_SECONDS = 8         # Seconds allowed to complete liveness check
REFLECTION_THRESHOLD = 240           # Pixel brightness for screen reflection detect
TEXTURE_VARIANCE_THRESHOLD = 180     # Low variance = flat printed/screen image
OBJECT_DETECTION_ENABLED = True      # Detect phones/objects held up

# ─── Low Light / Color Grading ────────────────────────────────────────────────
LOW_LIGHT_THRESHOLD = 60             # Mean brightness below this = low light mode
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)

# ─── UI / Dark Theme ─────────────────────────────────────────────────────────
BG_COLOR = "#0d0d0d"
PANEL_COLOR = "#1a1a2e"
ACCENT_COLOR = "#7f5af0"
ACCENT_HOVER = "#6b46e0"
SUCCESS_COLOR = "#2cb67d"
WARNING_COLOR = "#f4a261"
DANGER_COLOR = "#e63946"
TEXT_PRIMARY = "#fffffe"
TEXT_SECONDARY = "#94a1b2"
BORDER_COLOR = "#2d2d44"
CARD_COLOR = "#16213e"

FONT_TITLE = ("Segoe UI", 22, "bold")
FONT_HEADING = ("Segoe UI", 14, "bold")
FONT_BODY = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO = ("Courier New", 10)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
