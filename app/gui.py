"""
gui.py - Main Tkinter GUI (Dark Theme)
=======================================
Full smart attendance interface with:
  - Live camera feed panel
  - Registration tab
  - Attendance tab
  - Logs / History tab
  - Settings tab
"""

import os
import cv2
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
from datetime import datetime
import numpy as np

from app.config import *
from app.camera import Camera, frame_to_rgb, draw_face_box, draw_hud, resize_for_display
from app.face_engine import (
    get_all_faces, get_face_encoding, identify_face, register_student,
    get_registered_students, load_encodings, delete_student, get_face_rect_coords
)
from app.anti_malpractice import MalpracticeChecker, save_malpractice_snapshot
from app.iris_detector import IrisAnalyzer


# ─── Lazy-load dlib predictor ─────────────────────────────────────────────────
_predictor = None
def _get_predictor():
    global _predictor
    if _predictor is None:
        import dlib
        if os.path.exists(os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")):
            _predictor = dlib.shape_predictor(
                os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")
            )
    return _predictor


# ─── Styled Widget Helpers ────────────────────────────────────────────────────

def styled_button(parent, text, command, color=ACCENT_COLOR, width=18, **kwargs):
    btn = tk.Button(
        parent, text=text, command=command,
        bg=color, fg=TEXT_PRIMARY, activebackground=ACCENT_HOVER,
        activeforeground=TEXT_PRIMARY, relief="flat",
        font=FONT_BODY, cursor="hand2", width=width,
        pady=8, **kwargs
    )
    btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT_HOVER))
    btn.bind("<Leave>", lambda e: btn.config(bg=color))
    return btn


def styled_label(parent, text, font=FONT_BODY, color=TEXT_PRIMARY, **kwargs):
    bg = kwargs.pop('bg', PANEL_COLOR)
    return tk.Label(parent, text=text, bg=bg, fg=color, font=font, **kwargs)


def styled_entry(parent, width=24, **kwargs):
    return tk.Entry(
        parent, width=width, bg="#1e1e35", fg=TEXT_PRIMARY,
        insertbackground=TEXT_PRIMARY, relief="flat",
        font=FONT_BODY, highlightthickness=1,
        highlightbackground=BORDER_COLOR, highlightcolor=ACCENT_COLOR,
        **kwargs
    )


def card_frame(parent, **kwargs):
    return tk.Frame(parent, bg=CARD_COLOR, relief="flat",
                    highlightthickness=1, highlightbackground=BORDER_COLOR,
                    **kwargs)


# ─── Main App ─────────────────────────────────────────────────────────────────

class SmartAttendanceApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart Attendance System")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1280x780")
        self.root.minsize(1100, 700)

        # State
        self.camera = Camera(0)
        self.camera_running = False
        self.current_mode = "attendance"   # "attendance" | "register"
        self.db = load_encodings()
        self.malpractice_checker = MalpracticeChecker()
        self.iris_analyzer = None
        self.sheets_enabled = False
        self._try_init_sheets()

        # Registration state
        self.reg_captures = []
        self.reg_capturing = False
        self.reg_student_id = ""
        self.reg_name = ""
        self.reg_last_capture_time = 0  # Track time between captures
        self.reg_capture_interval = 0.8  # Seconds between captures (allows face repositioning)

        # Liveness / attendance tracking
        self.attendance_cooldown = {}   # student_id -> last marked time
        self.last_result = {}
        self.fps_counter = 0
        self.fps_time = time.time()
        self.fps = 0
        self.alert_message = ""
        self.alert_color = SUCCESS_COLOR
        self.alert_timer = 0
        self.iris_analyzer_state = {}  # Persist iris state per student for liveness across frames

        # Build UI
        self._build_ui()
        self._apply_ttk_styles()
        
        # Load initial data
        self.root.after(500, self._refresh_attendance)
        self.root.after(500, self._refresh_students)

        # Start camera
        self.root.after(200, self._start_camera)

    def _try_init_sheets(self):
        """Attempt to connect to Google Sheets."""
        try:
            from app import sheets_manager
            self.sheets = sheets_manager
            self.sheets_enabled = True
        except Exception as e:
            self.sheets = None
            self.sheets_enabled = False
            print(f"[Sheets] Not connected: {e}")

    # ─── UI Builder ───────────────────────────────────────────────────────────

    def _apply_ttk_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=PANEL_COLOR, foreground=TEXT_SECONDARY,
                        font=FONT_BODY, padding=[16, 8],
                        borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", CARD_COLOR), ("active", BORDER_COLOR)],
                  foreground=[("selected", TEXT_PRIMARY)])
        style.configure("Treeview",
                        background=CARD_COLOR, foreground=TEXT_PRIMARY,
                        fieldbackground=CARD_COLOR, rowheight=28,
                        font=FONT_BODY, borderwidth=0)
        style.configure("Treeview.Heading",
                        background=PANEL_COLOR, foreground=ACCENT_COLOR,
                        font=FONT_HEADING)
        style.map("Treeview", background=[("selected", ACCENT_COLOR)])
        style.configure("TScrollbar", background=PANEL_COLOR, troughcolor=BG_COLOR)

    def _build_ui(self):
        # ── Header ─────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=PANEL_COLOR, height=60)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(header, text="◈  Smart Attendance System",
                 bg=PANEL_COLOR, fg=TEXT_PRIMARY, font=FONT_TITLE).pack(side="left", padx=20, pady=10)

        # Connection badge
        self.sheets_badge = tk.Label(
            header,
            text="● Sheets Connected" if self.sheets_enabled else "○ Sheets Offline",
            bg=PANEL_COLOR,
            fg=SUCCESS_COLOR if self.sheets_enabled else WARNING_COLOR,
            font=FONT_SMALL
        )
        self.sheets_badge.pack(side="right", padx=20)

        self.time_label = tk.Label(header, text="", bg=PANEL_COLOR,
                                   fg=TEXT_SECONDARY, font=FONT_BODY)
        self.time_label.pack(side="right", padx=20)
        self._update_time()

        # ── Main layout: left=camera, right=tabs ───────────────────────────
        main = tk.Frame(self.root, bg=BG_COLOR)
        main.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Left panel - camera
        left = tk.Frame(main, bg=BG_COLOR, width=660)
        left.pack(side="left", fill="both", padx=(0, 8))
        left.pack_propagate(False)
        self._build_camera_panel(left)

        # Right panel - tabs
        right = tk.Frame(main, bg=BG_COLOR)
        right.pack(side="right", fill="both", expand=True)
        self._build_tabs(right)

        # ── Status bar ─────────────────────────────────────────────────────
        statusbar = tk.Frame(self.root, bg=PANEL_COLOR, height=28)
        statusbar.pack(fill="x", side="bottom")
        self.status_label = tk.Label(statusbar, text="System ready.",
                                     bg=PANEL_COLOR, fg=TEXT_SECONDARY, font=FONT_SMALL)
        self.status_label.pack(side="left", padx=10)

        self.alert_label = tk.Label(statusbar, text="",
                                    bg=PANEL_COLOR, fg=SUCCESS_COLOR, font=FONT_SMALL)
        self.alert_label.pack(side="right", padx=10)

    def _build_camera_panel(self, parent):
        panel = card_frame(parent)
        panel.pack(fill="both", expand=True)

        # Title row
        title_row = tk.Frame(panel, bg=CARD_COLOR)
        title_row.pack(fill="x", padx=10, pady=(8, 0))

        styled_label(title_row, "Live Camera Feed", FONT_HEADING, ACCENT_COLOR,
                     bg=CARD_COLOR).pack(side="left")

        # Mode buttons
        self.mode_var = tk.StringVar(value="attendance")
        for mode, label in [("attendance", "Attendance Mode"), ("register", "Register Mode")]:
            rb = tk.Radiobutton(
                title_row, text=label, variable=self.mode_var, value=mode,
                command=self._on_mode_change,
                bg=CARD_COLOR, fg=TEXT_SECONDARY, selectcolor=CARD_COLOR,
                activebackground=CARD_COLOR, activeforeground=ACCENT_COLOR,
                font=FONT_SMALL, cursor="hand2"
            )
            rb.pack(side="right", padx=5)

        # Camera canvas
        self.camera_canvas = tk.Label(panel, bg="#050510", cursor="crosshair")
        self.camera_canvas.pack(padx=10, pady=8, fill="both", expand=True)

        # Low light indicator
        self.lowlight_label = tk.Label(panel, text="", bg=CARD_COLOR,
                                       fg=WARNING_COLOR, font=FONT_SMALL)
        self.lowlight_label.pack(pady=(0, 4))

        # Live result card
        result_frame = tk.Frame(panel, bg=PANEL_COLOR, height=80)
        result_frame.pack(fill="x", padx=10, pady=(0, 10))
        result_frame.pack_propagate(False)

        self.result_name_label = tk.Label(result_frame, text="Awaiting face...",
                                          bg=PANEL_COLOR, fg=TEXT_SECONDARY,
                                          font=FONT_HEADING)
        self.result_name_label.pack(pady=(12, 2))

        self.result_status_label = tk.Label(result_frame, text="",
                                            bg=PANEL_COLOR, fg=TEXT_SECONDARY,
                                            font=FONT_SMALL)
        self.result_status_label.pack()

    def _build_tabs(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        self._build_register_tab()
        self._build_attendance_tab()
        self._build_logs_tab()
        self._build_students_tab()

    def _build_register_tab(self):
        tab = tk.Frame(self.notebook, bg=PANEL_COLOR)
        self.notebook.add(tab, text="  ➕ Register  ")

        tk.Label(tab, text="Register New Student",
                 bg=PANEL_COLOR, fg=TEXT_PRIMARY, font=FONT_HEADING).pack(pady=(15, 5))
        tk.Frame(tab, bg=BORDER_COLOR, height=1).pack(fill="x", padx=15)

        form = card_frame(tab)
        form.pack(fill="x", padx=15, pady=12)

        # ID field
        row1 = tk.Frame(form, bg=CARD_COLOR)
        row1.pack(fill="x", padx=12, pady=(12, 5))
        styled_label(row1, "Student ID:", bg=CARD_COLOR).pack(side="left", padx=(0, 8))
        self.reg_id_entry = styled_entry(row1, width=20)
        self.reg_id_entry.pack(side="left")

        # Name field
        row2 = tk.Frame(form, bg=CARD_COLOR)
        row2.pack(fill="x", padx=12, pady=5)
        styled_label(row2, "Full Name:   ", bg=CARD_COLOR).pack(side="left", padx=(0, 8))
        self.reg_name_entry = styled_entry(row2, width=20)
        self.reg_name_entry.pack(side="left")

        # Capture progress
        prog_frame = tk.Frame(form, bg=CARD_COLOR)
        prog_frame.pack(fill="x", padx=12, pady=8)

        self.cap_progress = ttk.Progressbar(prog_frame, length=280, mode="determinate",
                                            maximum=REGISTRATION_CAPTURES)
        self.cap_progress.pack(side="left")
        self.cap_count_label = tk.Label(prog_frame, text=" 0/5",
                                        bg=CARD_COLOR, fg=TEXT_SECONDARY, font=FONT_SMALL)
        self.cap_count_label.pack(side="left", padx=5)

        # Buttons
        btn_frame = tk.Frame(form, bg=CARD_COLOR)
        btn_frame.pack(pady=(5, 12))

        self.reg_start_btn = styled_button(
            btn_frame, "📷 Start Capture", self._start_registration, width=16)
        self.reg_start_btn.pack(side="left", padx=5)

        self.reg_submit_btn = styled_button(
            btn_frame, "✓ Register", self._complete_registration,
            color="#2cb67d", width=12
        )
        self.reg_submit_btn.pack(side="left", padx=5)
        self.reg_submit_btn.config(state="disabled")

        # Registration log
        tk.Label(tab, text="Registration Log",
                 bg=PANEL_COLOR, fg=TEXT_SECONDARY, font=FONT_SMALL).pack(anchor="w", padx=15)
        self.reg_log = tk.Text(tab, height=10, bg="#0d0d1a", fg=TEXT_SECONDARY,
                               font=FONT_MONO, relief="flat", state="disabled",
                               padx=8, pady=5, wrap="word")
        self.reg_log.pack(fill="both", expand=True, padx=15, pady=(0, 10))

    def _build_attendance_tab(self):
        tab = tk.Frame(self.notebook, bg=PANEL_COLOR)
        self.notebook.add(tab, text="  ✓ Attendance  ")

        # Stats row
        stats_row = tk.Frame(tab, bg=PANEL_COLOR)
        stats_row.pack(fill="x", padx=15, pady=12)

        self.stat_cards = {}
        for label, key, color in [
            ("Present", "present", SUCCESS_COLOR),
            ("Malpractice", "malpractice", DANGER_COLOR),
            ("Registered", "total", ACCENT_COLOR),
        ]:
            card = card_frame(stats_row, padx=15, pady=10)
            card.pack(side="left", expand=True, fill="x", padx=5)
            tk.Label(card, text=label, bg=CARD_COLOR, fg=TEXT_SECONDARY,
                     font=FONT_SMALL).pack()
            val_label = tk.Label(card, text="0", bg=CARD_COLOR, fg=color,
                                 font=("Segoe UI", 22, "bold"))
            val_label.pack()
            self.stat_cards[key] = val_label

        # Attendance table
        tk.Label(tab, text="Today's Attendance", bg=PANEL_COLOR,
                 fg=TEXT_PRIMARY, font=FONT_HEADING).pack(anchor="w", padx=15, pady=(5, 3))

        tree_frame = tk.Frame(tab, bg=PANEL_COLOR)
        tree_frame.pack(fill="both", expand=True, padx=15, pady=(0, 5))

        self.attendance_tree = ttk.Treeview(
            tree_frame,
            columns=("id", "name", "time", "status"),
            show="headings", height=12
        )
        cols = [("id", "Student ID", 100), ("name", "Name", 160),
                ("time", "Time", 90), ("status", "Status", 100)]
        for col, heading, width in cols:
            self.attendance_tree.heading(col, text=heading)
            self.attendance_tree.column(col, width=width, anchor="center")

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical",
                                  command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)
        self.attendance_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Tag colors
        self.attendance_tree.tag_configure("present", foreground=SUCCESS_COLOR)
        self.attendance_tree.tag_configure("malpractice", foreground=DANGER_COLOR)
        self.attendance_tree.tag_configure("late", foreground=WARNING_COLOR)

        # Refresh button
        btn_row = tk.Frame(tab, bg=PANEL_COLOR)
        btn_row.pack(pady=5)
        styled_button(btn_row, "🔄 Refresh", self._refresh_attendance, width=14).pack(side="left", padx=5)
        styled_button(btn_row, "📊 Open Sheet", self._open_sheet, color="#16213e", width=14).pack(side="left", padx=5)

    def _build_logs_tab(self):
        tab = tk.Frame(self.notebook, bg=PANEL_COLOR)
        self.notebook.add(tab, text="  📋 Logs  ")

        tk.Label(tab, text="System Logs", bg=PANEL_COLOR,
                 fg=TEXT_PRIMARY, font=FONT_HEADING).pack(anchor="w", padx=15, pady=(15, 5))

        self.log_text = tk.Text(
            tab, bg="#080812", fg=TEXT_SECONDARY, font=FONT_MONO,
            relief="flat", state="disabled", padx=10, pady=8, wrap="word"
        )
        self.log_text.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        # Log tags
        self.log_text.tag_configure("success", foreground=SUCCESS_COLOR)
        self.log_text.tag_configure("warning", foreground=WARNING_COLOR)
        self.log_text.tag_configure("error", foreground=DANGER_COLOR)
        self.log_text.tag_configure("info", foreground=ACCENT_COLOR)

        styled_button(tab, "Clear Logs", self._clear_logs, color=PANEL_COLOR,
                      width=12).pack(pady=(0, 10))

    def _build_students_tab(self):
        tab = tk.Frame(self.notebook, bg=PANEL_COLOR)
        self.notebook.add(tab, text="  👥 Students  ")

        tk.Label(tab, text="Registered Students", bg=PANEL_COLOR,
                 fg=TEXT_PRIMARY, font=FONT_HEADING).pack(anchor="w", padx=15, pady=(15, 5))

        tree_frame = tk.Frame(tab, bg=PANEL_COLOR)
        tree_frame.pack(fill="both", expand=True, padx=15)

        self.students_tree = ttk.Treeview(
            tree_frame,
            columns=("id", "name", "registered"),
            show="headings", height=16
        )
        for col, heading, width in [("id", "Student ID", 120),
                                     ("name", "Name", 180),
                                     ("registered", "Registered At", 160)]:
            self.students_tree.heading(col, text=heading)
            self.students_tree.column(col, width=width, anchor="center")

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical",
                                  command=self.students_tree.yview)
        self.students_tree.configure(yscrollcommand=scrollbar.set)
        self.students_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        btn_row = tk.Frame(tab, bg=PANEL_COLOR)
        btn_row.pack(pady=8)
        styled_button(btn_row, "🔄 Refresh", self._refresh_students, width=12).pack(side="left", padx=5)
        styled_button(btn_row, "🗑 Delete Selected", self._delete_student,
                      color=DANGER_COLOR, width=16).pack(side="left", padx=5)

        self._refresh_students()

    # ─── Camera Loop ──────────────────────────────────────────────────────────

    def _start_camera(self):
        if self.camera.open():
            self.camera_running = True
            self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self._camera_thread.start()
            self._log("Camera opened successfully.", "success")
        else:
            messagebox.showerror("Camera Error", "Could not open webcam.\nPlease check your camera connection.")

    def _camera_loop(self):
        """Main camera processing loop (runs in background thread)."""
        while self.camera_running:
            ret, raw_frame, frame, is_low_light = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            # FPS calculation
            self.fps_counter += 1
            if time.time() - self.fps_time >= 1.0:
                self.fps = self.fps_counter
                self.fps_counter = 0
                self.fps_time = time.time()

            # Process frame
            display_frame = frame.copy()
            mode = self.current_mode

            if mode == "attendance":
                display_frame = self._process_attendance_frame(display_frame, raw_frame)
            elif mode == "register":
                display_frame = self._process_register_frame(display_frame)

            # HUD overlay
            display_frame = draw_hud(display_frame, {
                "mode": mode,
                "fps": self.fps,
                "low_light": is_low_light,
                "student_count": len(self.db),
                "time": datetime.now().strftime("%H:%M:%S")
            })

            # Update UI
            self.root.after(0, self._update_camera_display, display_frame, is_low_light)
            time.sleep(0.03)

    def _process_attendance_frame(self, frame: np.ndarray, raw_frame: np.ndarray) -> np.ndarray:
        """Process frame for attendance marking."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = get_all_faces(rgb)

        if not faces:
            self.root.after(0, self._set_result, "No face detected", "", TEXT_SECONDARY)
            return frame

        for encoding, det in faces:
            x, y, w, h = det.left(), det.top(), det.width(), det.height()

            # Extract face ROI for malpractice check
            face_roi = cv2.cvtColor(
                frame[max(0,y):min(frame.shape[0],y+h),
                      max(0,x):min(frame.shape[1],x+w)],
                cv2.COLOR_BGR2GRAY
            ) if w > 0 and h > 0 else None

            # ── Anti-malpractice check ──────────────────────────────────────
            if face_roi is not None:
                malp = self.malpractice_checker.check(frame, (x, y, w, h), face_roi)
                if malp["is_malpractice"]:
                    # Save snapshot
                    snap_path = save_malpractice_snapshot(raw_frame, "unknown", malp["reason"])

                    # Log malpractice
                    if self.sheets_enabled:
                        try:
                            self.sheets.log_malpractice("UNKNOWN", "Unknown", malp["reason"], snap_path)
                        except Exception:
                            pass

                    draw_face_box(frame, det, "⚠ MALPRACTICE", "malpractice")
                    self.root.after(0, self._set_result,
                                    "🚨 MALPRACTICE DETECTED",
                                    malp["reason"][:60], DANGER_COLOR)
                    self.root.after(0, self._log,
                                    f"Malpractice detected: {malp['reason']}", "error")
                    continue

            # ── Identify face ──────────────────────────────────────────────
            sid, name, confidence = identify_face(encoding, self.db)

            if sid is None:
                draw_face_box(frame, det, "NOT REGISTERED", "unknown", confidence)
                self.root.after(0, self._set_result,
                                "⚠ Not Registered",
                                "Student not found in system", WARNING_COLOR)
                continue

            # ── Iris / liveness check ──────────────────────────────────────
            liveness_ok = False  # REQUIRE liveness by default (changed from True)
            predictor = _get_predictor()
            if predictor and IRIS_DETECTION_ENABLED:
                # Use student_id as key for iris analyzer (same student, same analyzer across frames)
                if sid not in self.iris_analyzer_state:
                    from app.iris_detector import IrisAnalyzer
                    self.iris_analyzer_state[sid] = IrisAnalyzer(predictor)
                
                iris_analyzer = self.iris_analyzer_state[sid]

                try:
                    analysis = iris_analyzer.analyze(frame, det)
                    frame = iris_analyzer.draw_iris_overlay(frame, analysis)
                    liveness_ok = analysis["liveness"]["passed"]
                    liveness_msg = analysis["liveness"]["status_msg"]
                    left_ear = analysis.get("left_ear", 0)
                    right_ear = analysis.get("right_ear", 0)

                    # Show liveness prompt with EAR values for debugging
                    ear_debug = f"L:{left_ear:.2f} R:{right_ear:.2f}"
                    cv2.putText(frame, ear_debug, (x, y - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    
                    cv2.putText(frame, liveness_msg, (x, y - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 215, 0) if not liveness_ok else (45, 178, 100), 1)
                except Exception as e:
                    liveness_msg = "Blink required"
                    cv2.putText(frame, liveness_msg, (x, y - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 215, 0), 1)
                    # Log error but don't block (will fail liveness check)
                    self.root.after(0, self._log, f"Iris detection error: {str(e)[:50]}", "info")

            # ── Mark attendance ────────────────────────────────────────────
            if liveness_ok:
                now = time.time()
                cooldown = self.attendance_cooldown.get(sid, 0)
                if now - cooldown > 10:  # 10-second cooldown per student
                    self.attendance_cooldown[sid] = now
                    
                    # Mark attendance
                    self.root.after(0, self._set_result,
                                    f"✓  {name}", f"ID: {sid} | Confidence: {confidence*100:.1f}%",
                                    SUCCESS_COLOR)
                    self.root.after(0, self._log,
                                    f"✓ Attendance marked: {name} ({sid}) - LIVENESS VERIFIED", "success")
                    draw_face_box(frame, det, name, "present", confidence)
                    
                    # Mark in sheets (async)
                    if self.sheets_enabled:
                        threading.Thread(
                            target=self._mark_attendance_async,
                            args=(sid, name),
                            daemon=True
                        ).start()
                else:
                    draw_face_box(frame, det, f"{name} ✓", "present", confidence)
            else:
                draw_face_box(frame, det, f"{name} - Blink to confirm", "processing", confidence)

        return frame

    def _process_register_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame during registration (show face guide)."""
        # Ensure frame is BGR
        if frame is None or len(frame.shape) < 3:
            return frame
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = get_all_faces(rgb)

        h, w = frame.shape[:2]
        # Draw guide oval (ACCENT_COLOR #7f5af0 -> BGR: 240, 90, 127)
        cv2.ellipse(frame, (w//2, h//2), (120, 155), 0, 0, 360, (240, 90, 127), 2)
        cv2.putText(frame, "Position face in frame", (w//2 - 110, h//2 + 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 90, 127), 1)
        
        # Show detection count for debugging
        faces_detected_text = f"Faces detected: {len(faces)}"
        cv2.putText(frame, faces_detected_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 90, 127), 1)
        
        current_time = time.time()
        
        if faces and len(faces) > 0:
            _, det = faces[0]
            draw_face_box(frame, det, "✓ Face Detected", "registered")
            
            # Auto-capture during registration with timing (anywhere in frame)
            if self.reg_capturing and len(self.reg_captures) < REGISTRATION_CAPTURES:
                time_since_last = current_time - self.reg_last_capture_time
                if time_since_last >= self.reg_capture_interval:
                    self.reg_captures.append(frame.copy())
                    self.reg_last_capture_time = current_time
                    count = len(self.reg_captures)
                    self.root.after(0, self._update_reg_progress, count)
                    self.root.after(0, self._log_reg, f"✓ Captured frame {count}/{REGISTRATION_CAPTURES}")
                    
                    if count >= REGISTRATION_CAPTURES:
                        self.reg_capturing = False
                        self.root.after(0, self._on_capture_complete)
        else:
            cv2.putText(frame, "No face detected - move closer", (w//2 - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return frame

    def _mark_attendance_async(self, student_id: str, name: str):
        """Mark attendance in Google Sheets (background thread)."""
        try:
            if not self.sheets_enabled:
                self.root.after(0, self._log, "Sheets not connected, but attendance marked locally.", "warning")
                return
            
            result = self.sheets.mark_attendance(student_id, name, status="Present", notes="Liveness verified")
            if result["success"]:
                self.root.after(0, self._log, f"✓ Sheets updated: {result.get('message', 'Attendance recorded')}", "success")
                self.root.after(0, self._refresh_attendance)
                self.root.after(0, self._update_stats)
            else:
                self.root.after(0, self._log, f"Sheets write failed: {result.get('message', 'Unknown error')}", "error")
        except Exception as e:
            self.root.after(0, self._log, f"❌ Sheets error: {str(e)[:80]}", "error")

    # ─── Registration ─────────────────────────────────────────────────────────

    def _start_registration(self):
        sid = self.reg_id_entry.get().strip()
        name = self.reg_name_entry.get().strip()

        if not sid or not name:
            messagebox.showwarning("Missing Info", "Please enter both Student ID and Name.")
            return

        self.reg_student_id = sid
        self.reg_name = name
        self.reg_captures = []
        self.reg_capturing = True
        self.reg_last_capture_time = time.time()  # Initialize capture timer
        self.current_mode = "register"
        self.mode_var.set("register")

        self.reg_start_btn.config(state="disabled", text="Capturing...")
        self.reg_submit_btn.config(state="disabled")
        self._log(f"Starting registration for: {name} ({sid}). Position face in the oval.", "info")

    def _update_reg_progress(self, count: int):
        self.cap_progress["value"] = count
        self.cap_count_label.config(text=f" {count}/{REGISTRATION_CAPTURES}")

    def _on_capture_complete(self):
        self.reg_start_btn.config(state="normal", text="📷 Start Capture")
        self.reg_submit_btn.config(state="normal")
        self._log(f"Capture complete! {REGISTRATION_CAPTURES} frames captured.", "success")

    def _complete_registration(self):
        if len(self.reg_captures) < 2:
            messagebox.showwarning("Not Enough Captures",
                                   "Please capture faces first using 'Start Capture'.")
            return

        success = register_student(self.reg_student_id, self.reg_name, self.reg_captures)

        if success:
            self.db = load_encodings()  # Reload DB
            self._log(f"✓ Registered: {self.reg_name} ({self.reg_student_id})", "success")
            messagebox.showinfo("Registered!", f"{self.reg_name} registered successfully.")

            # Reset
            self.reg_id_entry.delete(0, tk.END)
            self.reg_name_entry.delete(0, tk.END)
            self.reg_captures = []
            self.cap_progress["value"] = 0
            self.cap_count_label.config(text=" 0/5")
            self.reg_submit_btn.config(state="disabled")
            self.current_mode = "attendance"
            self.mode_var.set("attendance")
            self._refresh_students()
            self._update_stats()
        else:
            self._log("Registration failed: not enough valid face detections.", "error")
            messagebox.showerror("Registration Failed",
                                 "Could not detect enough faces.\nPlease ensure good lighting and retry.")

    # ─── UI Updates ───────────────────────────────────────────────────────────

    def _update_camera_display(self, frame: np.ndarray, is_low_light: bool):
        """Update camera canvas with latest frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = img.resize((640, 460), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.camera_canvas.config(image=photo)
        self.camera_canvas._photo = photo  # Keep reference

        if is_low_light:
            self.lowlight_label.config(text="⚡ Low light detected — Color grading active")
        else:
            self.lowlight_label.config(text="")

    def _set_result(self, name: str, detail: str, color: str):
        self.result_name_label.config(text=name, fg=color)
        self.result_status_label.config(text=detail, fg=TEXT_SECONDARY)

    def _on_mode_change(self):
        self.current_mode = self.mode_var.get()
        if self.current_mode == "register":
            self.notebook.select(0)
        self._log(f"Switched to {self.current_mode} mode.", "info")

    def _update_time(self):
        self.time_label.config(text=datetime.now().strftime("%a, %d %b %Y  %H:%M:%S"))
        self.root.after(1000, self._update_time)

    def _log(self, message: str, level: str = "info"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[{ts}] ", "info")
        self.log_text.insert("end", f"{message}\n", level)
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.status_label.config(text=message[:80])

    def _log_reg(self, message: str):
        self.reg_log.config(state="normal")
        self.reg_log.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.reg_log.see("end")
        self.reg_log.config(state="disabled")

    def _refresh_attendance(self):
        """Refresh the attendance treeview from Google Sheets."""
        if not self.sheets_enabled:
            return
        threading.Thread(target=self._fetch_attendance, daemon=True).start()

    def _fetch_attendance(self):
        try:
            if not self.sheets_enabled:
                self.root.after(0, self._log, "Sheets not enabled - skipping fetch", "warning")
                return
            
            records = self.sheets.get_today_attendance()
            self.root.after(0, self._log, f"Fetched {len(records)} attendance records from sheets", "info")
            self.root.after(0, self._populate_attendance_tree, records)
        except Exception as e:
            self.root.after(0, self._log, f"❌ Failed to fetch attendance: {str(e)[:80]}", "error")

    def _populate_attendance_tree(self, records: list):
        """Populate attendance treeview with records from sheets."""
        # Clear existing rows
        for item in self.attendance_tree.get_children():
            self.attendance_tree.delete(item)
        
        # If no records, show empty state message
        if not records or len(records) == 0:
            self.stat_cards["present"].config(text="0")
            self.stat_cards["malpractice"].config(text="0")
            self.attendance_tree.insert("", "end",
                values=("No attendance yet", "", "", ""))
            return
        
        # Add records to tree
        for r in records:
            if not r or not isinstance(r, dict):
                continue
            
            student_id = r.get("Student ID", "")
            name = r.get("Name", "")
            time = r.get("Time", "")
            status = r.get("Status", "")
            
            # Skip empty rows
            if not student_id and not name:
                continue
            
            tag = status.lower() if status.lower() in ("present", "malpractice", "late") else ""
            self.attendance_tree.insert("", "end",
                values=(student_id, name, time, status),
                tags=(tag,))
        
        # Update stats
        self._update_stats_from_records(records)

    def _update_stats(self):
        total = len(self.db)
        self.stat_cards["total"].config(text=str(total))

    def _update_stats_from_records(self, records: list):
        present = sum(1 for r in records if r.get("Status") == "Present")
        malp = sum(1 for r in records if r.get("Status") == "Malpractice")
        self.stat_cards["present"].config(text=str(present))
        self.stat_cards["malpractice"].config(text=str(malp))
        self.stat_cards["total"].config(text=str(len(self.db)))

    def _refresh_students(self):
        self.students_tree.delete(*self.students_tree.get_children())
        students = get_registered_students()
        for s in students:
            reg_at = s.get("registered_at", "N/A")
            if "T" in reg_at:
                reg_at = reg_at.split("T")[0]
            self.students_tree.insert("", "end",
                values=(s["id"], s["name"], reg_at))
        self._update_stats()

    def _delete_student(self):
        selection = self.students_tree.selection()
        if not selection:
            messagebox.showinfo("Select Student", "Please select a student to delete.")
            return
        item = self.students_tree.item(selection[0])
        sid, name = item["values"][0], item["values"][1]
        if messagebox.askyesno("Confirm Delete", f"Delete {name} ({sid})?"):
            delete_student(sid)
            self.db = load_encodings()
            self._refresh_students()
            self._log(f"Deleted student: {name} ({sid})", "warning")

    def _clear_logs(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _open_sheet(self):
        if SPREADSHEET_ID and SPREADSHEET_ID != "YOUR_SPREADSHEET_ID_HERE":
            import webbrowser
            webbrowser.open(f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit")
        else:
            messagebox.showinfo("Setup Required",
                                "Please set your SPREADSHEET_ID in app/config.py")

    def __del__(self):
        self.camera_running = False
        self.camera.release()
