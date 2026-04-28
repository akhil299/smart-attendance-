"""
Smart Attendance System - Main Entry Point
==========================================
Run this file to start the application.

Requirements:
    pip install opencv-python dlib face-recognition gspread google-auth
                pillow numpy scipy imutils

Setup:
    1. Place your Google Sheets credentials JSON as 'credentials.json' in root folder
    2. Update SPREADSHEET_ID in app/config.py
    3. Run: python main.py
"""

import tkinter as tk
from app.gui import SmartAttendanceApp

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartAttendanceApp(root)
    root.mainloop()
