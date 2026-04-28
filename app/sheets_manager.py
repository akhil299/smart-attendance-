"""
sheets_manager.py - Google Sheets Live Attendance Integration
=============================================================
Handles:
  - Authentication with Google Sheets API
  - Auto-creating daily attendance sheets
  - Marking/updating student attendance in real time
"""

import os
import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials
from app.config import SPREADSHEET_ID, SHEET_SCOPE, CREDENTIALS_PATH


# ─── Auth & Connection ────────────────────────────────────────────────────────

_client = None

def get_client():
    """Get or create authenticated gspread client."""
    global _client
    if _client is None:
        if not os.path.exists(CREDENTIALS_PATH):
            print(f"❌ Google credentials not found at: {CREDENTIALS_PATH}")
            print("Please follow setup instructions in README.md")
            raise FileNotFoundError(
                f"Google credentials not found at: {CREDENTIALS_PATH}\n"
                "Please follow setup instructions in README.md"
            )
        try:
            creds = Credentials.from_service_account_file(CREDENTIALS_PATH, scopes=SHEET_SCOPE)
            _client = gspread.authorize(creds)
            print("[✓ Sheets] Successfully authenticated with Google Sheets API")
        except Exception as e:
            print(f"❌ Failed to authenticate with Google Sheets: {e}")
            raise
    return _client


def get_spreadsheet():
    """Open the attendance spreadsheet."""
    try:
        client = get_client()
        sheet = client.open_by_key(SPREADSHEET_ID)
        print(f"[✓ Sheets] Opened spreadsheet: {SPREADSHEET_ID}")
        return sheet
    except Exception as e:
        print(f"❌ Failed to open spreadsheet {SPREADSHEET_ID}: {e}")
        raise


# ─── Sheet Management ─────────────────────────────────────────────────────────

def get_or_create_daily_sheet(date_str: str = None):
    """
    Get today's attendance sheet, or create it if it doesn't exist.

    Sheet name format: "DD-MM-YYYY"
    Columns: Student ID | Name | Time | Status | Notes
    """
    if date_str is None:
        date_str = datetime.now().strftime("%d-%m-%Y")

    spreadsheet = get_spreadsheet()
    sheet_titles = [ws.title for ws in spreadsheet.worksheets()]

    if date_str not in sheet_titles:
        # Create new sheet for today
        sheet = spreadsheet.add_worksheet(title=date_str, rows=200, cols=6)

        # Add headers with formatting
        headers = ["Student ID", "Name", "Time", "Date", "Status", "Notes"]
        sheet.append_row(headers)

        # Bold header row
        sheet.format("A1:F1", {
            "textFormat": {"bold": True, "fontSize": 11},
            "backgroundColor": {"red": 0.13, "green": 0.13, "blue": 0.18},
        })

        # Set column widths
        sheet.format("A1:A200", {"horizontalAlignment": "CENTER"})
        sheet.freeze(rows=1)

        print(f"[Sheets] Created new sheet: {date_str}")
    else:
        sheet = spreadsheet.worksheet(date_str)

    return sheet


def mark_attendance(student_id: str, name: str, status: str = "Present", notes: str = "") -> dict:
    """
    Mark a student's attendance in today's Google Sheet.

    Args:
        student_id: Student's unique ID
        name: Student's full name
        status: "Present", "Late", "Malpractice"
        notes: Additional notes (e.g. malpractice reason)

    Returns:
        dict with success status and row number
    """
    try:
        sheet = get_or_create_daily_sheet()
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%d-%m-%Y")

        # Check if student already marked today
        records = sheet.get_all_records()
        for i, record in enumerate(records):
            if str(record.get("Student ID", "")) == str(student_id):
                # Update existing row
                row_num = i + 2  # +2 for header and 0-index
                sheet.update(f"C{row_num}:F{row_num}", [[time_str, date_str, status, notes]])

                # Color-code by status
                _apply_status_color(sheet, row_num, status)
                
                print(f"[✓ Sheets] Updated attendance for {name} ({student_id}): {status}")

                return {
                    "success": True,
                    "action": "updated",
                    "row": row_num,
                    "message": f"Updated {name}'s attendance to {status}"
                }

        # New entry
        row_data = [student_id, name, time_str, date_str, status, notes]
        sheet.append_row(row_data)

        # Get the row we just added
        all_values = sheet.get_all_values()
        row_num = len(all_values)
        _apply_status_color(sheet, row_num, status)
        
        print(f"[✓ Sheets] Marked {name} ({student_id}) as {status}")

        return {
            "success": True,
            "action": "added",
            "row": row_num,
            "message": f"Marked {name} as {status}"
        }

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Failed to mark attendance for {name}: {error_msg}")
        return {
            "success": False,
            "action": "error",
            "message": f"Sheets error: {error_msg}"
        }


def _apply_status_color(sheet, row_num: int, status: str):
    """Apply background color to row based on attendance status."""
    color_map = {
        "Present":    {"red": 0.08, "green": 0.27, "blue": 0.17},   # Dark green
        "Late":       {"red": 0.30, "green": 0.20, "blue": 0.05},   # Dark orange
        "Malpractice":{"red": 0.35, "green": 0.05, "blue": 0.05},   # Dark red
        "Absent":     {"red": 0.20, "green": 0.05, "blue": 0.05},   # Muted red
    }
    color = color_map.get(status, {"red": 0.1, "green": 0.1, "blue": 0.15})
    try:
        sheet.format(f"A{row_num}:F{row_num}", {
            "backgroundColor": color,
            "textFormat": {"foregroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9}}
        })
    except Exception:
        pass  # Formatting failure shouldn't break attendance marking


def get_today_attendance() -> list:
    """
    Fetch all attendance records for today.

    Returns list of dicts with attendance data.
    """
    try:
        sheet = get_or_create_daily_sheet()
        records = sheet.get_all_records()
        print(f"[✓ Sheets] Fetched {len(records)} attendance records")
        if records:
            print(f"[Sheets] Sample record: {records[0]}")
        return records
    except Exception as e:
        print(f"❌ Sheets: Error fetching records: {e}")
        return []


def get_attendance_summary() -> dict:
    """Get today's attendance count by status."""
    records = get_today_attendance()
    summary = {"Present": 0, "Late": 0, "Malpractice": 0, "Total": len(records)}
    for r in records:
        status = r.get("Status", "")
        if status in summary:
            summary[status] += 1
    return summary


def log_malpractice(student_id: str, name: str, reason: str, snapshot_path: str = ""):
    """Log a malpractice incident with evidence."""
    notes = f"MALPRACTICE: {reason}"
    if snapshot_path:
        notes += f" | Snapshot: {os.path.basename(snapshot_path)}"
    return mark_attendance(student_id, name, status="Malpractice", notes=notes)


def check_already_marked(student_id: str) -> bool:
    """Check if student is already marked present today."""
    try:
        records = get_today_attendance()
        for record in records:
            if str(record.get("Student ID", "")) == str(student_id):
                return True
        return False
    except Exception:
        return False
