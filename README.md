# Smart Attendance System 🎓
> Face Recognition · Iris Detection · Anti-Malpractice · Google Sheets · Dark UI

---

## 📦 Installation

### 1. Install Python dependencies
```bash
pip install opencv-python dlib face-recognition gspread google-auth pillow numpy scipy imutils
```

> ⚠️ dlib requires CMake. On Windows: `pip install cmake` first.
> On Ubuntu: `sudo apt install cmake libopenblas-dev liblapack-dev`

---

### 2. Download dlib models (required!)
Place both files in the `models/` folder:

| File | Download Link |
|------|--------------|
| `shape_predictor_68_face_landmarks.dat` | http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 |
| `dlib_face_recognition_resnet_model_v1.dat` | http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 |

Extract .bz2 files and place .dat files in `models/`

---

### 3. Google Sheets Setup

1. Go to https://console.cloud.google.com
2. Create a new project → Enable **Google Sheets API** and **Google Drive API**
3. Create a **Service Account** → Download JSON key
4. Rename the JSON file to `credentials.json` and place in project root
5. Create a Google Sheet → Copy the Spreadsheet ID from the URL:
   ```
   https://docs.google.com/spreadsheets/d/THIS_IS_THE_ID/edit
   ```
6. Open `app/config.py` and set:
   ```python
   SPREADSHEET_ID = "your_actual_id_here"
   ```
7. **Share your Google Sheet** with the service account email
   (found in credentials.json as `client_email`)

---

## 🚀 Run

```bash
python main.py
```

---

## 📁 Project Structure

```
smart_attendance/
├── main.py                    # Entry point
├── credentials.json           # Google API credentials (you add this)
├── README.md
│
├── app/
│   ├── config.py              # All settings & constants
│   ├── gui.py                 # Tkinter dark UI
│   ├── camera.py              # Camera + low-light processing
│   ├── face_engine.py         # Face recognition (dlib)
│   ├── iris_detector.py       # Iris + blink liveness detection
│   ├── anti_malpractice.py    # Spoof/phone detection
│   └── sheets_manager.py      # Google Sheets live integration
│
├── models/
│   ├── shape_predictor_68_face_landmarks.dat   ← you download
│   └── dlib_face_recognition_resnet_model_v1.dat ← you download
│
└── data/
    ├── encodings.pkl              # Face encodings database
    ├── registered_faces/          # Saved face images
    └── malpractice_snapshots/     # Evidence photos
```

---

## 🎯 Features

### ✅ Face Registration
- Switch to **Register Mode** in the camera panel
- Enter Student ID and Name
- Click **Start Capture** — system auto-captures 5 frames
- Click **Register** to save

### ✅ Attendance Marking
- Keep camera in **Attendance Mode** (default)
- System recognizes faces and marks attendance automatically
- Live update to Google Sheets with timestamp

### ✅ Iris Detection
- Blink detection using Eye Aspect Ratio (EAR)
- Iris circle detection via Hough Circles
- Student must blink once to confirm liveness

### 🚨 Anti-Malpractice
| Check | Method |
|-------|--------|
| Printed photo | Texture variance analysis |
| Phone screen | Screen glow + blue-shift detection |
| Moiré pattern | FFT frequency analysis |
| Phone object | Contour + rectangle detection |
| Flat image | Depth/3D luminance analysis |

> Requires **2+ signals** before flagging (reduces false positives)
> Evidence snapshot saved to `data/malpractice_snapshots/`

### ⚡ Low Light Mode
- Auto-detects dark environment (brightness < 60)
- CLAHE enhancement on luminance channel
- Noise reduction + warm color grading applied

---

## ⚙️ Configuration (`app/config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `FACE_RECOGNITION_TOLERANCE` | 0.45 | Lower = stricter matching |
| `LIVENESS_REQUIRED_BLINKS` | 1 | Blinks needed to confirm liveness |
| `LIVENESS_TIMEOUT_SECONDS` | 8 | Seconds to complete liveness check |
| `TEXTURE_VARIANCE_THRESHOLD` | 180 | Spoof detection sensitivity |
| `LOW_LIGHT_THRESHOLD` | 60 | Brightness below = enhance |

---

## 🛠 Troubleshooting

**Camera not opening**: Check webcam index in `Camera(0)` — try `Camera(1)`

**dlib import error**: Install cmake first, then reinstall dlib

**Google Sheets not connecting**: Ensure credentials.json is valid and sheet is shared with service account email

**Face not detecting**: Ensure models/*.dat files are downloaded and extracted
