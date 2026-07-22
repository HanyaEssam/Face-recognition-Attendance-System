# Face Recognition Attendance — POC

## Step 1 — Open the project in VS Code
1. Unzip/copy this folder somewhere on your machine.
2. Open VS Code → File → Open Folder → select `face_attendance_poc`.
3. Install the **Python extension** for VS Code if you don't have it (Extensions panel, search "Python").

## Step 2 — Create a virtual environment
Open a terminal in VS Code (`` Ctrl+` ``) and run:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

In VS Code, select this venv as your interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → pick the one inside `venv`.

## Step 3 — Download the pretrained models
These are the real YuNet + SFace ONNX weights from the OpenCV Zoo (no training involved):

```bash
python download_models.py
```

This creates `models/face_detection_yunet.onnx` and `models/face_recognition_sface.onnx`. If either file ends up tiny (~1KB), GitHub served a Git-LFS pointer instead of the binary — in that case open the URL printed in the script directly in your browser and use the "Download raw file" button, then save it to the same path.

## Step 4 — Enroll mock employees
With your webcam connected:

```bash
python enroll.py
```

Follow the prompts (name, department, shift start time), then press `c` in the camera window ~5 times to capture different angles, `q` when done. Repeat this for 10–15 mock employees — you can literally do this with willing coworkers/friends standing in for "employees."

## Step 5 — Seed some fake attendance history (optional but recommended)
So the dashboard doesn't look empty on first open:

```bash
python seed_mock_data.py
```

This generates ~3 weeks of simulated check-in/out times (with some late arrivals and absences worked in) for whoever you've enrolled.

## Step 6 — Run the dashboard

```bash
streamlit run app.py
```

This opens a browser tab with four tabs:
- **Check In / Out** — take a photo, pick check-in or check-out, see the recognition result live
- **Employees** — list of enrolled mock employees
- **Attendance Log** — filterable table of all records
- **Summary** — on-time/late percentages and charts, for the manager-facing view

## Project structure
```
face_attendance_poc/
├── requirements.txt
├── download_models.py      # fetches YuNet + SFace .onnx weights
├── db.py                    # SQLite schema + attendance logic
├── face_pipeline.py         # YuNet detection + SFace recognition wrapper
├── enroll.py                 # CLI tool to register a new mock employee
├── seed_mock_data.py         # generates fake historical attendance
├── app.py                    # Streamlit dashboard
└── models/                   # (created by download_models.py)
```

## Known limitations to flag to your manager
- **Liveness detection is a placeholder** right now (`LivenessChecker` in `face_pipeline.py` always returns "live"). Swapping in MiniFASNet (from `minivision-ai/Silent-Face-Anti-Spoofing` on GitHub) is a Phase 2 item — the interface is already there, just needs the model plugged in.
- The SFace match threshold (`0.363`) is OpenCV's documented default; you may want to tune it after testing with your specific mock employees and lighting.
- Embeddings are stored as raw SQLite BLOBs, which is fine for a POC (tens of employees) but not how you'd scale this to hundreds/thousands.
