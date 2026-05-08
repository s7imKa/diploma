# Installation & Running Guide

## System Requirements

- Python 3.7+
- Windows / macOS / Linux
- ~500MB free space
- Intel i3+ or equivalent (faster for SIFT)

## Step 1: Install Python

### Windows

Download from https://www.python.org/downloads/

- During installation, **check** "Add Python to PATH"
- Verify: Open cmd and type `python --version`

### macOS

```bash
brew install python3
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get install python3 python3-pip
```

## Step 2: Clone or Download Project

```bash
cd path/to/projects
git clone <repo-url> cv_matcher_app
# OR download ZIP and extract
```

## Step 3: Create Virtual Environment (Recommended)

### Windows

```cmd
cd cv_matcher_app
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
cd cv_matcher_app
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` prefix in terminal.

## Step 4: Install Dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

This installs:

- PyQt5 (UI framework)
- opencv-python (Computer vision)
- numpy (Numerical operations)

**Installation time:** ~2-5 minutes (depending on internet)

## Step 5: Run the Application

```bash
python main.py
```

**Expected output:**

```
✓ Application starts with main window
✓ Three tabs visible (Image, Video, Navigation)
✓ Dark theme applied
✓ Ready to load images
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'PyQt5'`

**Solution:**

```bash
pip install PyQt5
```

### Issue: `ModuleNotFoundError: No module named 'cv2'`

**Solution:**

```bash
pip install opencv-python
```

### Issue: Application crashes on startup

**Check Python version:**

```bash
python --version  # Should be 3.7+
```

**Try reinstalling:**

```bash
python3 -m pip uninstall PyQt5 opencv-python numpy
python3 -m pip install -r requirements.txt
```

### Issue: "cannot find module" errors

**Make sure you're in the right directory:**

```bash
cd cv_matcher_app
python main.py  # NOT: python ui/main_window.py
```

### Issue: GUI looks broken / fonts are wrong

**This is usually PyQt5 styling issue:**

- Close app
- Delete `__pycache__` folders
- Rerun: `python main.py`

---

## Testing the Installation

### Quick Test

```python
# In Python terminal (python):
>>> import PyQt5
>>> import cv2
>>> import numpy as np
>>> print(f"PyQt5: OK")
>>> print(f"OpenCV: {cv2.__version__}")
>>> print(f"NumPy: {np.__version__}")
```

### Full Test

```bash
python main.py
```

Once app opens:

1. Click "📂 Перше зображення" (File dialog opens)
2. Select any JPG/PNG image
3. Click "📂 Друге зображення"
4. Select another image
5. Click "▶️ Порівняти"
6. Watch results appear in center panel

---

## Performance Notes

### Recommended Image Sizes

| Size     | Quality | Speed            |
| -------- | ------- | ---------------- |
| 100x100  | Low     | ⚡⚡⚡ Very fast |
| 320x240  | Medium  | ⚡⚡ Fast        |
| 640x480  | Good    | ⚡ Normal        |
| 1280x960 | High    | 🐢 Slow          |

### Video Frame Processing Time

Per frame (640x480):

- **ORB:** ~20ms
- **AKAZE:** ~50ms
- **SIFT:** ~200ms

For 30fps video:

- ORB: ~600 frames/min
- AKAZE: ~360 frames/min
- SIFT: ~90 frames/min

---

## Running from Command Line

```bash
# From project root directory
python main.py

# With debug output
python -u main.py 2>&1 | tee debug.log

# In background (Unix)
python main.py &

# With profiling (check speed bottlenecks)
python -m cProfile -s cumtime main.py
```

---

## Uninstallation

### Remove Virtual Environment

```bash
# Windows
rmdir /s venv

# macOS / Linux
rm -rf venv
```

### Clean up

```bash
# Remove cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete
```

---

## Using Different Python Versions

If you have Python 3.8, 3.9, 3.10+ installed:

```bash
# Explicitly use Python 3.9
python3.9 main.py

# Or create venv with specific version
python3.9 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
python main.py
```

---

## Setting up IDE (Optional)

### Visual Studio Code

1. Open folder with cv_matcher_app
2. Install Python extension
3. Select interpreter: `./venv/bin/python` (or `./venv/Scripts/python.exe` on Windows)
4. Press `F5` to run (configure launch.json if needed)

### PyCharm

1. File → Open → cv_matcher_app
2. Configure interpreter: Preferences → Project → Python Interpreter
3. Select or create venv
4. Right-click main.py → Run

### Sublime Text

1. Tools → Build System → New Build System
2. Save as "Python (venv)" with content:

```json
{
    "shell_cmd": "./venv/bin/python -u \"$file\"",
    "file_regex": "^[ ]*File \"(...*?)\", line ([0-9]*)",
    "selector": "source.python"
}
```

---

## File Structure After Installation

```
cv_matcher_app/
├── main.py                           ← Run this
├── README.md                         ← Documentation
├── UX_GUIDE.md                       ← User manual
├── ARCHITECTURE.md                   ← Technical docs
├── requirements.txt                  ← Dependencies
│
├── ui/
│   ├── main_window.py               ← UI code
│   ├── image_viewer.py              ← Image zoom/pan
│   └── __pycache__/
│
├── vision/
│   ├── feature_strategies.py        ← Algorithm selection
│   ├── matcher.py                   ← Core CV logic
│   ├── video_processor.py           ← Video handling
│   └── __pycache__/
│
└── venv/                             ← Virtual environment (auto-created)
    ├── bin/ or Scripts/
    ├── lib/
    └── ...
```

---

## Production Deployment

### Freezing to Executable

Use PyInstaller to create standalone .exe:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed main.py
```

Output: `dist/main.exe` (Windows) or `dist/main` (Unix)

### Docker Image

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```

Build & run:

```bash
docker build -t cv-matcher .
docker run -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY cv-matcher
```

---

## Getting Help

1. **Check UX_GUIDE.md** for usage questions
2. **Check ARCHITECTURE.md** for technical questions
3. **Check **pycache**** — sometimes a full rebuild helps:
    ```bash
    find . -name __pycache__ -exec rm -rf {} +
    python main.py
    ```
4. **OpenCV docs:** https://docs.opencv.org/
5. **PyQt5 docs:** https://www.riverbankcomputing.com/static/Docs/PyQt5/

---

## Success Checklist

- ✓ Python 3.7+ installed
- ✓ Virtual environment activated
- ✓ requirements.txt installed
- ✓ `python main.py` runs without errors
- ✓ GUI window opens with 3 tabs
- ✓ Can load images and compare them
- ✓ Results display correctly

**You're ready to use Feature Matcher Studio!** 🎉
