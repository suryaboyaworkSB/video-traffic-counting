# Running the project on Windows

This is the Windows-equivalent of `HOWTO_RUN_LOCALLY.md`. Same code, same workflow, just different shell syntax.

## 1. Prerequisites

Install these on the Windows machine if you don't already have them:

- **Python 3.10 or newer**: <https://www.python.org/downloads/windows/>  
  During install, check **"Add python.exe to PATH"** at the bottom of the first installer screen.
- **FFmpeg** (only needed if you want to extract test frames or write debug videos):  
  Easiest path: `winget install ffmpeg` in PowerShell, or download a build from <https://www.gyan.dev/ffmpeg/builds/> and add `bin\` to PATH.
- Optional: **Git** if you want version control.

Open **PowerShell** (right-click Start → "Windows PowerShell") and verify:

```powershell
python --version          # should print 3.10+ 
ffmpeg -version           # only needed for debug video extraction
```

## 2. Unzip the project

Extract the zip to `C:\Users\sboya\Documents\trafficCounting\`. Important: the zip's top-level folder is `Video traffic counting/`, so after extracting your final structure should be exactly:

```
C:\Users\sboya\Documents\trafficCounting\Video traffic counting\
```

If you'd prefer the project files to sit directly under `trafficCounting\` without the inner folder, extract first then move the contents up one level — but you don't have to.

After extracting, the inner folder structure looks like:

```
C:\Users\sboya\Documents\trafficCounting\
├── 01_Volume\
│   ├── config\
│   ├── scripts\
│   ├── videos\         (empty — you'll drop your .mp4 here)
│   └── outputs\        (empty — results go here)
├── 02_Class_Count\
├── 03_TMC_Class_Count\
├── shared\
├── requirements.txt
└── ...
```

## 3. Drop the video in

Copy your `.mp4` (e.g. `183907.mp4`) into:

```
C:\Users\sboya\Documents\trafficCounting\01_Volume\videos\
```

## 4. Create a virtual environment and install packages

In PowerShell:

```powershell
cd C:\TrafficCounting
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `Activate.ps1` errors with "running scripts is disabled", run this once and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

First install takes 3–10 minutes (PyTorch is large). On Windows with an NVIDIA GPU, you'll want CUDA PyTorch instead of CPU PyTorch — see the optional GPU section below.

## 5. Smoke test on 60 seconds

```powershell
cd C:\Users\sboya\Documents\trafficCounting\01_Volume\scripts
python count_volume.py --config ..\config\config.yaml --max-seconds 60
```

Expected: a printed direction summary (something like Direction A 4, Direction B 5, TOTAL 9 for the daylight clip) and a fresh `..\outputs\volume_counts.xlsx`.

## 6. Run the full daytime YOLO count

```powershell
python count_volume.py --config ..\config\config.yaml
```

Time on CPU: very long, often 4–10× real-time. Best to run overnight.

## 7. Night-mode test (optional, after daytime works)

```powershell
python count_volume_night.py --config ..\config\config.yaml --start-seconds 19800 --max-seconds 60 --debug-video
```

Output: `..\outputs\volume_counts_night.xlsx` and `..\outputs\debug_night.mp4`.

## 8. Auto day/night routing for the full video

```powershell
python count_volume_auto.py --config ..\config\config.yaml
```

This samples each minute of video, classifies it as day or night by brightness, and dispatches to the right detector. Final output: `..\outputs\volume_counts_auto.xlsx`.

## Optional: GPU acceleration on Windows with NVIDIA

If your Windows machine has an NVIDIA GPU, install CUDA-enabled PyTorch (10–30× faster):

```powershell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Test with:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

If `True`, YOLO will use the GPU automatically.

## Path differences vs macOS

- Use backslashes `\` in Windows paths, or forward slashes `/` (Python accepts both).
- Activate venv with `.\.venv\Scripts\Activate.ps1` (PowerShell) or `.\.venv\Scripts\activate.bat` (CMD).
- The configs use relative paths (`../videos/...`) which work the same on both platforms.

## Troubleshooting

**"ffmpeg not found"** when running `debug_visualize.py` — install FFmpeg (step 1) or skip the debug video.

**Permission errors writing to `outputs\`** — make sure you've extracted the project outside any system-protected directory (e.g. avoid `C:\Program Files\`).

**Out of memory on YOLO** — switch from `yolov8m.pt` back to `yolov8n.pt` in `01_Volume/config/config.yaml`; you'll lose some accuracy on dark vehicles but it'll fit on smaller machines.

**Antivirus / SmartScreen blocking `yolov8m.pt`** download — let it through, the file is from Ultralytics' official source.

## What to do once it's running

When you're confident the daytime count looks right, kick off the full 10-hour run before bed. In the morning you'll have `volume_counts.xlsx` ready. If you want day + night merged, use `count_volume_auto.py` instead.

Ping me with the printed direction summary after the smoke test and we can validate together.
