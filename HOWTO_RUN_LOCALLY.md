# How to run the Volume count on `183907.mp4`

You'll do this from your own terminal — the sandbox can't install PyTorch fast enough.

## 1. One-time setup

```bash
cd "/Users/suryaprasanthboya/WORK PROJECTS/Video traffic counting"

# Make a virtual env so we don't pollute system Python
python3 -m venv .venv
source .venv/bin/activate

# Install everything
pip install --upgrade pip
pip install -r requirements.txt
```

The first install pulls down PyTorch (~750 MB) and the YOLO weights. Expect 3–10 minutes depending on bandwidth.

If you have an Apple Silicon Mac, PyTorch will automatically use the MPS GPU backend — much faster than CPU.

## 2. Smoke test on 60 seconds of video

Always validate the pipeline end-to-end on a tiny slice before launching a 10-hour run.

```bash
cd 01_Volume/scripts
python count_volume.py --config ../config/config.yaml --max-seconds 60
```

You should see:

- A short download of `yolov8n.pt` the first time (~6 MB)
- Per-frame tracking output
- A printed summary like:
  ```
  Direction      Vehicles
  Direction A    2
  Direction B    1
  TOTAL          3
  ```
- A file at `01_Volume/outputs/volume_counts.xlsx`

Open the Excel and skim the "Raw crossings" sheet — confirm each crossing is a real vehicle and the direction labels feel right.

## 3. Fix the direction labels (if needed)

After the smoke test, you'll know which side the camera-facing traffic ended up on. Open `01_Volume/config/config.yaml` and set real names:

```yaml
direction_names:
  A: "Eastbound"     # or whichever direction came up as A
  B: "Westbound"
```

## 4. Full 10-hour run

```bash
cd 01_Volume/scripts
python count_volume.py --config ../config/config.yaml
```

On CPU only, expect roughly 4–10 hours for a 10-hour 640×480 @ 10fps video using `yolov8n.pt`. On Apple Silicon MPS, more like 1–2 hours. Worth running overnight.

To watch progress while it runs, in another terminal:

```bash
ls -la "/Users/suryaprasanthboya/WORK PROJECTS/Video traffic counting/01_Volume/outputs/"
```

The Excel is only written at the end; if you want intermediate snapshots, ask me to add periodic flushing.

## 5. Speed knobs (if you need it faster)

In `01_Volume/config/config.yaml`:

- `model: yolov8n.pt` — already the smallest. Switch to `yolov8s.pt` for more accuracy at ~2.5× the cost.
- `conf: 0.35` — lower this (e.g. 0.25) if vehicles are being missed; raise it if you're getting false positives on trees/shadows.

Speed flags to add to `shared/utils/tracker.py` if needed:
- `imgsz=640` (default; can drop to 480 for ~30% speedup, slight accuracy hit)
- `half=True` on GPU for FP16 inference

Let me know once the smoke test runs and I'll help interpret the output.
