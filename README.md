# Video Traffic Counting

A working folder for processing traffic video and producing three kinds of counts that traffic engineers typically need.

## Three sections

```
01_Volume/             Total vehicle volume per count line, ignoring class
02_Class_Count/        Per-class counts (car, truck, bus, motorcycle, bicycle)
03_TMC_Class_Count/    Turning Movement Counts split by vehicle class
```

Each section has the same layout: `videos/` for input footage, `config/` for the count-line / zone geometry, `scripts/` for the detection script, and `outputs/` for the resulting Excel workbook.

A `shared/` folder holds the YOLO+ByteTrack tracking wrapper, vehicle-class mapping, zone math, and time-bin helpers that all three sections reuse.

## Tech stack

Python with Ultralytics YOLO (v8/v11) for detection, the built-in ByteTrack tracker for stable IDs across frames, OpenCV for video I/O, and pandas + xlsxwriter for the Excel outputs.

Install with:

```
pip install -r requirements.txt
```

## How a study runs

1. Drop the video in the appropriate section's `videos/` folder.
2. Copy `config.example.yaml` to `config.yaml` and edit the line/zone pixel coordinates to match the camera angle. Use any frame extractor or a notebook to pick the coordinates.
3. Run the counter:

```
cd 01_Volume/scripts && python count_volume.py --config ../config/config.yaml
cd 02_Class_Count/scripts && python count_class.py --config ../config/config.yaml
cd 03_TMC_Class_Count/scripts && python count_tmc.py --config ../config/config.yaml
```

4. The script writes an `.xlsx` to that section's `outputs/` folder. Each workbook has both a binned summary (15-min by default) and the raw per-vehicle rows.

## Output shape

Volume: rows = 15-min bin, columns = each count line, plus a Total column.

Class: rows = (bin, line), columns = vehicle category (Passenger Car, Truck, Bus, Motorcycle, Bicycle), plus a Total.

TMC: one sheet of (bin, entry leg) by movement (Through / Left / Right / U-turn), one sheet of (entry, movement) by vehicle class, and a raw movements sheet.

## Next steps

The example configs use placeholder pixel coordinates. Before running on real footage, open the first frame of your video and pick coordinates for the count lines (Volume / Class) or approach polygons (TMC). A small helper script for clicking points and writing them into the YAML is a natural next thing to add.
