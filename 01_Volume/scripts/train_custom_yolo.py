"""
Fine-tune YOLOv8m on the camera-specific labeled dataset.

This is a SINGLE command that loads the pretrained yolov8m weights, then
fine-tunes them on your 1000 labeled frames. Fine-tuning (vs training from
scratch) is the right call here because:

  1. We only have 1000 frames - too few to train from scratch
  2. yolov8m already knows what 'vehicle' looks like in general
  3. Fine-tuning adapts the pretrained features to THIS specific camera

The training learns the camera's quirks: oblique angle, low resolution,
dawn/dusk lighting, EB-lane occlusion, etc. After 50-100 epochs, the
fine-tuned model should detect vehicles with 95%+ accuracy on this camera.

Output:
  runs/detect/train/weights/best.pt   - the trained model checkpoint
  runs/detect/train/weights/last.pt   - the final epoch checkpoint
  runs/detect/train/results.png       - training curves
  runs/detect/train/confusion_matrix.png - validation confusion matrix

Usage:
    python train_custom_yolo.py
    python train_custom_yolo.py --epochs 50 --batch 4   # smaller/faster
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../training/dataset/data.yaml",
                   help="Path to data.yaml from Roboflow export")
    p.add_argument("--epochs", type=int, default=100,
                   help="Max training epochs (early stops if validation plateaus)")
    p.add_argument("--batch", type=int, default=8,
                   help="Batch size. Lower if you run out of memory.")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Training image size (must match dataset resize)")
    p.add_argument("--patience", type=int, default=20,
                   help="Stop early if no validation improvement for N epochs")
    p.add_argument("--device", default="cpu",
                   help="Training device: 'cpu', '0' (CUDA), or 'intel:gpu' "
                        "(OpenVINO doesn't support training - use CPU)")
    p.add_argument("--name", default="traffic_yolov8m",
                   help="Run name (output goes to runs/detect/<name>/)")
    args = p.parse_args()

    # Resolve data path
    data_path = (Path(__file__).parent / args.data).resolve()
    if not data_path.exists():
        print(f"ERROR: data.yaml not found at {data_path}")
        print("Make sure you've extracted the Roboflow zip to:")
        print("  01_Volume/training/dataset/")
        return

    print(f"Dataset: {data_path}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, ImgSize: {args.imgsz}")
    print(f"Device: {args.device}")
    print()

    from ultralytics import YOLO

    # Load pretrained yolov8m weights as the starting point
    print("Loading yolov8m.pt as starting weights...")
    model = YOLO("yolov8m.pt")

    print("Starting fine-tuning...")
    print()

    # Fine-tune
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        # Lower initial learning rate than from-scratch training to avoid
        # destroying the pretrained features
        lr0=0.001,
        lrf=0.01,            # final lr = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Validation and saving
        val=True,
        save=True,
        save_period=10,      # save checkpoint every 10 epochs
        # Output location
        project="runs/detect",
        name=args.name,
        exist_ok=False,      # don't overwrite previous runs
        # Logging
        verbose=True,
        plots=True,          # save training curves
        # Reduce workers if you have memory issues
        workers=4,
        # Cache disables for memory-constrained setups; enable if you have RAM
        cache=False,
        # Augmentation (already done in Roboflow, but a few more help)
        mosaic=1.0,          # combine 4 images into one for richer scenes
        mixup=0.1,           # blend pairs of images (improves robustness)
        copy_paste=0.0,      # off; we don't have segmentation labels
        # Don't freeze any layers - we want all weights to adapt
        freeze=None,
    )

    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print()
    print(f"Best model: runs/detect/{args.name}/weights/best.pt")
    print(f"Last model: runs/detect/{args.name}/weights/last.pt")
    print()
    print("Validation metrics (final epoch):")
    print(f"  mAP50:    {results.box.map50:.4f}  (should be > 0.85 for good model)")
    print(f"  mAP50-95: {results.box.map:.4f}    (should be > 0.50)")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall:    {results.box.mr:.4f}    (recall is what we need high)")
    print()
    print("Next step: copy best.pt to your project root, then update config.yaml:")
    print("  model: runs/detect/" + args.name + "/weights/best.pt")
    print()
    print("Or copy it:")
    print(f"  copy runs\\detect\\{args.name}\\weights\\best.pt yolov8m_traffic.pt")


if __name__ == "__main__":
    main()
