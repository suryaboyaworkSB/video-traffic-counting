"""
Interactive count-line picker.

Click twice on the image to define the line (start point, then end point).
Press 'r' to reset, 's' to save, 'q' to quit.

Usage:
    python3 pick_line.py --video path/to/video.mp4
    python3 pick_line.py --video path/to/video.mp4 --frame 500
"""

import argparse
import cv2

points = []


def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--frame", type=int, default=100, help="Frame index to display")
    ap.add_argument("--output", default="line_preview.png", help="Where to save annotated frame")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: cannot read frame {args.frame}")
        return

    h, w = frame.shape[:2]
    print(f"\nFrame size: {w} x {h}")
    print("Click TWO points on the road where you want the count line.")
    print("Keys: r=reset  s=save  q=quit\n")

    cv2.namedWindow("pick_line", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("pick_line", on_mouse)

    global points
    while True:
        disp = frame.copy()
        for i, pt in enumerate(points):
            cv2.circle(disp, pt, 6, (0, 255, 0), -1)
            cv2.putText(disp, str(i + 1), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(points) == 2:
            cv2.line(disp, points[0], points[1], (0, 255, 0), 3)
            cv2.putText(disp,
                        f"points: [{points[0]}, {points[1]}]",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("pick_line", disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            points = []
            print("\nReset.\n")
        if key == ord("s") and len(points) == 2:
            cv2.imwrite(args.output, disp)
            print(f"\nSaved annotated frame to: {args.output}")
            print(f"\nConfig YAML snippet:")
            print(f"lines:")
            print(f"  - name: Main")
            print(f"    points: [[{points[0][0]}, {points[0][1]}], "
                  f"[{points[1][0]}, {points[1][1]}]]")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
