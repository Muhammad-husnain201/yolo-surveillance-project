"""Task 7: Real-time webcam multi-object detection dashboard.

Keys:  q = quit   s = save current frame to disk
Usage: python live_detection.py [--cam 0] [--model yolov8n.pt] [--conf 0.4]
"""
import argparse
import os
import time
from datetime import datetime

os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")  # Windows camera fix
import cv2
from ultralytics import YOLO


def open_camera(index):
    """Try DirectShow first (most reliable on Windows), then default backends."""
    for backend in (getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY), cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            for _ in range(10):          # warm-up: first frames are often empty
                cap.read()
            return cap
        cap.release()
    raise SystemExit(f"Cannot open camera {index}. Close other apps using it, "
                     "check Windows camera privacy settings, or try --cam 1.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0, help="camera index")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--out", default="captures", help="folder for saved frames")
    return p.parse_args()


def draw_detections(frame, result):
    for b in result.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        label = f"{result.names[int(b.cls[0])]} {float(b.conf[0]):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.model)

    cap = open_camera(args.cam)

    fps, prev = 0.0, time.perf_counter()
    fails = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    print("Frame grab failed repeatedly; exiting.")
                    break
                continue
            fails = 0

            result = model(frame, conf=args.conf, verbose=False)[0]
            frame = draw_detections(frame, result)

            now = time.perf_counter()
            inst = 1.0 / max(now - prev, 1e-9)
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst  # smoothed
            prev = now
            draw_fps(frame, fps)

            cv2.imshow("Live Detection (q=quit, s=save)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = os.path.join(args.out, datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg")
                cv2.imwrite(path, frame)
                print("Saved", path)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
