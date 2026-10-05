"""Task 8: Edge-triggered webcam surveillance with event logging.

When a priority class is seen with confidence > 0.65, a snapshot is saved to
alerts/YYYYMMDD_HHMMSS.jpg and a CSV line is appended to events.log:
    timestamp,class,confidence,x1,y1,x2,y2

Keys: q = quit
Usage: python surveillance_logger.py [--cam 0] [--classes person "cell phone"]
"""
import argparse
import csv
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

ALERT_DIR = "alerts"
LOG_FILE = "events.log"
CONF_THRES = 0.65
COOLDOWN_S = 2.0        # min seconds between snapshots (avoids one per frame)
INDICATOR_S = 1.5       # how long the on-screen indicator stays visible


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--classes", nargs="+", default=["person", "cell phone"])
    return p.parse_args()


def init_log():
    new = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
    f = open(LOG_FILE, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp", "class", "confidence", "x1", "y1", "x2", "y2"])
        f.flush()
    return f, w


def main():
    args = parse_args()
    targets = set(args.classes)
    os.makedirs(ALERT_DIR, exist_ok=True)
    model = YOLO(args.model)
    log_f, log_w = init_log()

    cap = open_camera(args.cam)

    last_save = 0.0
    fails = 0
    indicator_until = 0.0
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
            raw = frame.copy()                    # clean copy for the snapshot
            result = model(frame, conf=0.25, verbose=False)[0]

            triggers = []
            for b in result.boxes:
                name, conf = result.names[int(b.cls[0])], float(b.conf[0])
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                hit = name in targets and conf > CONF_THRES
                color = (0, 0, 255) if hit else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                if hit:
                    triggers.append((name, conf, x1, y1, x2, y2))

            now = time.time()
            if triggers and now - last_save >= COOLDOWN_S:
                stamp = datetime.now()
                path = os.path.join(ALERT_DIR, stamp.strftime("%Y%m%d_%H%M%S") + ".jpg")
                cv2.imwrite(path, raw)
                for name, conf, x1, y1, x2, y2 in triggers:
                    log_w.writerow([stamp.strftime("%Y-%m-%d %H:%M:%S"), name,
                                    f"{conf:.3f}", x1, y1, x2, y2])
                log_f.flush()
                last_save = now
                indicator_until = now + INDICATOR_S
                print(f"[ALERT] {len(triggers)} event(s) -> {path}")

            if now < indicator_until:               # visual indicator
                cv2.circle(frame, (frame.shape[1] - 30, 30), 12, (0, 0, 255), -1)
                cv2.putText(frame, "LOGGING EVENT", (frame.shape[1] - 230, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1),
                              (0, 0, 255), 4)

            cv2.imshow("Surveillance (q=quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_f.close()


if __name__ == "__main__":
    main()
