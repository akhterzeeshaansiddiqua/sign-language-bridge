"""
collect_data.py
===============
Collect hand landmark training data for gesture recognition.

For each gesture label, captures N samples via webcam and saves the
63-float landmark vectors to data/landmarks/<label>.csv.

Usage:
    python collect_data.py --label hello --samples 200
    python collect_data.py --label thank_you --samples 200 --cam 1
"""

import argparse
import os
import csv
import cv2

from utils.hand_detection import HandDetector


def collect(args):
    detector = HandDetector()
    cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.cam}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{args.label}.csv")
    existing = 0

    # Count existing samples
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            existing = sum(1 for _ in f)
        print(f"[INFO] Found {existing} existing samples for '{args.label}'.")

    collected = 0
    recording = False

    print(f"\n[INFO] Target: {args.samples} new samples for gesture: '{args.label}'")
    print("[INFO] Press SPACE to start/stop recording | Q to quit early\n")

    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)

        while collected < args.samples:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            landmarks, annotated = detector.detect(frame)

            # Status overlay
            status = "RECORDING" if recording else "PAUSED — Press SPACE"
            color = (0, 200, 80) if recording else (0, 180, 255)
            cv2.putText(annotated, status, (15, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
            cv2.putText(annotated, f"Label: {args.label}", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(annotated,
                        f"Collected: {collected} / {args.samples}",
                        (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # Save sample when recording and hand detected
            if recording and landmarks:
                writer.writerow(landmarks)
                collected += 1

            cv2.imshow("Data Collection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = not recording
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    total = existing + collected
    print(f"\n[DONE] Saved {collected} new samples → {csv_path}")
    print(f"[INFO] Total samples for '{args.label}': {total}")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect gesture landmark training data")
    parser.add_argument("--label", required=True, help="Gesture label (e.g. hello, stop)")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to collect")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    parser.add_argument("--output-dir", default="data/landmarks", help="Output CSV directory")
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
