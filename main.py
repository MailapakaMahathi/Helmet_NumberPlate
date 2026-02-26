import cv2
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.detector import HelmetDetector
from app.ocr_plate import NumberPlateReader
from app.challan import ChallanGenerator

# ─── CONFIG ───────────────────────────────────
MODEL_PATH      = "model/best.pt"
VIDEO_SOURCE    = "video3.mp4"
OUTPUT_PATH     = "output.mp4"
SAVE_VIOLATIONS = True
VIOLATIONS_DIR  = "violations"
# ──────────────────────────────────────────────

def most_common_plate(plate_list):
    """Return most frequently seen plate text."""
    if not plate_list:
        return "UNREADABLE"
    # Find most common
    counter = Counter(plate_list)
    return counter.most_common(1)[0][0]

def main():
    print("Loading model...")
    detector = HelmetDetector(MODEL_PATH)

    print("Initializing OCR...")
    ocr = NumberPlateReader()

    challan_gen = ChallanGenerator()
    os.makedirs(VIOLATIONS_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {VIDEO_SOURCE}")
        return

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    print(f"✅ Video loaded: {VIDEO_SOURCE}")
    print(f"   FPS:{fps} Size:{width}x{height} Frames:{total}")
    print(f"\n✅ Processing...\n")

    frame_count      = 0
    challan_generated = False  # ✅ Only ONE challan per video
    plate_readings   = []      # Collect plate readings across frames
    best_frame       = None    # Save best violation frame
    best_plate_bbox  = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}/{total}", end="\r")

        # Process every 3rd frame
        if frame_count % 3 != 0:
            out.write(frame)
            continue

        # Step 1: Detect
        detections = detector.detect(frame)

        # Step 2: Check violation and plate
        has_violation = False
        plate_bbox    = None

        for det in detections:
            class_name = det['class'].lower()
            if class_name == 'without helmet':
                has_violation = True
            if class_name == 'number plate':
                plate_bbox = det['bbox']

        # Step 3: Collect plate readings (don't generate challan yet)
        if has_violation and plate_bbox and not challan_generated:
            plate_text, plate_bbox = ocr.extract_plate_text(frame, plate_bbox)

            if plate_text != "UNREADABLE":
                plate_readings.append(plate_text)
                best_frame      = frame.copy()
                best_plate_bbox = plate_bbox
                print(f"\n   📋 Plate reading #{len(plate_readings)}: {plate_text}")

            # After collecting 5 readings pick the most common one
            if len(plate_readings) >= 5 and not challan_generated:
                final_plate = most_common_plate(plate_readings)
                print(f"\n🚨 FINAL PLATE: {final_plate}")

                # Draw on best frame
                if best_plate_bbox:
                    x1, y1, x2, y2 = map(int, best_plate_bbox)
                    cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    label = f"PLATE: {final_plate}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    cv2.rectangle(best_frame,
                                 (x1, y2 + 5),
                                 (x1 + text_size[0] + 10, y2 + 35),
                                 (0, 255, 255), -1)
                    cv2.putText(best_frame, label,
                               (x1 + 5, y2 + 28),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.8, (0, 0, 0), 2)

                cv2.rectangle(best_frame, (0, 0),
                             (best_frame.shape[1]-1, best_frame.shape[0]-1),
                             (0, 0, 255), 5)
                cv2.putText(best_frame, "VIOLATION DETECTED",
                           (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                           0.9, (0, 0, 255), 2)
                cv2.putText(best_frame, f"Plate: {final_plate}",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                           0.9, (0, 0, 255), 2)

                # Save violation image
                violation_img = f"{VIOLATIONS_DIR}/violation_{final_plate}.jpg"
                cv2.imwrite(violation_img, best_frame)

                # Generate ONE challan
                challan_gen.generate(final_plate, violation_img)
                challan_generated = True

        # Draw boxes on current frame
        if has_violation:
            cv2.rectangle(frame, (0, 0),
                         (frame.shape[1]-1, frame.shape[0]-1),
                         (0, 0, 255), 3)
            cv2.putText(frame, "VIOLATION DETECTED",
                       (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 0, 255), 2)

        frame = detector.draw_boxes(frame, detections)
        out.write(frame)

    cap.release()
    out.release()

    print(f"\n\n✅ Video processing complete!")
    print(f"✅ Output saved as '{OUTPUT_PATH}'")
    print(f"✅ Challans saved in '{VIOLATIONS_DIR}/' folder")

    # Auto open output video
    os.startfile(OUTPUT_PATH)

if __name__ == "__main__":
    main()