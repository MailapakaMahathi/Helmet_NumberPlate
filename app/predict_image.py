import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.detector import HelmetDetector
from app.ocr_plate import NumberPlateReader
from app.challan import ChallanGenerator

# ─── CONFIG ───────────────────────────────────
MODEL_PATH  = "model/best.pt"
IMAGE_PATH  = "test4.jpg"
OUTPUT_PATH = "output.jpg"
# ──────────────────────────────────────────────

def main():
    print("Loading model...")
    detector = HelmetDetector(MODEL_PATH)

    print("Initializing OCR...")
    ocr = NumberPlateReader()

    challan_gen = ChallanGenerator()
    os.makedirs("violations", exist_ok=True)

    frame = cv2.imread(IMAGE_PATH)
    if frame is None:
        print(f"❌ Cannot read image: {IMAGE_PATH}")
        return

    print(f"✅ Image loaded: {IMAGE_PATH}")

    # Step 1: Detect all objects
    detections = detector.detect(frame)
    print(f"✅ Detections: {[d['class'] for d in detections]}")

    # Step 2: Separate detections by class
    riders     = [d for d in detections if d['class'].lower() == 'rider']
    no_helmets = [d for d in detections if d['class'].lower() == 'without helmet']
    plates     = [d for d in detections if d['class'].lower() == 'number plate']

    plate_bbox    = plates[0]['bbox'] if plates else None
    has_violation = False

    if plates:
        print(f"✅ Number plate detected by model!")

    # Step 3: Only flag violation if without helmet is INSIDE a rider bbox
    for nh in no_helmets:
        nx1, ny1, nx2, ny2 = map(int, nh['bbox'])
        nh_cx = (nx1 + nx2) // 2
        nh_cy = (ny1 + ny2) // 2

        for rider in riders:
            rx1, ry1, rx2, ry2 = map(int, rider['bbox'])
            if rx1 < nh_cx < rx2 and ry1 < nh_cy < ry2:
                has_violation = True
                print(f"   🚨 Rider without helmet confirmed!")
                break
        if has_violation:
            break

    # If no rider detected but without helmet found still flag
    if no_helmets and not riders:
        has_violation = True
        print(f"   🚨 Without helmet detected (no rider box found)!")

    # Step 4: Run OCR if violation found
    if has_violation:
        print("🚨 Violation found! Running OCR...")
        plate_text, plate_bbox = ocr.extract_plate_text(frame, plate_bbox)
        print(f"🚨 VIOLATION | Plate: {plate_text}")

        # Draw plate bounding box
        if plate_bbox:
            x1, y1, x2, y2 = map(int, plate_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            label = f"PLATE: {plate_text}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame,
                         (x1, y2 + 5),
                         (x1 + text_size[0] + 10, y2 + 35),
                         (0, 255, 255), -1)
            cv2.putText(frame, label,
                       (x1 + 5, y2 + 28),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 0, 0), 2)

        # Draw red border
        cv2.rectangle(frame, (0, 0),
                     (frame.shape[1]-1, frame.shape[0]-1),
                     (0, 0, 255), 5)
        cv2.putText(frame, "VIOLATION DETECTED",
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                   0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"Plate: {plate_text}",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                   0.9, (0, 0, 255), 2)

        # Save violation image and generate challan
        violation_img = f"violations/violation_{plate_text}.jpg"
        cv2.imwrite(violation_img, frame)
        challan_gen.generate(plate_text, violation_img)

    else:
        print("✅ No violation detected.")
        cv2.putText(frame, "NO VIOLATION",
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                   0.9, (0, 255, 0), 2)

    # Step 5: Draw all bounding boxes
    frame = detector.draw_boxes(frame, detections)

    # Step 6: Save and show output
    cv2.imwrite(OUTPUT_PATH, frame)
    from PIL import Image
    Image.open(OUTPUT_PATH).show()
    print(f"\n✅ Output saved as '{OUTPUT_PATH}'")

if __name__ == "__main__":
    main()