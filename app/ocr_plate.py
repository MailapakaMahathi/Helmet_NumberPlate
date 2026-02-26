import cv2
import re
import easyocr

class NumberPlateReader:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        print("✅ OCR engine initialized.")

    def clean_text(self, text):
        text = text.upper()
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text

    def is_valid_plate(self, text):
        if len(text) < 4 or len(text) > 12:
            return False
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        return has_letter and has_number

    def preprocess_variants(self, img):
        variants = []
        big = cv2.resize(img, None, fx=3, fy=3,
                        interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        variants.append(gray)

        _, otsu = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(otsu)
        variants.append(cv2.bitwise_not(otsu))

        adaptive = cv2.adaptiveThreshold(gray, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        variants.append(adaptive)
        return variants

    def run_ocr(self, img):
        variants = self.preprocess_variants(img)
        all_results = []

        for variant in variants:
            results = self.reader.readtext(
                variant, detail=1,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                paragraph=False,
                width_ths=0.9,
                add_margin=0.1
            )
            combined = ""
            for res in results:
                text = self.clean_text(res[1])
                conf = res[2]
                if text and conf > 0.1:
                    combined += text
                    print(f"   OCR piece: '{text}' conf:{conf:.2f}")

            print(f"   Combined: '{combined}'")
            if combined and self.is_valid_plate(combined):
                all_results.append((combined, 0.9))

            for res in results:
                text = self.clean_text(res[1])
                conf = res[2]
                if text:
                    all_results.append((text, conf))

        all_results.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
        for text, conf in all_results:
            if self.is_valid_plate(text):
                print(f"   Best plate: '{text}'")
                return text
        return None

    def extract_plate_text(self, frame, plate_bbox=None):
        """Use number plate bbox from main model detection."""
        h, w = frame.shape[:2]

        # Step 1: Use plate bbox detected by main model
        if plate_bbox:
            x1, y1, x2, y2 = map(int, plate_bbox)
            pad = 8
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:
                print("   Running OCR on detected plate region...")
                text = self.run_ocr(crop)
                if text:
                    return text, plate_bbox

        # Step 2: Fallback — scan horizontal strips
        print("   Scanning horizontal strips...")
        strips = [
            (int(h*0.55), int(h*0.70)),
            (int(h*0.65), int(h*0.80)),
            (int(h*0.70), int(h*0.85)),
            (int(h*0.75), int(h*0.90)),
        ]
        for y_start, y_end in strips:
            x_start = int(w * 0.25)
            x_end   = int(w * 0.75)
            strip   = frame[y_start:y_end, x_start:x_end]
            if strip.size == 0:
                continue
            text = self.run_ocr(strip)
            if text:
                bbox = [x_start, y_start, x_end, y_end]
                print(f"   Strip found plate: {text}")
                return text, bbox

        return "UNREADABLE", plate_bbox