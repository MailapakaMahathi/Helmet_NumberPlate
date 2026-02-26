import datetime
import os
import json

class ChallanGenerator:
    def __init__(self, output_dir="challans"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.challan_counter = 1

    def generate(self, plate_number, violation_image_path=None):
        timestamp = datetime.datetime.now()
        challan_id = f"CH{timestamp.strftime('%Y%m%d')}{self.challan_counter:04d}"
        self.challan_counter += 1

        challan = {
            "challan_id": challan_id,
            "date": timestamp.strftime("%Y-%m-%d"),
            "time": timestamp.strftime("%H:%M:%S"),
            "violation": "Riding without helmet",
            "vehicle_number": plate_number,
            "fine_amount": "Rs. 1000",
            "status": "Pending",
            "image": violation_image_path or "N/A"
        }

        # Save as JSON
        challan_file = os.path.join(self.output_dir, f"{challan_id}.json")
        with open(challan_file, 'w') as f:
            json.dump(challan, f, indent=4)

        print("\n" + "="*50)
        print("🚨 E-CHALLAN GENERATED")
        print("="*50)
        for k, v in challan.items():
            print(f"  {k.upper()}: {v}")
        print("="*50 + "\n")

        return challan