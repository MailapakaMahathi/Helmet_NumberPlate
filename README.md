<img width="618" height="719" alt="image" src="https://github.com/user-attachments/assets/ec35aa32-f233-4cab-911d-7c2f2c7d9c5d" />🪖 Helmet Detection & E-Challan System
An automated traffic surveillance system that detects helmet violations on motorcycles, reads number plates using OCR, and generates e-challans for violators.

📌 Project Overview
This system uses YOLOv8 deep learning model and Computer Vision techniques to:

Detect riders on motorcycles
Identify whether the rider is wearing a helmet or not
Detect and localize the vehicle's number plate
Extract number plate text using OCR
Automatically generate an e-challan for violations

🗂️ Project Structure:
Helmet/
├── app/
│   ├── detector.py        ← YOLOv8 helmet & plate detection
│   ├── ocr_plate.py       ← EasyOCR number plate reader
│   ├── predict_image.py   ← Run detection on images
│   └── challan.py         ← E-challan generator
├── model/
│   └── best.pt            ← Trained YOLOv8 model
├── main.py                ← Run detection on video/webcam
├── violations/            ← Saved violation images
├── challans/              ← Generated e-challan JSON files
├── requirements.txt       ← Python dependencies
└── README.md

🏷️ Model Details
Detail             Info
Architecture       YOLOv8Large
Parameters         43.6 Million
Classes            4
Input Size         640x640

📊 How It Works
Input Image/Video
      ↓
YOLOv8 Detection
      ↓
Without Helmet detected?
      ↓ YES
Check if inside Rider bbox
      ↓ CONFIRMED
Number Plate Detection
      ↓
EasyOCR reads plate text
      ↓
E-Challan Generated (JSON)
      ↓
Violation image saved

📄 E-Challan Format
{
    "challan_id": "CH202602210001",
    "date": "2026-02-21",
    "time": "23:15:00",
    "violation": "Riding without helmet",
    "vehicle_number": "KL09A03439",
    "fine_amount": "Rs. 1000",
    "status": "Pending",
    "image": "violations/violation_KL09A03439.jpg"
}

🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| YOLOv8 (Ultralytics) | Object detection |
| OpenCV | Image/video processing |
| EasyOCR | Number plate text recognition |
| Python | Core programming language |
| PyTorch | Deep learning backend |

📦 Requirements
ultralytics
opencv-python
easyocr
pillow
numpy
pandas
torch
torchvision

🗃️ Dataset
Detail              Info
Source              Roboflow Universe
Total Images        20,287
Train               17,742
Valid               1,690
Test                855
Classes             4

Results
main.py
![violation_KL409A03439](https://github.com/user-attachments/assets/12f33ac9-8613-43f8-884a-b36e6d5a5762)
<img width="399" height="203" alt="image" src="https://github.com/user-attachments/assets/66d048ed-2146-4199-81aa-7cda98c89f6e" />
![violation_INS7VJ6878](https://github.com/user-attachments/assets/a7d122d7-c4f0-4a3e-abe9-60dfff768abf)
<img width="380" height="196" alt="image" src="https://github.com/user-attachments/assets/53a62f6d-13cc-4640-a563-20f48906213b" />
predict_image.py
![violation_INS7VJ6878](https://github.com/user-attachments/assets/b04dcdfa-da41-4920-9ab6-90de2b5e5d70)
<img width="504" height="768" alt="image" src="https://github.com/user-attachments/assets/116ac4be-3ec2-4d82-85e6-b30586f981ea" />









