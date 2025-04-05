import cv2
import numpy as np
from deepface import DeepFace
import os  # เพิ่มบรรทัดนี้

# โหลดโมเดล ArcFace
arcface_model = DeepFace.build_model("ArcFace")

# สร้างโฟลเดอร์ /result ถ้ายังไม่มี
os.makedirs("result", exist_ok=True)

def encode_face(frame, face):
    x1, y1, x2, y2 = face

    if x1 >= x2 or y1 >= y2:
        print(f"Invalid face coordinates: {face}")
        return None

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
        print("Invalid face crop: Zero dimension")
        return None

    # บันทึกรูปใบหน้าที่ตัดไว้ในโฟลเดอร์ result/
    filename = f"result/face_{x1}_{y1}.jpg"
    cv2.imwrite(filename, face_crop)
    print(f"Saved face to: {filename}")

    # ปรับขนาดให้ตรงกับโมเดล (112x112)
    face_resized = cv2.resize(face_crop, (112, 112))

    representation = DeepFace.represent(face_resized, model_name="ArcFace", enforce_detection=False)

    if not representation:
        print("No embedding generated")
        return None

    return np.array(representation[0]['embedding']).flatten()
