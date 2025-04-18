# face_encoding.py

import cv2
import numpy as np
from deepface import DeepFace
import os
import tensorflow as tf

# ตั้งค่าให้ใช้ GPU 0 ใน TensorFlow
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')  # ใช้ GPU 0
else:
    print("GPU 0 not found, defaulting to CPU.")

arcface_model = DeepFace.build_model("ArcFace")

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

    # ปรับขนาดก่อนที่จะบันทึก
    face_resized = cv2.resize(face_crop, (300, 300))  # ปรับขนาดที่นี่

    # บันทึกรูปใบหน้าที่ตัดไว้ในโฟลเดอร์ result/
    filename = f"images/resultFromFRAS/test-30min/face_{x1}_{y1}.jpg"
    cv2.imwrite(filename, face_resized)  # บันทึกภาพที่ปรับขนาดแล้ว
    print(f"Saved face to: {filename}")

    representation = DeepFace.represent(face_resized, model_name="ArcFace", enforce_detection=False)

    if not representation:
        print("No embedding generated")
        return None

    return np.array(representation[0]['embedding']).flatten()
