import cv2
import numpy as np
import torch
from deepface import DeepFace

# โหลดโมเดล ArcFace
arcface_model = DeepFace.build_model("ArcFace")  # ไม่ต้องใช้ .to(device)

def encode_face(frame, face):
    x1, y1, x2, y2 = face
    face_crop = frame[y1:y2, x1:x2]

    # ใช้ DeepFace RetinaFace ดึงใบหน้า
    face_aligned = DeepFace.extract_faces(frame, detector_backend="retinaface", enforce_detection=False)[0]["face"]

    # ปรับขนาดภาพให้ตรงกับโมเดล (112x112)
    face_resized = cv2.resize(face_aligned, (112, 112))

    # แปลงภาพเป็นเวกเตอร์ 512 มิติ
    face_tensor = np.expand_dims(face_resized, axis=0)  # เพิ่มมิติให้ตรงกับโมเดล
    embedding = arcface_model.predict(face_tensor)[0]  # ใช้ predict() แทน

    return embedding
