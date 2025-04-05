import cv2
import numpy as np
from deepface import DeepFace

# ใช้ ArcFace สำหรับการเข้ารหัสใบหน้า
def encode_face(frame, face):
    x1, y1, x2, y2 = face

    if x1 >= x2 or y1 >= y2:
        print(f"Invalid face coordinates: {face}")
        return None

    face_crop = frame[y1:y2, x1:x2]

    if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
        print("Invalid face crop: Zero dimension")
        return None

    # ปรับขนาดใบหน้าให้เป็น 224x224
    face_resized = cv2.resize(face_crop, (224, 224))

    # บันทึกรูปใบหน้าที่ตัดไว้ในโฟลเดอร์ result/
    filename = f"images/resultFromFRAS/Retina_ArcFace-f5-e224/face_{x1}_{y1}.jpg"
    cv2.imwrite(filename, face_resized)  # บันทึกภาพที่ปรับขนาดแล้ว
    print(f"Saved face to: {filename}")

    # ใช้ ArcFace ในการสร้าง face embedding
    representation = DeepFace.represent(face_resized, model_name="ArcFace", enforce_detection=False)

    if not representation:
        print("No embedding generated")
        return None

    return np.array(representation[0]['embedding']).flatten()
