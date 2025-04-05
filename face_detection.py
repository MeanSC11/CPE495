import cv2
import insightface
from insightface.app import FaceAnalysis

# โหลดโมเดล RetinaFace
face_app = FaceAnalysis()
face_app.prepare(ctx_id=0)  # ใช้ GPU ถ้ามี

def detect_faces(frame):
    # ตรวจจับใบหน้าจากวิดีโอ
    faces = face_app.get(frame)

    # เลือกใบหน้าที่ดีที่สุด (ถ้ามีหลายใบหน้า)
    detected_faces = []
    for face in faces:
        # ดึงตำแหน่งของใบหน้า (x1, y1, x2, y2)
        x1, y1, x2, y2 = face.bbox.astype(int)
        detected_faces.append((x1, y1, x2, y2))

    return detected_faces
