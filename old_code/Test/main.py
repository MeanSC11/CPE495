import cv2
import tensorflow as tf
from face_detection import detect_faces
from face_encoding import encode_face
from database import get_student_encodings, recognize_face, mark_attendance
from deepface import DeepFace
from datetime import date
import torch
import os
from concurrent.futures import ThreadPoolExecutor

# ตรวจสอบ GPU
print("TensorFlow CUDA Available:", tf.config.list_physical_devices('GPU'))
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

# สร้างโฟลเดอร์ result ถ้ายังไม่มี
os.makedirs("result", exist_ok=True)

video_path = "videos/VideoCut_5_min/output_segment_1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Can't Open Video")
    exit()

frame_count = 0

# ใช้ ThreadPoolExecutor เพื่อทำงานพร้อมกัน
with ThreadPoolExecutor() as executor:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video End!")
            break

        frame_count += 1

        # ประมวลผลทุก ๆ 10 เฟรม
        if frame_count % 10 != 0:
            continue

        print(f"Processing Frame {frame_count}")

        # ส่งภาพไปตรวจจับใบหน้าในเธรดแยก
        futures = [executor.submit(detect_faces, frame)]
        faces = futures[0].result()

        print(f"Detected {len(faces)} Faces")

        for face in faces:
            encoding = encode_face(frame, face)
            if encoding is None:
                print("Skipping invalid face")
                continue

            print("Face Encoding Complete")

            student_encodings = get_student_encodings()
            print(f"Loaded {len(student_encodings)} Students From Database")

            recognized_id = recognize_face(encoding, student_encodings)

            if recognized_id:
                print(f"Recognized Student ID: {recognized_id}")
                mark_attendance(recognized_id, date.today())
            else:
                print("Face Not Recognized")

cap.release()
cv2.destroyAllWindows()
print("Program End...")
