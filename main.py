import tensorflow as tf
import torch
import os
from datetime import date
from concurrent.futures import ThreadPoolExecutor
import cv2

from face_detection import detect_faces
from face_encoding import encode_face
from database import get_student_encodings, recognize_face, mark_attendance

# ตรวจสอบ GPU
print("TensorFlow CUDA Available:", tf.config.list_physical_devices('GPU'))
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# ตั้งค่า TensorFlow ให้ใช้ GPU 0 (การ์ดจอหลัก)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.set_visible_devices(physical_devices[0], 'GPU')  # ใช้ GPU 0
else:
    print("GPU 0 not found, defaulting to CPU.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

os.makedirs("result", exist_ok=True)

video_path = "videos/VideoCut_5_min/output_segment_1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Can't Open Video")
    exit()

frame_count = 0
student_encodings = get_student_encodings()  # โหลดเพียงครั้งเดียว

def process_face(face, frame):
    encoding = encode_face(frame, face)
    if encoding is None:
        print("Skipping invalid face")
        return

    print("Face Encoding Complete")
    recognized_id = recognize_face(encoding, student_encodings)

    if recognized_id:
        print(f"Recognized Student ID: {recognized_id}")
        mark_attendance(recognized_id, date.today())
    else:
        print("Face Not Recognized")

with ThreadPoolExecutor() as executor:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video End!")
            break

        frame_count += 1

        if frame_count % 5 != 0:
            continue

        print(f"Processing Frame {frame_count}")

        frame_resized = cv2.resize(frame, (640, 360))
        faces = detect_faces(frame_resized)

        print(f"Detected {len(faces)} Faces")
        for face in faces:
            executor.submit(process_face, face, frame_resized.copy())

cap.release()
cv2.destroyAllWindows()
print("Program End...")
