# main.py

import tensorflow as tf
import torch
import os
import time
from datetime import date
from concurrent.futures import ThreadPoolExecutor
import cv2  # ไม่ลืมเพิ่มการ import cv2

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

# Path to the file where CapIpCamTest.py will write the output video filenames
video_info_file = "video_info.txt"

# Start CapIpCamTest.py to record video
print("Starting video recording...")
os.system("python CapIpCamTest.py")

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

processed_videos = set()

print("Waiting for video segments to process...")
while True:
    if os.path.exists(video_info_file):
        with open(video_info_file, "r") as f:
            video_files = [line.strip() for line in f.readlines()]

        # Process only new video files
        for video_path in video_files:
            if video_path in processed_videos:
                continue

            print(f"Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Can't Open Video: {video_path}")
                continue

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f"Video End: {video_path}")
                    break

                frame_count += 1

                if frame_count % 5 != 0:
                    continue

                print(f"Processing Frame {frame_count} of {video_path}")

                frame_resized = cv2.resize(frame, (640, 360))
                faces = detect_faces(frame_resized)

                print(f"Detected {len(faces)} Faces")
                for face in faces:
                    process_face(face, frame_resized.copy())

            cap.release()
            processed_videos.add(video_path)

        # Check if recording is complete
        if len(processed_videos) == len(video_files) and time.time() - os.path.getmtime(video_info_file) > 10:
            print("All video segments processed. Exiting...")
            break

    time.sleep(1)

cv2.destroyAllWindows()
print("Program End...")
