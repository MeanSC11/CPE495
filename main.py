import cv2
from face_detection import detect_faces
from face_encoding import encode_face
from database import get_student_encodings, recognize_face, mark_attendance
from deepface import DeepFace
from datetime import date
import torch

print("TensorFlow CUDA Available:", tf.config.list_physical_devices('GPU'))
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device.upper()}")

video_path = "videos/65012466-5-901.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Can't Open Video")
    exit()

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video End!")
        break

    frame_count += 1
    print(f"Processing Frame {frame_count}")

    faces = detect_faces(frame)
    print(f"Detected {len(faces)} Faces")

    for face in faces:
        encoding = encode_face(frame, face)
        print("Face Encoding Complete")

        student_encodings = get_student_encodings()
        print(f"Loaded {len(student_encodings)} Students From Database")

        recognized_id = recognize_face(encoding, student_encodings)

        if recognized_id:
            print(f"Recognized Student ID: {recognized_id}")
            mark_attendance(recognized_id, date.today())  # บันทึกลงฐานข้อมูล
        else:
            print("Face Not Recognized")

cap.release()
cv2.destroyAllWindows()
print("Program End...")
