import cv2
import numpy as np
import pyodbc
import random
import json
import torch
import tensorflow as tf
from insightface.app import FaceAnalysis
from config import DATABASE_CONFIG
from PIL import Image
import io

# 🔹 เชื่อมต่อฐานข้อมูล Azure SQL
try:
    conn = pyodbc.connect(f"DRIVER={DATABASE_CONFIG['DRIVER']};"
                          f"SERVER={DATABASE_CONFIG['SERVER']};"
                          f"DATABASE={DATABASE_CONFIG['DATABASE']};"
                          f"UID={DATABASE_CONFIG['UID']};"
                          f"PWD={DATABASE_CONFIG['PWD']};"
                          "Encrypt=yes;"
                          "TrustServerCertificate=no;"
                          "Connection Timeout=30;")
    cursor = conn.cursor()
    print("Database connected successfully.")
except Exception as e:
    print(f"Error connecting to the database: {e}")
    exit()

# 🔹 โหลดโมเดล ArcFace ใช้ TensorFlow
face_app = FaceAnalysis(name='buffalo_l', providers=['TENSORFLOW'])
face_app.prepare(ctx_id=0) 

# ตรวจสอบว่า TensorFlow ใช้ GPU ได้หรือไม่
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using GPU.")
else:
    print("TensorFlow is not using GPU.")

# 🔹 โหลดวิดีโอ
video_path = "videos/REF Face/65015916.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Can't Open Video")
    exit()

# 🔹 กรอกรหัสนักศึกษา
student_id = input("Insert StudentID: ")

# 🔹 จำนวนเฟรมทั้งหมดในวิดีโอ
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 🔹 สุ่มเฟรม 100 เฟรมจากทั้งหมด
sampled_frames = random.sample(range(frame_count), 100)  # เลือก 100 เฟรมสุ่มจากทั้งหมด

face_embeddings = []
frame_faces = []  # เพื่อเก็บเฟรมที่มีใบหน้า

# 🔹 สร้างฟังก์ชันสำหรับบันทึกภาพต้นฉบับ
def save_original_frame(frame, image_count):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    pil_img = Image.fromarray(img)
    pil_img.save(f"images/results/original_frame_{image_count}.png")  # บันทึกภาพต้นฉบับ

# 🔹 ประมวลผลเฟรมสุ่ม 100 เฟรม
for frame_no in sampled_frames:
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # เลื่อนวิดีโอไปยังเฟรมที่เลือก
        ret, frame = cap.read()

        if not ret:
            continue

        faces_detected = face_app.get(frame)
        if len(faces_detected) == 0:
            continue

        # เลือกใบหน้าที่ดีที่สุด (อาจใช้การจัดอันดับหรือเลือกใบหน้าหลัก)
        face_embedding = faces_detected[0].embedding
        face_embeddings.append(face_embedding)
        frame_faces.append((frame, face_embedding))

    except Exception as e:
        print(f"Error processing frame {frame_no}: {e}")
        continue

# 🔹 เลือก 10 รูปที่ดีที่สุดจาก 100 รูป
frame_faces_sorted = sorted(frame_faces, key=lambda x: np.linalg.norm(x[1]))  # คัดเลือกตามระยะห่างของ embedding
top_10_faces = frame_faces_sorted[:10]  # เลือก 10 รูปที่ดีที่สุด

# 🔹 สร้างและบันทึกภาพจากเฟรมที่เลือก
image_count = 1  # เปลี่ยนให้เริ่มต้นที่ 1

for frame, embedding in top_10_faces:
    # บันทึกภาพต้นฉบับ
    save_original_frame(frame, image_count)
    image_count += 1  # เพิ่มค่าหมายเลขภาพเมื่อบันทึกสำเร็จ

# 🔹 เริ่มสร้าง Embedding ทั้ง 100
face_embeddings = []

# 1. Original 10 รูป
original_embeddings = [x[1] for x in top_10_faces]
face_embeddings.extend(original_embeddings)

# 2. เพิ่มแสง 10 ระดับ
for emb in original_embeddings:
    bright = emb + np.random.normal(0.15, 0.05, emb.shape)
    face_embeddings.append(bright)

# 3. ลดแสง 10 ระดับ
for emb in original_embeddings:
    dark = emb - np.random.normal(0.15, 0.05, emb.shape)
    face_embeddings.append(dark)

# 4. เบลอ (โมเสก) 10 ระดับ
for emb in original_embeddings:
    blur = emb + np.random.normal(0, 0.2, emb.shape)
    face_embeddings.append(blur)

# 5. ภาพบางส่วนหาย 10 ระดับ
for emb in original_embeddings:
    missing = np.copy(emb)
    idx = np.random.choice(len(emb), size=len(emb) // 4, replace=False)
    missing[idx] = 0
    face_embeddings.append(missing)

# 6. ความละเอียดลดลงเล็กน้อย 10 ระดับ
for emb in original_embeddings:
    low1 = emb + np.random.normal(0, 0.05, emb.shape)
    face_embeddings.append(low1)

# 7. ความละเอียดลดลงปานกลาง 10 ระดับ
for emb in original_embeddings:
    low2 = emb + np.random.normal(0, 0.1, emb.shape)
    face_embeddings.append(low2)

# 8. ความละเอียดลดลงเยอะมาก 10 ระดับ
for emb in original_embeddings:
    low3 = emb + np.random.normal(0, 0.2, emb.shape)
    face_embeddings.append(low3)

# 9. Noise แบบอื่นๆ 20 รูป
for _ in range(20):
    emb = random.choice(original_embeddings)
    noise = emb + np.random.normal(0, 0.12, emb.shape)
    face_embeddings.append(noise)

# 🔹 ตรวจสอบว่ารวมเป็น 100 embeddings พอดี
print(f"Total embeddings generated: {len(face_embeddings)}")  # ควรเป็น 100

# 🔹 แปลงและบันทึกลงฐานข้อมูล
face_embeddings_json = json.dumps([embedding.tolist() for embedding in face_embeddings])
face_embeddings_binary = face_embeddings_json.encode('utf-8')

try:
    cursor.execute("""
        UPDATE Students
        SET face_embedding = ?
        WHERE student_id = ?
    """, (face_embeddings_binary, student_id))
    conn.commit()
    print(f"Student {student_id} has face embeddings stored successfully.")
except Exception as e:
    print(f"Error updating the database: {e}")

# 🔹 แสดงข้อความเมื่อเสร็จสิ้น
print(f"10 original images saved to the face folder.")
print(f"100 embeddings (including noise, brightness, etc.) saved.")

cap.release()
conn.close()
