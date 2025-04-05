import cv2
import numpy as np
import pyodbc
import random
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis  # ใช้ ArcFace
from config import DATABASE_CONFIG
from PIL import Image
import io

# 🔹 เชื่อมต่อฐานข้อมูล Azure SQL
conn = pyodbc.connect(f"DRIVER={DATABASE_CONFIG['DRIVER']};"
                      f"SERVER={DATABASE_CONFIG['SERVER']};"
                      f"DATABASE={DATABASE_CONFIG['DATABASE']};"
                      f"UID={DATABASE_CONFIG['UID']};"
                      f"PWD={DATABASE_CONFIG['PWD']};"
                      "Encrypt=yes;"
                      "TrustServerCertificate=no;"
                      "Connection Timeout=30;")
cursor = conn.cursor()

# 🔹 โหลดโมเดล ArcFace (ใช้ GPU ถ้ามี)
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0)  # ลบ nms ออก

# 🔹 โหลดวิดีโอ
video_path = "videos/REF Face/65015916.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Can't Open Video")
    exit()

# 🔹 กรอกรหัสนักศึกษา
student_id = input("Insert StudentID: ")

# เลือกเฟรมสุ่ม 10 ภาพจากวิดีโอ
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
selected_frames = sorted(random.sample(range(frame_count), 10))

face_embeddings = []

# 🔹 ฟังก์ชันเพื่อแปลง embedding เป็นภาพ (ใช้ Heatmap แทนการ reshape)
def embedding_to_image(embedding):
    # ใช้ Heatmap แสดงผล embedding ขนาด 512
    embedding_image = np.reshape(embedding, (16, 32))  # ปรับให้เหมาะสมกับขนาดของข้อมูล (16x32 หรือขนาดอื่น ๆ ที่เหมาะสม)
    plt.imshow(embedding_image, cmap='hot', interpolation='nearest')
    plt.axis('off')  # ไม่แสดงแกน x, y
    # แปลง matplotlib plot เป็นภาพที่บันทึกได้
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_img = Image.open(buf)
    return pil_img

# 🔹 สร้างและบันทึกภาพจาก embeddings
image_count = 0
for idx, frame_no in enumerate(selected_frames):
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # เลื่อนวิดีโอไปยังเฟรมที่เลือก
        ret, frame = cap.read()

        if not ret:
            print(f"Skipped Frame {frame_no}: Failed to read frame.")
            continue

        print(f"Processing Frame {frame_no}")

        # 🔹 ตรวจจับใบหน้า
        faces_detected = face_app.get(frame)
        if len(faces_detected) == 0:
            print(f"Skipped Frame {frame_no}: No face detected.")
            continue

        # เลือกใบหน้าหรือใบหน้าที่ดีที่สุด (ใกล้เคียงที่สุดกับ student_id หรือที่อยู่ในตำแหน่งที่ต้องการ)
        face_embedding = faces_detected[0].embedding  # หากพบหลายใบหน้า สามารถเลือกใบหน้าที่ต้องการที่เหมาะสม
        face_embeddings.append(face_embedding)

        # 🔹 เพิ่ม Noise และสร้าง 100 ภาพ
        for _ in range(9):  # ทำให้ครบ 100 ภาพ
            noise = np.random.normal(0, 0.05, face_embedding.shape)  # เพิ่ม Noise
            noisy_embedding = face_embedding + noise

            # แปลง embeddings เป็นภาพแล้วบันทึก
            img = embedding_to_image(noisy_embedding)
            img.save(f"images/results/embedding_{image_count}.png")  # บันทึกไฟล์เป็นภาพ
            image_count += 1

            # เพิ่มความหลากหลาย เช่น เพิ่มแสง (Brightness Adjustment)
            bright_embedding = face_embedding * (1 + random.uniform(-0.1, 0.1))  # เพิ่ม/ลดความสว่าง
            img = embedding_to_image(bright_embedding)
            img.save(f"images/results/embedding_{image_count}.png")  # บันทึกไฟล์เป็นภาพ
            image_count += 1

            # เพิ่มการเบลอเล็กน้อย (Blur effect)
            blur_embedding = np.roll(face_embedding, shift=random.randint(1, 5))  # การหมุนค่าเพื่อสร้าง effect
            img = embedding_to_image(blur_embedding)
            img.save(f"images/results/embedding_{image_count}.png")  # บันทึกไฟล์เป็นภาพ
            image_count += 1

    except Exception as e:
        print(f"Error processing frame {frame_no}: {e}")
        continue

# 🔹 แสดงข้อความเมื่อเสร็จสิ้น
print(f"100 images saved to the images folder.")

cap.release()
conn.close()
