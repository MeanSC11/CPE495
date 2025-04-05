#Create Face Encoding Referance
import cv2
import numpy as np
import pyodbc
import random
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis  # ใช้ ArcFace
from config import DATABASE_CONFIG
from PIL import Image
import io
import json  # นำเข้าไลบรารี json เพื่อแปลงข้อมูลเป็น JSON

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
face_app.prepare(ctx_id=0)  # ใช้ GPU ถ้ามี

# 🔹 โหลดวิดีโอ
video_path = "videos/REF Face/65010219.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Can't Open Video")
    exit()

# 🔹 กรอกรหัสนักศึกษา
student_id = input("Insert StudentID: ")

# 🔹 เลือกเฟรมที่กระจายจากวิดีโอ
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
selected_frames = sorted([int(i * frame_count / 10) for i in range(10)])  # เลือกเฟรมที่กระจาย

face_embeddings = []

# 🔹 สร้างฟังก์ชันสำหรับบันทึกภาพต้นฉบับ
def save_original_frame(frame, image_count):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    pil_img = Image.fromarray(img)
    pil_img.save(f"images/results/original_frame_{image_count}.png")  # บันทึกภาพต้นฉบับ

# 🔹 สร้างและบันทึกภาพจากเฟรมต้นฉบับ 10 เฟรม
image_count = 1  # เปลี่ยนให้เริ่มต้นที่ 1
processed_frames = 0  # ใช้นับจำนวนเฟรมที่ประมวลผลสำเร็จ

for idx, frame_no in enumerate(selected_frames):
    while processed_frames < 10:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)  # เลื่อนวิดีโอไปยังเฟรมที่เลือก
            ret, frame = cap.read()

            if not ret:
                print(f"Skipped Frame {frame_no}: Failed to read frame, trying next.")
                frame_no += 1  # เลือกเฟรมถัดไป
                if frame_no >= frame_count:
                    print("Reached end of video.")
                    break
                continue  # ข้ามเฟรมที่ไม่สามารถอ่านได้

            print(f"Processing Frame {frame_no}")

            # 🔹 ตรวจจับใบหน้า
            faces_detected = face_app.get(frame)
            if len(faces_detected) == 0:
                print(f"Skipped Frame {frame_no}: No face detected.")
                frame_no += 1  # เลือกเฟรมถัดไป
                if frame_no >= frame_count:
                    print("Reached end of video.")
                    break
                continue  # ข้ามเฟรมที่ไม่มีใบหน้า

            # เลือกใบหน้าหรือใบหน้าที่ดีที่สุด
            face_embedding = faces_detected[0].embedding
            face_embeddings.append(face_embedding)

            # 🔹 บันทึกภาพต้นฉบับ
            save_original_frame(frame, image_count)
            image_count += 1  # เพิ่มค่าหมายเลขภาพเมื่อบันทึกสำเร็จ

            processed_frames += 1  # นับเฟรมที่ประมวลผลสำเร็จ

            # 🔹 เพิ่มแสง
            for _ in range(10):  
                bright_embedding = face_embedding + np.random.normal(0.1, 0.1, face_embedding.shape)
                face_embeddings.append(bright_embedding)  # บันทึก embedding ที่มีแสงเพิ่ม

            # 🔹 ลดแสง
            for _ in range(10):  
                dark_embedding = face_embedding - np.random.normal(0.1, 0.1, face_embedding.shape)
                face_embeddings.append(dark_embedding)  # บันทึก embedding ที่มีแสงลด

            # 🔹 เบลอภาพ
            for _ in range(10):  
                blur_embedding = np.copy(face_embedding)  # เบลอภาพโดยการทำให้ค่าของ embedding เปลี่ยนไป
                blur_embedding += np.random.normal(0, 0.1, blur_embedding.shape)
                face_embeddings.append(blur_embedding)  # บันทึก embedding ที่เบลอ

            # 🔹 ภาพหาย
            for _ in range(10):  
                missing_embedding = face_embedding + np.random.normal(0, 0.15, face_embedding.shape)  # ทำให้มีการหายบางข้อมูล
                face_embeddings.append(missing_embedding)  # บันทึก embedding ที่ขาดหาย

            # 🔹 เพิ่ม Noise แบบอื่นๆ
            for _ in range(50):  
                modified_embedding = face_embedding + np.random.normal(0, 0.1, face_embedding.shape)
                face_embeddings.append(modified_embedding)  # บันทึก embedding ที่มี noise

            # 🔹 แปลง `face_embeddings` ทั้งหมดเป็น JSON
            face_embeddings_json = json.dumps([embedding.tolist() for embedding in face_embeddings])

            # 🔹 บันทึก face_embeddings ลงในฐานข้อมูล
            cursor.execute("""
                UPDATE Students
                SET face_embedding = ?
                WHERE student_id = ?
            """, (face_embeddings_json, student_id))
            conn.commit()

            print(f"Student {student_id} has face embeddings stored")

            break  # หากการประมวลผลเฟรมสำเร็จ ให้หยุดลูปนี้และไปที่เฟรมถัดไป

        except Exception as e:
            print(f"Error processing frame {frame_no}: {e}")
            frame_no += 1  # เลือกเฟรมถัดไปในกรณีที่มีข้อผิดพลาด
            if frame_no >= frame_count:
                print("Reached end of video.")
                break
            continue  # ข้ามกรณีที่มีข้อผิดพลาดแล้วไปที่เฟรมถัดไป

# 🔹 แสดงข้อความเมื่อเสร็จสิ้น
print(f"10 original images saved to the face folder.")
print(f"100 embeddings (including noise, brightness, etc.) saved.")
print(f"Student {student_id} has face embeddings stored.")

cap.release()
conn.close()
