import cv2
import dlib
import numpy as np
import pyodbc
from config import DATABASE_CONFIG

# โหลดโมเดลตรวจจับใบหน้าและสร้าง Encoding
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

# เชื่อมต่อฐานข้อมูล Azure SQL
conn = pyodbc.connect(f"DRIVER={DATABASE_CONFIG['DRIVER']};"
                      f"SERVER={DATABASE_CONFIG['SERVER']};"
                      f"DATABASE={DATABASE_CONFIG['DATABASE']};"
                      f"UID={DATABASE_CONFIG['UID']};"
                      f"PWD={DATABASE_CONFIG['PWD']};"
                      "Encrypt=yes;"
                      "TrustServerCertificate=no;"
                      "Connection Timeout=30;")
cursor = conn.cursor()

# โหลดภาพ
image_path = "images/65062264.jpg"  # เปลี่ยนเป็นไฟล์รูปภาพของคุณ
img = cv2.imread(image_path)

# แปลงเป็นภาพขาวดำ
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ตรวจจับใบหน้า
faces = detector(gray)

for face in faces:
    shape = sp(gray, face)
    encoding = np.array(facerec.compute_face_descriptor(img, shape))

    # กรอกรหัสนักศึกษา
    student_id = input("Insert StudentID: ")

    # ตรวจสอบว่ามี Student_id อยู่ในฐานข้อมูลหรือไม่
    query_check = "SELECT COUNT(*) FROM FaceEncodings WHERE Student_id = ?"
    cursor.execute(query_check, (student_id,))
    exists = cursor.fetchone()[0]

    encoding_bytes = encoding.tobytes()

    if exists:
        # อัปเดตค่า face_encoding ถ้ามีอยู่แล้ว
        query_update = """
        UPDATE FaceEncodings 
        SET face_encoding = ?, updated_at = GETDATE() 
        WHERE Student_id = ?
        """
        cursor.execute(query_update, (encoding_bytes, student_id))
        print(f"Update Face Encoding For Student {student_id} Complete")
    else:
        # แทรกข้อมูลใหม่ถ้ายังไม่มี
        query_insert = """
        INSERT INTO FaceEncodings (Student_id, face_encoding)
        VALUES (?, ?)
        """
        cursor.execute(query_insert, (student_id, encoding_bytes))
        print(f"Save Face Encoding For Student {student_id} Complete")

    conn.commit()

conn.close()
cv2.imshow("Face Detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
