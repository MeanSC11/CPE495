import pyodbc
import numpy as np
from config import DATABASE_CONFIG
from scipy.spatial import distance

# ตั้งค่าการเชื่อมต่อ
conn = pyodbc.connect(f"DRIVER={DATABASE_CONFIG['DRIVER']};"
                      f"SERVER={DATABASE_CONFIG['SERVER']};"
                      f"DATABASE={DATABASE_CONFIG['DATABASE']};"
                      f"UID={DATABASE_CONFIG['UID']};"
                      f"PWD={DATABASE_CONFIG['PWD']};"
                      "Encrypt=yes;"
                      "TrustServerCertificate=no;"
                      "Connection Timeout=30;")
cursor = conn.cursor()

# ดึงข้อมูล face_embedding ของนักศึกษาทั้งหมด
def get_student_encodings():
    query = "SELECT student_id, face_embedding FROM Students WHERE face_embedding IS NOT NULL"
    print(f"Executing SQL Query: {query}")  # Debug log
    
    cursor.execute(query)
    students = cursor.fetchall()
    
    print(f"Found {len(students)} students with encodings")  # Debug log
    
    student_encodings = {}
    for student_id, encoding_bytes in students:
        try:
            # ตรวจสอบขนาดของข้อมูลในฐานข้อมูล
            print(f"Encoding Bytes Length: {len(encoding_bytes)}")  # Debug log
            
            # ตรวจสอบว่าขนาดของ encoding_bytes สามารถแปลงเป็น np.float64 ได้หรือไม่
            encoding = np.frombuffer(encoding_bytes, dtype=np.float32)  # เปลี่ยนเป็น np.float32 ตามที่โมเดลต้องการ
            
            # ตรวจสอบขนาดของ np.array ที่แปลงได้
            if encoding.size != 512:  # เปลี่ยนขนาดนี้ตามที่ใช้ในโมเดลของคุณ (เช่น 512)
                print(f"Invalid encoding size: {encoding.size}")
                continue
            
            student_encodings[student_id] = encoding
        except Exception as e:
            print(f"Error processing student {student_id}: {e}")
    
    return student_encodings

# ค้นหาใบหน้าที่ตรงกับฐานข้อมูล
def recognize_face(encoding, student_encodings):
    for student_id, db_encoding in student_encodings.items():
        dist = distance.euclidean(encoding, db_encoding)
        print(f"Comparing {student_id}: Distance = {dist}")  # Debug log
        if dist < 0.6:  # ค่าความคล้ายกัน
            return student_id
    return None

# เพิ่มฟังก์ชัน mark_attendance ใน database.py

def mark_attendance(student_id, course_id, date):
    try:
        # เพิ่มข้อมูลการเข้าเรียนในตาราง Attendance
        query = """INSERT INTO Attendance (student_id, course_id, attendance_date) 
                   VALUES (?, ?, ?)"""
        cursor.execute(query, (student_id, course_id, date))
        conn.commit()
        print(f"Attendance for student {student_id} marked successfully.")
    except Exception as e:
        print(f"Error marking attendance for student {student_id}: {e}")
