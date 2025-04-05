#database.py

import pyodbc
import numpy as np
from config import DATABASE_CONFIG
from scipy.spatial import distance
from datetime import date

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

# 🔹 บันทึกข้อมูลใบหน้าเข้าไปใน Students
def save_encoding(student_id, encoding):
    encoding_bytes = encoding.tobytes()
    query = "UPDATE Students SET face_embedding = ? WHERE student_id = ?"
    cursor.execute(query, (encoding_bytes, student_id))
    conn.commit()

# 🔹 ดึงข้อมูล Face Encoding ของนักศึกษาทั้งหมด
def get_student_encodings():
    query = "SELECT student_id, face_embedding FROM Students WHERE face_embedding IS NOT NULL"
    print(f"Executing SQL Query: {query}")  # Debug log
    
    cursor.execute(query)
    students = cursor.fetchall()
    
    print(f"Found {len(students)} students with encodings")  # Debug log
    
    student_encodings = {}
    for student_id, encoding_bytes in students:
        encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
        student_encodings[student_id] = encoding
    
    return student_encodings

# 🔹 ค้นหาใบหน้าที่ตรงกับฐานข้อมูล
def recognize_face(encoding, student_encodings):
    for student_id, db_encoding in student_encodings.items():
        dist = distance.euclidean(encoding, db_encoding)
        print(f"Comparing {student_id}: Distance = {dist}")  # Debug log
        if dist < 0.6:  # ค่าความคล้ายกัน
            return student_id
    return None

# 🔹 อัปเดตการเช็คชื่อใน Attendance
def mark_attendance(student_id, check_date):
    query = """
    INSERT INTO Attendance (course_id, course_name, student_id, first_name, last_name, date_check, status_student)
    SELECT e.course_id, c.course_name, s.student_id, s.student_name, '', ?, '00'
    FROM Students s
    JOIN Enrollment e ON s.student_id = e.student_id
    JOIN Courses c ON e.course_id = c.course_id
    WHERE s.student_id = ?
    """
    cursor.execute(query, check_date, student_id)
    conn.commit()
    print(f"Attendance Recorded for Student {student_id} on {check_date}")
