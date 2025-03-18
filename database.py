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

def save_encoding(student_id, encoding):
    encoding_bytes = encoding.tobytes()
    query = "INSERT INTO FaceEncodings (Student_id, face_encoding) VALUES (?, ?)"
    cursor.execute(query, (student_id, encoding_bytes))
    conn.commit()

def get_student_encodings():
    query = "SELECT Student_id, face_encoding FROM FaceEncodings"
    cursor.execute(query)
    students = cursor.fetchall()

    student_encodings = {}
    for student_id, encoding_bytes in students:
        encoding = np.frombuffer(encoding_bytes, dtype=np.float64)
        student_encodings[student_id] = encoding
    return student_encodings

def recognize_face(encoding, student_encodings):
    for student_id, db_encoding in student_encodings.items():
        dist = distance.euclidean(encoding, db_encoding)
        if dist < 0.6:  # ค่าความคล้ายกัน
            return student_id
    return None
