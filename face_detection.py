import cv2

# ใช้ OpenCV ในการตรวจจับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces  # คืนค่าตำแหน่งใบหน้าในรูปแบบ (x, y, w, h)
