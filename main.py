import cv2
from face_detection import detect_faces
from face_encoding import encode_face
from database import save_encoding, get_student_encodings, recognize_face

# โหลดวิดีโอ
video_path = "videos/65012466-5-901.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Can't Open Video")
    exit()

frame_count = 0  # นับจำนวนเฟรม

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video End!")
        break

    frame_count += 1
    print(f"Processing Fram {frame_count}")

    # ตรวจจับใบหน้า
    faces = detect_faces(frame)
    print(f"Face Detection {len(faces)} Person")

    for face in faces:
        encoding = encode_face(frame, face)
        print("Face Encoding Complete")

        # ค้นหาใบหน้าที่มีในฐานข้อมูล
        student_encodings = get_student_encodings()
        print(f"Load Student Data {len(student_encodings)} Person From Database")

        recognized_id = recognize_face(encoding, student_encodings)

        if recognized_id:
            print(f"Face Detection Student: {recognized_id}")
        else:
            print("Can't Face Detection Student")


cap.release()
cv2.destroyAllWindows()
print("Program End...")
