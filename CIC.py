import cv2
import time

# กำหนด URL RTSP
USERNAME = "MeanSC11"
PASSWORD = "Mean2003."
CAMERA_IP = "192.168.1.100"
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/stream1"
#RTSP_URL = f"rtsp://MeanSC11:Mean2003.@192.168.1.100:554/stream1"

# เปิดการเชื่อมต่อกล้อง
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("Error: Unable to connect to the camera.")
    exit()
else :
    print("Connected Camara!!")

# ตรวจสอบว่ากล้องส่งภาพมาได้หรือไม่
ret, frame = cap.read()
if not ret:
    print("Error: Failed to grab frame from camera.")
    cap.release()
    exit()

# ดึงขนาดวิดีโอจากกล้อง
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps ของวิดีโอจากกล้อง
fps = cap.get(cv2.CAP_PROP_FPS)

# ใช้ค่าเริ่มต้นหากดึงข้อมูลไม่ได้
if frame_width == 0 or frame_height == 0:
    frame_width, frame_height = 1920, 1080  # ปรับค่าให้เหมาะกับกล้องของคุณ

print(f"Connected to camera. Resolution: {frame_width}x{frame_height}")
print(f"FPS from camera: {fps}")

# ตั้งค่าการบันทึกวิดีโอ
fourcc = cv2.VideoWriter_fourcc(*"avc1")  # ใช้ mp4v ถ้า H264 มีปัญหา
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# ตรวจสอบว่า VideoWriter เปิดได้หรือไม่
if not out.isOpened():
    print("Error: Failed to initialize VideoWriter.")
    cap.release()
    exit()

# เริ่มจับเวลา
record_time = 60 # ระยะเวลาบันทึก (วินาที)
start_time = time.time()

print(f"Recording video for {record_time} seconds...")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab frame.")
        break

    out.write(frame)  # บันทึกเฟรมลงไฟล์

    # หยุดการบันทึกเมื่อครบเวลา
    if time.time() - start_time > record_time:
        break

# ปล่อยการเชื่อมต่อกล้องและปิดไฟล์
cap.release()
out.release()

print("Recording stopped. Video saved as 'output.mp4'.")
