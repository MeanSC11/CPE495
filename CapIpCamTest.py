#CapIpCamTest.py (Cap Video From RTSP of IP Camera)

import cv2
import time
from datetime import datetime

# กำหนด URL RTSP
USERNAME = "MeanSC11"
PASSWORD = "Mean2003."
CAMERA_IP = "192.168.1.100"
RTSP_URL = "rtsp://MeanSC11:Mean2003.@192.168.1.100:554/stream1"

# เวลาที่ต้องการเริ่มบันทึก
start_hour = 8
start_minute = 45

print(f"Starting recording at {start_hour}:{start_minute}")

# เปิดการเชื่อมต่อกล้อง
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("Error: Unable to connect to the camera.")
    exit()
else:
    print("Connected to Camera!!")

# ตรวจสอบว่ากล้องส่งภาพมาได้หรือไม่
ret, frame = cap.read()
if not ret:
    print("Error: Failed to grab frame from camera.")
    cap.release()
    exit()

# รอจนถึงเวลาเริ่ม
while True:
    now = datetime.now()
    if now.hour == start_hour and now.minute == start_minute:
        print(f"Starting recording at {start_hour}:{start_minute}")
        break
    time.sleep(1)  # เช็คทุก 1 วินาที

# ตรวจสอบอีกครั้งว่ากล้องยังส่งภาพมาได้อยู่
ret, frame = cap.read()
if not ret:
    print("Error: Failed to grab frame from camera.")
    cap.release()
    exit()

# ดึงขนาดวิดีโอจากกล้อง
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ใช้ค่าเริ่มต้นหากดึงข้อมูลไม่ได้
if frame_width == 0 or frame_height == 0:
    frame_width, frame_height = 1920, 1080  # ปรับค่าให้เหมาะกับกล้องของคุณ

print(f"Connected to camera. Resolution: {frame_width}x{frame_height}")
print(f"FPS from camera: {fps}")

# ตั้งค่าการบันทึกวิดีโอ
fourcc = cv2.VideoWriter_fourcc(*"avc1")  # ใช้ mp4v ถ้า H264 มีปัญหา
segment_duration = 5 * 60  # 5 นาที (300 วินาที)
total_duration = 45 * 60  # 45 นาที (2700 วินาที)
start_time = time.time()

print(f"Recording video in 5-minute segments for a total of {total_duration / 60} minutes...")

segment_count = 0
while time.time() - start_time < total_duration:
    segment_start_time = time.time()
    segment_filename = f"segment_{segment_count + 1}.mp4"
    out = cv2.VideoWriter(segment_filename, fourcc, fps, (frame_width, frame_height))

    print(f"Recording segment {segment_count + 1}: {segment_filename}")
    while time.time() - segment_start_time < segment_duration:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # เพิ่มข้อความเวลานับถอยหลังลงบนเฟรม
        elapsed_time = time.time() - segment_start_time
        remaining_time = max(0, segment_duration - elapsed_time)
        minutes = int(remaining_time // 60)
        seconds = int(remaining_time % 60)
        countdown_text = f"Time Left: {minutes:02}:{seconds:02}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, countdown_text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Segment {segment_count + 1} completed: {segment_filename}")

    # Notify main.py by appending the segment filename to video_info.txt
    with open("video_info.txt", "a") as f:
        f.write(f"{segment_filename}\n")

    segment_count += 1

print("All segments recorded.")
cap.release()
print("Camera released.")