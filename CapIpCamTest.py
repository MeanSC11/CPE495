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
start_hour = 11
start_minute = 45

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
out = cv2.VideoWriter("full_record.mp4", fourcc, fps, (frame_width, frame_height))

# เริ่มจับเวลา (บันทึก 45 นาที)
record_time = 45 * 60  # 45 นาที (2700 วินาที)
start_time = time.time()
result_time = record_time/60

print(f"Recording video for {result_time} minutes...")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # คำนวณเวลานับถอยหลัง
    elapsed_time = time.time() - start_time
    remaining_time = max(0, record_time - elapsed_time)
    minutes = int(remaining_time // 60)
    seconds = int(remaining_time % 60)
    countdown_text = f"Time Left: {minutes:02}:{seconds:02}"

    # เพิ่มข้อความเวลานับถอยหลังลงบนเฟรม
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, countdown_text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    out.write(frame)

    if remaining_time <= 0:
        break

cap.release()
out.release()
print("Recording completed. Video saved as 'full_record.mp4'.")

# *** แบ่งไฟล์ออกเป็น 30 นาที และ 15 นาที ***
print("Splitting video into 30-minute and 15-minute parts...")

# โหลดวิดีโอที่บันทึกเสร็จ
cap = cv2.VideoCapture("full_record.mp4")

# คำนวณจำนวนเฟรม
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_30min = int(fps * 30 * 60)
frame_15min = total_frames - frame_30min

# ฟังก์ชันตัดวิดีโอ
def split_video(start_frame, end_frame, output_name):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out = cv2.VideoWriter(output_name, fourcc, fps, (frame_width, frame_height))
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    out.release()

# ตัดคลิป
split_video(0, frame_30min, "part1_30min.mp4")
split_video(frame_30min, total_frames, "part2_15min.mp4")

cap.release()
out.release()
print("Splitting completed: 'part1_30min.mp4' and 'part2_15min.mp4'.")