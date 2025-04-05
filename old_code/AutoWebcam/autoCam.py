import cv2
import time
import schedule
import threading
import datetime

# ตั้งค่ากล้อง
FRAME_RATE = 20.0
FRAME_SIZE = (640, 480)
VIDEO_DURATION = 10  # ระยะเวลาถ่ายวิดีโอ (วินาที)

def record_video():
    print("Start Recording...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Can't Open Webcam")
        return
    
    # ตั้งชื่อไฟล์ตาม timestamp
    timestamp = int(time.time())
    filename_avi = f"record_{timestamp}.avi"
    filename_mp4 = f"record_{timestamp}.mp4"

    # ใช้ codec ที่รองรับ .avi และ .mp4
    fourcc_avi = cv2.VideoWriter_fourcc(*"XVID")
    fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")

    # สร้าง VideoWriter สำหรับแต่ละไฟล์
    out_avi = cv2.VideoWriter(filename_avi, fourcc_avi, FRAME_RATE, FRAME_SIZE)
    out_mp4 = cv2.VideoWriter(filename_mp4, fourcc_mp4, FRAME_RATE, FRAME_SIZE)

    start_time = time.time()

    while time.time() - start_time < VIDEO_DURATION:
        ret, frame = cap.read()
        if not ret:
            break
        out_avi.write(frame)
        out_mp4.write(frame)
        cv2.imshow("Recording...", frame)

        # กด 'q' เพื่อหยุดก่อนเวลา
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_avi.release()
    out_mp4.release()
    cv2.destroyAllWindows()
    print(f"Record Done!: {filename_avi}, {filename_mp4}")

# ตั้งเวลาถ่ายวิดีโอเป็น 1 นาทีข้างหน้า
now = datetime.datetime.now()
next_minute = (now + datetime.timedelta(minutes=1)).strftime("%H:%M")
schedule.every().day.at(next_minute).do(record_video)

print(f"Setting Time {next_minute}")

def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

# ใช้ Thread แยกการทำงานของ schedule
threading.Thread(target=run_schedule, daemon=True).start()

# ทำให้โปรแกรมรันตลอดเวลา พร้อมปิดโปรแกรมได้สวยงาม
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("\n Program stopped by user!")
