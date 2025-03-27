#import torch

#print("Torch version:", torch.__version__)  # แสดงเวอร์ชันที่ติดตั้ง
#print("CUDA available:", torch.cuda.is_available())  # ตรวจสอบว่ามองเห็น GPU หรือไม่
#print("CUDA version:", torch.version.cuda)  # แสดงเวอร์ชันของ CUDA
#print("cuDNN version:", torch.backends.cudnn.version())  # แสดงเวอร์ชันของ cuDNN

# สร้าง Tensor บน GPU
#print(f"CUDA Available: {torch.cuda.is_available()}")
#print(f"Using Device: {torch.cuda.get_device_name(0)}")

#x = torch.rand(10000, 10000, device="cuda")
#y = torch.rand(10000, 10000, device="cuda")

# คำนวณเมทริกซ์บน GPU
#result = torch.matmul(x, y)

#print("GPU Computation Complete!")


# เริ่มจับเวลา (บันทึก 45 นาที)
record_time = 45 * 60  # 45 นาที (2700 วินาที)
#start_time = time.time()
result_time = record_time/60

print(f"Recording video for {result_time} minutes...")