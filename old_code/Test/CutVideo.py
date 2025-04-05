import os
import subprocess

# ฟังก์ชันตัดแบ่งวิดีโอ
def split_video(input_video, output_folder, segment_duration=300):
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs(output_folder, exist_ok=True)

    # หาความยาวของวิดีโอ (ในวินาที)
    probe = subprocess.Popen(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_video],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    duration, _ = probe.communicate()
    duration = float(duration)

    # คำนวณจำนวนการตัด (รวม segment สุดท้ายด้วย ถ้ามี)
    num_segments = int(duration // segment_duration) + (1 if duration % segment_duration > 0 else 0)
    
    for i in range(num_segments):
        start_time = i * segment_duration
        current_duration = min(segment_duration, duration - start_time)
        output_file = f"{output_folder}/output_segment_{i+1}.mp4"
        
        # สั่งให้ FFmpeg ตัดวิดีโอ (ใช้ accurate seek)
        command = [
            "ffmpeg", 
            "-i", input_video,             # <-- ใส่ -i ก่อน -ss
            "-ss", str(start_time),        # <-- ใช้ accurate seek
            "-t", str(current_duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-strict", "experimental",
            "-y",  # overwrite ถ้ามีไฟล์เดิม
            output_file
        ]
        
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Segment {i+1} saved as {output_file}")

# การใช้งานฟังก์ชัน
input_video = "full_record.mp4"
output_folder = "output_segments"
segment_duration = 300  # 5 นาที

split_video(input_video, output_folder, segment_duration)
