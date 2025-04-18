# face_detection.py

import cv2
import numpy as np
import os

# โหลดโมเดล DNN สำหรับตรวจจับใบหน้า
model_file = "models/opencv_face_detector_uint8.pb"
config_file = "models/opencv_face_detector.pbtxt"

net = cv2.dnn.readNetFromTensorflow(model_file, config_file)

# ตั้งค่าให้ใช้ GPU 0 (การ์ดจอหลัก) โดยอัตโนมัติ
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    cv2.cuda.setDevice(0)  # เลือก GPU 0
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            faces.append((x1, y1, x2, y2))

    return faces
