import dlib
import cv2
import numpy as np

# โหลดโมเดล Face Recognition
facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def encode_face(frame, face):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, face)
    encoding = np.array(facerec.compute_face_descriptor(frame, shape))
    return encoding
