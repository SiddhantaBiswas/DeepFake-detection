# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:46:29 2021

@author: Siddhanta Biswas
"""

import cv2
import numpy as np
import dlib
import os
from tqdm import tqdm
from glob import glob

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")



def readImage(filename=None, directory=None, source="webcam"):
    if source == "webcam":
        _, frame = cap.read()
    elif source == "file" and filename != None:
        frame = cv2.imread(filename)
    elif source == "directory" and directory != None:
        frame = cv2.imread(filename)
    else:
        return
    return frame


def extractFacialLandmarks(img, points, scale=5, apply_mask=True):
    if apply_mask:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255,255,255))
        # mask = cv2.GaussianBlur(mask, (9,9), 10)
        img = cv2.bitwise_and(img, mask)
    
    bbox = cv2.boundingRect(points)
    x,y,w,h = bbox
    crop = img[y:y+h, x:x+w]
    crop = cv2.resize(crop, (0,0), None, scale, scale)
    return crop


def getEyePoints(landmarks):
    points = []
    for i in range(17,28):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append([x,y])
    for i in range(36,42):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append([x,y])
    for i in range(42,48):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append([x,y])
    points = np.array(points)
    return points


def getNosePoints(landmarks):
    points = []
    for i in range(27,36):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append([x,y])
    for i in range(49,54):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append([x,y])
    points = np.array(points)
    return points


def getLipsPoints(landmarks):
    points = []
    for i in range(48,61):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append([x,y])
    points = np.array(points)
    return points


# =============================================================================
# while True:
#     frame = readImage(filename="862_0002.jpg", source="file")
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     
#     for face in faces:
#         # x1 = face.left()
#         # y1 = face.top()
#         # x2 = face.right()
#         # y2 = face.bottom()
#         # cv2.rectangle(frame, (x1, y1)q, (x2, y2), (0, 255, 0), 3)
#         
#         landmarks = predictor(gray, face)
#         
#         eyes = extractFacialLandmarks(frame, getEyePoints(landmarks), apply_mask=False)
#         lips = extractFacialLandmarks(frame, getLipsPoints(landmarks), apply_mask=False)
#         nose = extractFacialLandmarks(frame, getNosePoints(landmarks), apply_mask=False)
#         cv2.imshow("Eyes", eyes)
#         cv2.imshow("Lips", lips)
#         cv2.imshow("Nose", nose)
#         cv2.imwrite("eyes.jpg", eyes)
#         cv2.imwrite("lips.jpg", lips)
#         cv2.imwrite("nose.jpg", nose)
#         
#     #cv2.imshow("Window", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     
# cap.release()
# cv2.destroyAllWindows()
# =============================================================================


def process_folder(folder, eye_size=(150, 50), lips_size=(100, 50), nose_size=(50, 100)):
    
    files = os.listdir(folder)
    
    for file in files:
        frame = cv2.imread(os.path.join(folder, file))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            # x1 = face.left()
            # y1 = face.top()
            # x2 = face.right()
            # y2 = face.bottom()
            # cv2.rectangle(frame, (x1, y1)q, (x2, y2), (0, 255, 0), 3)
            
            landmarks = predictor(gray, face)
            
            eyes = extractFacialLandmarks(frame, getEyePoints(landmarks), apply_mask=False)
            lips = extractFacialLandmarks(frame, getLipsPoints(landmarks), apply_mask=False)
            nose = extractFacialLandmarks(frame, getNosePoints(landmarks), apply_mask=False)
            
            eyes = (cv2.resize(eyes, eye_size))
            lips = (cv2.resize(lips, lips_size))
            nose = (cv2.resize(nose, nose_size))
            
            output_path = (r"D:\trial"+folder[2:])
            print(output_path)
            
            cv2.imwrite(output_path+"eyes.jpg", eyes)
            cv2.imwrite(output_path+"lips.jpg", lips)
            cv2.imwrite(output_path+"nose.jpg", nose)

root = r"D:\FaceForensics Extracted Faces\non_aligned"
directory = glob(root+'/*/*/*', recursive=True)
for folder in directory:
    files = os.listdir(folder)
print(folder)
print(files)
output_path = (r"D:\trial"+folder[2:])
print(output_path)
os.makedirs(output_path, exist_ok=True)