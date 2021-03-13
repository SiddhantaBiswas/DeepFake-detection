# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:46:29 2021

@author: Siddhanta Biswas
"""

import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def readImage(filename=None, source="webcam"):
    if source == "webcam":
        _, frame = cap.read()
    elif source == "file" and filename != None:
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


def process_batch(batch, eye_size=(300, 100), lips_size=(200, 100), nose_size=(100, 200)):
    extracted_eyes = []
    extracted_lips = []
    extracted_nose = []
    
    for frame in batch:
        frame = frame.astype(np.uint8)
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
            
            extracted_eyes.append(cv2.resize(eyes, eye_size))
            extracted_lips.append(cv2.resize(lips, lips_size))
            extracted_nose.append(cv2.resize(nose, nose_size))
            
    extracted_eyes = np.array(extracted_eyes)
    extracted_lips = np.array(extracted_lips)
    extracted_nose = np.array(extracted_nose)
    
    return extracted_eyes, extracted_lips, extracted_nose