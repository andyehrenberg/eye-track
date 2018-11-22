# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:03:21 2018

@author: andye
"""
import dlib
import cv2
import numpy as np

path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

def get_landmarks(im):
    """ Returns facial landmarks from image using the shape_predictor_68_face_landmarks.dat pre-trained model
    Args:
        im: image from webcam
    
    Returns:
        A matrix of coordinates of facial landmarks
    """
    #first detect face
    rects = detector(im, 1)
    
    #cases for if multiple or zero faces are detected
    if len(rects) != 1:
        return np.matrix([0])
        
    #return matrix of coordinates of facial landmarks
    return np.matrix([[pt.x, pt.y] for pt in predictor(im, rects[0]).parts()])

def get_eyes(im, landmarks):
    """ Gets coordinates of eyes, gets rectangular region for each eye, and normalizes these regions
    Args:
        im: image from webcam
        landmarks: coordinates of facial landmarks
    
    Returns:
        If a face is found in im: im, and a normalized image of each eye. If 0 or more than one faces are found: im
    """
    
    img = im.copy()
    
    #annotate image for demonstration purposes
    for idx, point in enumerate(landmarks[36:48]):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    
    #only get eye matrices if they were found   
    if len(landmarks) > 1:    
        
        #get the landmarks corresponding to the left and right eye
        right_points = landmarks[36:42]
        left_points = landmarks[42:48]
        
        #get a rectangular region for both eyes
        right_eye = im[right_points[2,1]-4:right_points[5,1]+4,right_points[0,0]-4:right_points[3,0]+4]
        left_eye = im[right_points[2,1]-4:right_points[5,1]+4,left_points[0,0]-4:left_points[3,0]+4]
        
        #give these matrices zero mean and unit variance
        right_eye = (right_eye - right_eye.mean())/right_eye.std()
        left_eye = (left_eye - left_eye.mean())/left_eye.std()
        
        #uncomment for demonstration purposes
        #cv2.imshow('left',left_eye)
        #cv2.imshow('right',right_eye)
        
        return img, left_eye, right_eye
    
    else:
        return [img]

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read() #read image from webcam
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #get greyscale for face detection
    
    landmarks = get_landmarks(img)
    with_landmarks = get_eyes(img, landmarks)
    
    try:
        cv2.imshow('img', with_landmarks[0])
    except:
        pass
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
