# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:03:21 2018

@author: andye
"""
import dlib
import cv2
import numpy as np
from VideoIO import VideoGet, VideoShow

path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

def get_landmarks(im):
    #first detect face
    rects = detector(im, 1)
    
    #cases for if multiple or zero faces are detected
    if len(rects) > 1:
        return np.matrix([0])
        #raise TooManyFaces
    if len(rects) == 0:
        return np.matrix([0])
        #raise NoFaces
        
    #return matrix of coordinates of facial landmarks
    return np.matrix([[pt.x, pt.y] for pt in predictor(im, rects[0]).parts()])

def get_eyes(im, landmarks):
    img = im.copy()
    
    #annotate image for demonstration purposes
    for idx, point in enumerate(landmarks[36:48]):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    
    #only get eyes if they are found   
    if len(landmarks) > 1: 
        
       #get the landmarks corresponding to the left and right eye
        right_points = landmarks[36:42]
        left_points = landmarks[42:48]
        
        rh_delta = int((right_points[3,0] - right_points[0,0])*0.2)
        lh_delta = int((left_points[3,0] - left_points[0,0])*0.2)
        rv_delta = int((right_points[5,1] - right_points[2,1])*0.2)
        lv_delta = int((left_points[5,1] - left_points[2,1])*0.2)
        
        #get a rectangular region for both eyes
        right_eye = (im[right_points[2,1]-rv_delta:right_points[5,1]+rv_delta,right_points[0,0]-rh_delta:right_points[3,0]+rh_delta])/255.
        left_eye = (im[right_points[2,1]-lv_delta:right_points[5,1]+lv_delta,left_points[0,0]-lh_delta:left_points[3,0]+lh_delta])/255.
        right_eye = cv2.resize(right_eye,(60,35))
        left_eye = cv2.resize(left_eye,(60,35))
        
        #show for demonstration purposes
        cv2.namedWindow('left',cv2.WINDOW_NORMAL)
        cv2.namedWindow('right',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('left', 180,105)
        cv2.resizeWindow('right', 180,105)
        cv2.imshow('left',left_eye)
        cv2.imshow('right',right_eye)
        
        return img, left_eye, right_eye
    
    else:
        return [img]

vid_get = VideoGet(0).start()

while 1:
    #ret, img = cap.read() #read image from webcam
    img = vid_get.frame
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #get greyscale for face detection
    landmarks = get_landmarks(img)
    with_landmarks = get_eyes(img, landmarks)
    try:
        cv2.imshow('img', with_landmarks[0])
    except:
        pass
    
    k = cv2.waitKey(30) & 0xff
    if k == 27 or vid_get.stopped:
        vid_get.stop()
        break

cv2.destroyAllWindows()
