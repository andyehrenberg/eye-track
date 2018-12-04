# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:03:21 2018

@author: andye
"""
import dlib
import cv2
import numpy as np
from VideoIO import VideoGet, VideoShow
import time

path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

def get_landmarks(im, roi):
    
    #first detect face
    if roi != None:
        im = im[roi[2]:roi[3],roi[0]:roi[1]]
    else:
        pass
        #img = im.copy()
    start = time.time()
    rects = detector(im, 1)
    end = time.time()
    #print('time to get face:',end-start)
    
    
    if len(rects) == 0:
        roi = None
        return np.matrix([0]), roi
    
    i = 0
    area = rects[0].area
    for j in range(1,len(rects)):
        temp = rects[j].area
        if temp > area:
            i = j
            area = temp
            
    left = rects[i].left()
    right = rects[i].right()
    top = rects[i].top()
    bottom = rects[i].bottom()
    dx = right - left
    dy = bottom - top
    left = max(0, int(left - dx*0.2))
    right = int(right + dx*0.2)
    top = max(0, int(top - dy*0.2))
    bottom = int(bottom + dy*0.2)
    
    roi = [left, right, top, bottom]

    #return matrix of coordinates of facial landmarks
    start = time.time()
    x = np.matrix([[pt.x, pt.y] for pt in predictor(im, rects[i]).parts()])
    end = time.time()
    #print('time to get landmarks: ',end - start)
    
    return x, roi

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
        start = time.time()
        #print(landmarks)
        #get the landmarks corresponding to the left and right eye
        right_points = landmarks[36:42]
        left_points = landmarks[42:48]
        
        rv_min = min([right_points[2,1], right_points[1,1]])
        rv_max = max([right_points[4,1], right_points[5,1]])
        lv_min = min([left_points[2,1], left_points[1,1]])
        lv_max = max([left_points[4,1], left_points[5,1]])
        rh_delta = int((right_points[3,0] - right_points[0,0])*0.3)
        lh_delta = int((left_points[3,0] - left_points[0,0])*0.3)
        rv_delta = int((rv_max - rv_min)*0.3)
        lv_delta = int((lv_max - lv_min)*0.3)
        
        #get a rectangular region for both eyes, covert to greyscale
        right_eye = im[rv_min-rv_delta:rv_max+rv_delta,right_points[0,0]-rh_delta:right_points[3,0]+rh_delta]
        left_eye = im[lv_min-lv_delta:lv_max+lv_delta,left_points[0,0]-lh_delta:left_points[3,0]+lh_delta]
        right_eye = cv2.resize(right_eye,(60,36)).astype('float32')/255.
        left_eye = cv2.resize(left_eye,(60,36)).astype('float32')/255.
        right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        
        #show for demonstration purposes
        cv2.namedWindow('left',cv2.WINDOW_NORMAL)
        cv2.namedWindow('right',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('left', 120,70)
        cv2.resizeWindow('right', 120,70)
        cv2.imshow('left',left_eye)
        cv2.imshow('right',right_eye)
        end = time.time()
        #print('time to get eye images: ',end - start)
        return im, left_eye, right_eye
    
    else:
        return [im]

roi = None

vid_get = VideoGet(0).start()

while 1:
    start = time.time()
    #ret, img = cap.read() #read image from webcam
    img = vid_get.frame
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #get greyscale for face detection
    landmarks, roi_ = get_landmarks(img, roi)
    if roi_ == None or roi == None:    
        with_landmarks = get_eyes(img, landmarks)
    else:
        with_landmarks = get_eyes(img[roi[2]:roi[3],roi[0]:roi[1]], landmarks)
    s1 = time.time()
    try:
        cv2.imshow('img', with_landmarks[0])
    except:
        pass
    e1 = time.time()
    #print('time to show image:',e1 - s1)
    k = cv2.waitKey(30) & 0xff
    if k == 27 or vid_get.stopped:
        vid_get.stop()
        break
    end = time.time()
    roi = roi_
    print(1./(end - start))

cv2.destroyAllWindows()
