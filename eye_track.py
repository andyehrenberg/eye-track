# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:03:21 2018

@author: andye
"""
#!/usr/bin/python

# Copyright (c) 2015 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

import dlib
import cv2
import numpy

path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path)

#class TooManyFaces(Exception):
#    pass
#
#class NoFaces(Exception):
#    pass

def get_landmarks(im):
    #first detect face
    rects = detector(im, 1)
    
    #cases for if multiple or zero faces are detected
    if len(rects) > 1:
        return numpy.matrix([0])
        #raise TooManyFaces
    if len(rects) == 0:
        return numpy.matrix([0])
        #raise NoFaces
        
    #return matrix of coordinates of facial landmarks
    return numpy.matrix([[pt.x, pt.y] for pt in predictor(im, rects[0]).parts()])

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
    
    #only get eye matrices if they were found   
    if len(landmarks) > 1:    
        
        #get the landmarks corresponding to the left and right eye
        right_points = landmarks[36:42]
        left_points = landmarks[42:48]
        
        #get a rectangular region for both eyes
        right_eye = im[right_points[2,1]:right_points[5,1],right_points[0,0]-2:right_points[3,0]+2]
        left_eye = im[right_points[2,1]:right_points[5,1],left_points[0,0]-2:left_points[3,0]+2]
        
        #give these matrices zero mean and unit variance
        right_eye = (right_eye - right_eye.mean())/right_eye.std()
        left_eye = (left_eye - left_eye.mean())/left_eye.std()
        
        #show for demonstration purposes
        cv2.imshow('left',left_eye)
        cv2.imshow('right',right_eye)
        
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








