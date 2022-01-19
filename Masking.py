# -*- coding: utf-8 -*-
"""
Created on Sun May 23 16:31:08 2021

"""

#pylint:disable=no-member
"""
Masking allows you to focus on certain parts of images that we want to focus on, for example
focusing on faces of people on an image.
"""
import cv2 as cv
import numpy as np

img = cv.imread('Images/cats 2.jpg')
cv.imshow('Cats', img)

#The mask has to be the same size of the image to be applied on.
blank = np.zeros(img.shape[:2], dtype='uint8') # this is the mask
cv.imshow('Blank Image', blank)

"""
Draw a circle on a copy of the blank image, give it its center, radius (100), color (255) 
and thickness (-1).
"""
circle = cv.circle(blank.copy(), (img.shape[1]//2 + 45,img.shape[0]//2), 100, 255, -1)
"""
Draw a recatangle on a copy of the blank
"""
rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)

"""
Merge the two shapes using the bitwise AND, and form a weird mask.
"""
weird_shape = cv.bitwise_and(circle,rectangle)
cv.imshow('Weird Shape', weird_shape)

"""
The mask is applied by applying bitwise_and between the original image and the created mask.
"""
masked = cv.bitwise_and(img,img,mask=weird_shape)
cv.imshow('Weird Shaped Masked Image', masked)

cv.waitKey(0)