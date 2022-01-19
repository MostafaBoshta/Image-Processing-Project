# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 10:16:56 2021

"""


import cv2 as cv
import numpy as np

img = cv.imread('Images/park.jpg')
cv.imshow('Park', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

"""
# Laplacian is used to find inward and outward edges.
    * The first step is to convert the image to a grayscale image.
# Dst = Laplacian(src, ddepth)
    * This method accepts the following parameters −
      # src − A Mat object representing the source (input image) for this operation.
      # Dst − A Mat object representing the destination (output image) for this operation.
      # ddepth − A variable of the type integer representing depth of the destination image.

"""
# Laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
#
"""
# Here, we compute the absolute because when we transition from black to white and white to
  black that is considered a positive and negative slope. The pixels of the image can not
  have negative values, so that we compute the slope.
# and after computing the absolute, we convert that to uint8 which is an image specific
  datatype.
""" 
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel
""" 
# sobelx = Compute the gradients in x-axis and it is indicated from 1, 0
# sobely = Compute the gradients in y-axis and it is indicated from 0, 1
# The last step is to combine both axis.
"""
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)

"""
Canny Edge Detection was explained in a previous section.
"""
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)
cv.waitKey(0)