# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import cv2 as cv

img = cv.imread('Images/cat.jpg')
cv.imshow('Cats', img)

# Convert the image to gray scale.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

"""
# cv.threshold: The method returns two outputs. The first is the threshold that was used 
  (in example below it will return 150) and the second output is the thresholded image:
    * The first arguement to this function is the image to be binarized.
    * The second arguement is the threshold value.
    * The third arguement is a value, if the pixel intensity is larger to the threshold it will
      be set to this value.
    * The last arguement can be one of the following:
          # cv2.THRESH_BINARY: If pixel intensity is greater than the set threshold, value 
            set to 255, else set to 0 (black).
          # cv2.THRESH_BINARY_INV: Inverted or Opposite case of cv2.THRESH_BINARY.
          # cv2.THRESH_TRUNC: If pixel intensity value is greater than threshold, it is truncated to the threshold. The pixel values are set to be the same as the threshold. All other values remain the same.
          # cv2.THRESH_TOZERO: Pixel intensity is set to 0, for all the pixels intensity, 
            less than the threshold value.
          # cv2.THRESH_TOZERO_INV: Inverted or Opposite case of cv2.THRESH_TOZERO.
"""
# Simple Thresholding
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY )
cv.imshow('Simple Thresholded', thresh)
"""
# It is the inverse of the example above, because we used the fourth arguement 
  cv2.THRESH_BINARY_INV instead of cv2.THRESH_BINARY. 
# So if the pixel value is higher than 150, it will turn to 0 and if the pixel value is less
  than 150, it will turn to 1.
"""
threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV )
cv.imshow('Simple Thresholded Inverse', thresh_inv)

# Adaptive Thresholding
"""
# Sometime, choosing manually the threshold would be difficult, and doesn't give good results.
  So, instead we want the value of the threshold to be determined automatically for us. This
  is done by adaptive thresholding.
# cv.adaptiveThreshold: It returns only the output thresholded image.
      * The first arguement to this function is the image to be binarized.
      * The second arguement is a value, if the pixel intensity is larger to the threshold it will
        be set to this value.
      * The third arguement is the adptive method. It can be one of the following values:
          # cv.ADAPTIVE_THRESH_MEAN_C: The threshold value is the mean of the neighbourhood 
            area minus the constant C.
          # cv.ADAPTIVE_THRESH_GAUSSIAN_C: The threshold value is a gaussian-weighted sum of
            the neighbourhood values minus the constant C. The guassian weight, is a weight
            that is added to each pixel value.
      * The fourth arguement can be one of the following:
          # cv2.THRESH_BINARY: If pixel intensity is greater than the set threshold, value 
            set to 255, else set to 0 (black).
          # cv2.THRESH_BINARY_INV: Inverted or Opposite case of cv2.THRESH_BINARY.
          # cv2.THRESH_TRUNC: If pixel intensity value is greater than threshold, it is truncated to the threshold. The pixel values are set to be the same as the threshold. All other values remain the same.
          # cv2.THRESH_TOZERO: Pixel intensity is set to 0, for all the pixels intensity, 
            less than the threshold value.
          # cv2.THRESH_TOZERO_INV: Inverted or Opposite case of cv2.THRESH_TOZERO.
      * The fifth parameter is the block size for neighbourhood: The blockSize determines
        the size of the neighbourhood area (to compute for example the mean to define 
        the optimal value). For ex/ if the block size is 11, itmeans 11x11.
      * The sixth parameter is the C parameter: is a constant that is subtracted from
        the mean or weighted sum of the neighbourhood pixels.
"""
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 9)
cv.imshow('Adaptive Thresholding', adaptive_thresh)

cv.waitKey(0)